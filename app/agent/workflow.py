"""
Grafo principal (workflow) do agente de vendas CRM.

Arquitetura:
- Parsing de entrada e classificação de intenção (LLM + Pydantic)
- Planejamento multi-intents com deduplicação de ações
- Roteamento condicional para executores especializados
- Subgrafos baseados em ReAct para operações complexas (leads, propostas)
- Geração de resposta unificada a partir das ações executadas

Nós principais:
- parse_and_classify: Classificação de intenção e extração de slots
- prepare_plan: Orquestração multi-intents e planejamento de ações
- router_node: Validação de slots e sinalização de resolução de leads
- route_intent: Roteamento condicional para handlers especializados
- resolve_lead: Resolução de referência de lead a partir do BD
- leads_agent: Subgrafo ReAct para operações de leads
- handle_notes: Execução de intent de notas
- handle_tasks: Execução de intent de tarefas
- proposals_agent: Subgrafo ReAct para operações de propostas
- update_context: Normalização e consolidação de estado
- execute_pending: Execução de ação enfileirada
- respond_final: Geração de resposta unificada
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from types import SimpleNamespace

import ast
import re
import json


# === IMPORTS e configuração ===
from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from app.agent.prompt import (
    PARSER_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    LEAD_REACT_PROMPT,
    PROPOSAL_REACT_PROMPT,
    FINALIZER_SYSTEM_PROMPT,
)
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from app.agent import tools as crm_tools
from app.agent import new_react
from app.agent.helpers import (
    extract_text_content,
    ensure_context,
    lead_context_from_result,
    get_lead_context,
    extract_ref_from_text,
    has_pending,
    push_ai_response,
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


load_dotenv()


# === Modelo LLM (configuração base) ===
# Modelo padrão (baixo esforço) para parser/roteador/lead/tasks/finalizer
model = ChatOpenAI(
    model="gpt-5-nano",
    output_version="responses/v1",
    reasoning={"effort": "low"},
    verbosity="medium",
)

# Modelo dedicado para PROPOSTAS (mais esforço de raciocínio)
proposals_model = ChatOpenAI(
    model="gpt-5-nano",
    output_version="responses/v1",
    reasoning={"effort": "medium"},
    verbosity="medium",
)

# Modelo dedicado para o finalizador (resposta única) — conciso
finalizer_model = ChatOpenAI(
    model="gpt-5-nano",
    output_version="responses/v1",
    reasoning={"effort": "low"},
    verbosity="low",
)

# === Tipos, intenções e constantes ===
IntentLiteral = Literal[
    "lead_criar",
    "lead_obter",
    "lead_buscar",
    "lead_listar",
    "lead_atualizar",
    "nota_adicionar",
    "nota_listar",
    "tarefa_criar",
    "tarefa_concluir",
    "tarefa_listar",
    "proposta_rascunhar",
    "proposta_adicionar_item",
    "proposta_calcular_totais",
    "proposta_listar",
    "proposta_exportar",
    "proposta_atualizar_corpo",
    "listar_status_lead",
    "conversa_geral",
]

LEAD_INTENTS = {
    "lead_criar",
    "lead_obter",
    "lead_buscar",
    "lead_listar",
    "lead_atualizar",
}
NOTE_INTENTS = {"nota_adicionar", "nota_listar"}
TASK_INTENTS = {"tarefa_criar", "tarefa_concluir", "tarefa_listar"}
PROPOSAL_INTENTS = {
    "proposta_rascunhar",
    "proposta_adicionar_item",
    "proposta_calcular_totais",
    "proposta_listar",
    "proposta_exportar",
    "proposta_atualizar_corpo",
}

INTENTS_REQUIRING_LEAD = {
    "lead_obter",
    "lead_atualizar",
    "nota_adicionar",
    "nota_listar",
    "tarefa_criar",
    "tarefa_listar",
    "proposta_rascunhar",
    "proposta_listar",
}

REQUIRED_SLOTS: Dict[str, List[str]] = {
    "lead_criar": ["nome"],
    "lead_obter": ["lead_ref_ou_id"],
    "lead_buscar": ["consulta"],
    "lead_atualizar": ["lead_ref_ou_id"],
    "nota_adicionar": ["texto"],
    "tarefa_criar": ["titulo"],
    "tarefa_concluir": ["tarefa_id"],
    "proposta_rascunhar": ["titulo"],
    "proposta_adicionar_item": ["proposta_id", "descricao", "quantidade", "preco_unitario"],
    "proposta_calcular_totais": ["proposta_id"],
    "proposta_exportar": ["proposta_id", "formato"],
    "proposta_atualizar_corpo": ["proposta_id", "corpo_md"],
}

DEFAULT_LIMIT = 20
DEFAULT_OFFSET = 0


# === Esquemas (Pydantic) e instâncias de LLM ===
class ParserResponse(BaseModel):
    intent: IntentLiteral
    slots: List[str]

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: IntentLiteral
    slots: Dict[str, Any]
    lead_atual: Optional[Dict[str, Any]]
    errors: List[str]
    context: Dict[str, Any]
    # Campos mínimos e didáticos para este grafo


parser_llm = model.with_structured_output(ParserResponse)

# Planejador multi-intents (modelo dedicado p/ robustez em NL)
planner_model = ChatOpenAI(
    model="gpt-5-nano",
    output_version="responses/v1",
    reasoning={"effort": "medium"},
    verbosity="low",
)
from pydantic import BaseModel
from pydantic import ConfigDict

class PlanAction(BaseModel):
    """Representa uma ação atômica que o planner extrai da mensagem do usuário.

    - intent: uma das intents suportadas pelo grafo
    - slots: lista de strings contendo um único objeto por ação (ex: "{titulo: 'X', data_limite: 'YYYY-MM-DD'}")
    """
    intent: IntentLiteral
    slots: List[str] = []
    model_config = ConfigDict(extra='forbid')

def normalize_json_like(s: str) -> str:
    """Normaliza variantes comuns de JSON retornadas por LLMs.

    - Converte actions=[...] → {"actions": [...]}
    - Envelopa arrays puros em um objeto {"actions": [...]} se necessário
    - Adiciona aspas às chaves e troca aspas simples por duplas
    """
    t = (s or "").strip()
    if t.startswith("actions="):
        t = "{" + t.replace("actions=", "\"actions\":", 1) + "}"
    if t.startswith("[") and t.endswith("]"):
        t = "{\"actions\": " + t + "}"
    t = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:", r'\1"\2":', t)
    t = t.replace("'", '"')
    return t

def parse_plan_actions_from_text(text: str) -> List[Dict[str, Any]]:
    """Extrai a lista de actions de uma resposta textual da LLM.

    Suporta respostas em JSON bem‑formado e variantes "quase JSON" comuns.
    Retorna uma lista de dicts em forma canônica: {intent: str, slots: List[str]}.
    """
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\bactions\b\s*:\s*\[.*\]\s*\}", raw, re.DOTALL)
        if m:
            snippet = normalize_json_like(m.group(0))
            try:
                obj = json.loads(snippet)
            except Exception:
                obj = None
        else:
            try:
                obj = json.loads(normalize_json_like(raw))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return []
    actions = obj.get("actions")
    if not isinstance(actions, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for it in actions:
        if not isinstance(it, dict):
            continue
        intent = it.get("intent")
        slots = it.get("slots", [])
        if not isinstance(slots, list):
            if isinstance(slots, dict):
                try:
                    slots = [json.dumps(slots, ensure_ascii=False)]
                except Exception:
                    slots = []
            else:
                slots = []
        if intent:
            cleaned.append({"intent": intent, "slots": slots})
    return cleaned

class PlannerWrapper:
    """Envoltório simples para o LLM do planner.

    - Invoca o modelo
    - Extrai texto da resposta
    - Faz o parsing robusto das actions
    - Retorna um objeto com atributo `actions` compatível com o consumo atual
    """
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, msgs):
        from app.agent.helpers import extract_text_content as extract
        resp = self._llm.invoke(msgs)
        txt = extract(resp)
        actions = parse_plan_actions_from_text(txt)
        return SimpleNamespace(
            actions=[SimpleNamespace(intent=a.get("intent"), slots=a.get("slots", [])) for a in actions]
        )

plan_llm = PlannerWrapper(planner_model)

def slots_strings_to_dict(slots_strings: List[str]) -> Dict[str, Any]:
    """Converte a lista de strings de objetos dos slots em um dict Python.

    Aceita entradas do tipo ["{titulo: 'X', data_limite: 'YYYY-MM-DD'}"].
    Também tolera o formato name=value como fallback (ex.: "titulo=Follow‑up").
    """
    import json as _json, re as _re
    out: Dict[str, Any] = {}
    for s in slots_strings or []:
        if not isinstance(s, str):
            continue
        txt = s.strip()
        if txt.startswith("{") and txt.endswith("}"):
            # Adiciona aspas às chaves não citadas e converte aspas simples → duplas
            txt2 = _re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:", r'\1"\2":', txt)
            txt2 = txt2.replace("'", '"')
            try:
                data = _json.loads(txt2)
                if isinstance(data, dict):
                    out.update({k: v for k, v in data.items()})
                    continue
            except Exception:
                pass
        # Fallback: name=value
        if "=" in txt:
            k, v = txt.split("=", 1)
            out[k.strip()] = v.strip()
    return out

# === Função auxiliar de decisão (permanece no workflow) ===
def route_intent(state: AgentState) -> str:
    """Decide o próximo nó com base no intent/contexto."""
    intent = state.get("intent")
    ctx = state.get("context") or {}
    if ctx.get("need_lead_resolution"):
        return "resolve_lead"
    if intent in LEAD_INTENTS:
        return "handle_leads"
    if intent in NOTE_INTENTS:
        return "handle_notes"
    if intent in TASK_INTENTS:
        return "handle_tasks"
    if intent in PROPOSAL_INTENTS:
        return "handle_proposals"
    if intent == "listar_status_lead":
        return "respond_final"
    return "respond_final"





# === Nós: parser e planejamento ===
def parse_and_classify(state: AgentState) -> AgentState:
    """Nó: parse_and_classify
    - Classifica a intenção e extrai slots com LLM + Pydantic
    - Entrada: última mensagem do usuário em `messages`
    - Saída: `intent` e `slots` normalizados (dict)
    """
    messages = state.get("messages") or []
    parsed = parser_llm.invoke([SystemMessage(content=PARSER_SYSTEM_PROMPT), *messages])
    
    slots_dict = {}
    for slot_str in parsed.slots:
        if "=" in slot_str:
            name, value = slot_str.split("=", 1)
            slots_dict[name.strip()] = value.strip()
    
    return {"intent": parsed.intent, "slots": slots_dict}


def prepare_plan(state: AgentState) -> AgentState:
    """Nó: prepare_plan
    - Planeja múltiplas ações do turno (multi-intents) e salva em `context.pending_actions`
    - Deduplica ações e normaliza slots (ex.: moeda BRL em propostas)
    - Se o `intent` atual exige lead e não há referência, antecipa um `lead_criar` do plano
    """
    messages = state.get("messages") or []
    ctx = ensure_context(state)
    pending = list(ctx.get("pending_actions") or [])
    # 1) Planejamento via LLM (multi-intents)
    try:
        plan = plan_llm.invoke([SystemMessage(content=PLAN_SYSTEM_PROMPT), *messages])
        # Debug opcional do planner (visível no Studio)
        try:
            ctx["planner_debug"] = [
                {"intent": x.intent, "slots": list(x.slots or [])} for x in (plan.actions or [])
            ]
        except Exception:
            pass
        primary_intent = state.get("intent")
        new_actions: List[Dict[str, Any]] = []
        lead_ctx = state.get("lead_atual")
        for a in plan.actions or []:
            intent_a: IntentLiteral = a.intent  # type: ignore
            if primary_intent and intent_a == primary_intent:
                continue
            # Evita lead_criar quando já houver lead no contexto
            if intent_a == "lead_criar" and lead_ctx and lead_ctx.get("lead_id"):
                continue
            raw_slots = getattr(a, "slots", None)
            if isinstance(raw_slots, dict):
                norm_slots = dict(raw_slots)
            else:
                norm_slots = slots_strings_to_dict(list(raw_slots or []))
            # Default de moeda para proposta
            if intent_a == "proposta_rascunhar" and "moeda" not in norm_slots:
                norm_slots["moeda"] = "BRL"
            # Injeta referência de lead quando necessário
            if intent_a in INTENTS_REQUIRING_LEAD and not norm_slots.get("lead_ref_ou_id"):
                if lead_ctx and lead_ctx.get("lead_id"):
                    norm_slots["lead_ref_ou_id"] = str(lead_ctx["lead_id"])
            new_actions.append({"intent": intent_a, "slots": norm_slots})
        def k(x):
            return (x.get("intent"), tuple(sorted((x.get("slots") or {}).items())))
        keys = {k(x) for x in pending}
        for act in new_actions:
            if k(act) not in keys:
                pending.append(act)
                keys.add(k(act))
    except Exception as e:
        # expõe erro do planner no contexto para debug no Studio
        try:
            ctx["planner_error"] = str(e)
        except Exception:
            pass
        # segue sem interromper o fluxo
        pass
    # 2) Atualiza context
    if pending:
        ctx["pending_actions"] = pending
    # 3) Se intent atual requer lead e não temos referência, antecipa lead_criar do plano
    intent = state.get("intent")
    slots = dict(state.get("slots") or {})
    needs_lead = intent in INTENTS_REQUIRING_LEAD and not (
        slots.get("lead_ref_ou_id") or (state.get("lead_atual") and state["lead_atual"].get("lead_id"))
    )
    if needs_lead and pending:
        for i, act in enumerate(pending):
            if act.get("intent") == "lead_criar":
                lead_action = pending.pop(i)
                ctx["pending_actions"] = pending
                return {"context": ctx, "intent": "lead_criar", "slots": dict(lead_action.get("slots") or {})}
    return {"context": ctx}


# === Nós: roteador e direção ===
def router_node(state: AgentState) -> AgentState:
    """Nó: router
    - Valida slots obrigatórios e injeta `lead_ref_ou_id` a partir de `lead_atual`
    - Sinaliza bloqueios e a necessidade de resolução de lead no `context`
    - Não executa ferramentas; apenas prepara o estado para o roteamento
    """
    intent = state.get("intent")
    slots = dict(state.get("slots") or {})
    errors = list(state.get("errors") or [])
    context = ensure_context(state)
    missing: List[str] = []
    required = REQUIRED_SLOTS.get(intent or "", [])
    for name in required:
        if name == "lead_ref_ou_id" and (
            slots.get(name)
            or (state.get("lead_atual") and state["lead_atual"].get("lead_id"))
        ):
            continue
        value = slots.get(name)
        if value in (None, "", []):
            missing.append(name)
    if intent in INTENTS_REQUIRING_LEAD and not slots.get("lead_ref_ou_id"):
        lead_atual = state.get("lead_atual")
        if lead_atual and lead_atual.get("lead_id"):
            slots["lead_ref_ou_id"] = str(lead_atual["lead_id"])
        else:
            missing.append("lead_ref_ou_id")
    if missing:
        errors.append(f"Para {intent} preciso de: {', '.join(sorted(set(missing)))}.")
        context["router_missing"] = sorted(set(missing))
        context["router_blocked"] = True
        # Marca que precisamos resolver lead se faltou a referência em intents que exigem lead
        if (
            intent in INTENTS_REQUIRING_LEAD
            and "lead_ref_ou_id" in missing
            and not context.get("lead_resolution_attempted")
        ):
            context["need_lead_resolution"] = True
    else:
        context["router_missing"] = []
        context["router_blocked"] = False
        context["need_lead_resolution"] = False
    # Caso haja uma referência textual de lead, mas nenhum lead_atual confirmado, solicite resolução
    if (
        intent in INTENTS_REQUIRING_LEAD
        and slots.get("lead_ref_ou_id")
        and not (state.get("lead_atual") and state["lead_atual"].get("lead_id"))
        and not context.get("lead_resolution_attempted")
    ):
        context["need_lead_resolution"] = True
    updates: AgentState = {"slots": slots, "context": context}
    if errors:
        updates["errors"] = errors
    return updates


# (route_intent movido para o bloco de auxiliares no topo)

# === Nó: resolução de lead ===
def resolve_lead(state: AgentState) -> AgentState:
    """Nó: resolve_lead
    - Tenta resolver a referência do lead (ID, e-mail, telefone ou termo)
    - Usa ferramentas `obter_lead`/`resolver_lead` e atualiza `slots`/`lead_atual`
    - Evita loops marcando `lead_resolution_attempted` no `context`
    """
    slots = dict(state.get("slots") or {})
    context = ensure_context(state)
    # Evita loops: registramos que já tentamos resolver nesta interação
    context["lead_resolution_attempted"] = True
    lead_atual = state.get("lead_atual")

    # 1) Se já há lead_atual confirmado, apenas siga
    if lead_atual and lead_atual.get("lead_id"):
        context["need_lead_resolution"] = False
        context["router_blocked"] = False
        return {"context": context}

    # 2) Tente resolver a partir do slot explícito (texto/uuid/email/telefone/nome/empresa)
    ref_slot = slots.get("lead_ref_ou_id")
    if ref_slot:
        try:
            resp = crm_tools.get_lead.invoke({"ref": str(ref_slot)})
        except Exception:
            resp = None
        lead_ctx = lead_context_from_result(resp or {})
        if lead_ctx:
            # Substitui a referência textual pelo id e segue
            slots["lead_ref_ou_id"] = str(lead_ctx["lead_id"]) 
            context["need_lead_resolution"] = False
            context["router_blocked"] = False
            return {"slots": slots, "lead_atual": lead_ctx, "context": context}
        # Desambiguação: se houver múltiplos candidatos, devolve lista para UI e bloqueia
        if resp and resp.get("error") and resp["error"].get("matches"):
            context["matches"] = resp["error"]["matches"]
            context["need_lead_resolution"] = False
            context["router_blocked"] = True
            return {"context": context}

    # 3) Fallback: tenta extrair uma referência direta do texto do usuário (uuid/email/telefone)
    messages = state.get("messages") or []
    text = extract_text_content(messages[-1]) if messages else ""
    ref_direct = extract_ref_from_text(text)
    if ref_direct:
        try:
            resp = crm_tools.get_lead.invoke({"ref": str(ref_direct)})
        except Exception:
            resp = None
        lead_ctx = lead_context_from_result(resp or {})
        if lead_ctx:
            slots["lead_ref_ou_id"] = str(lead_ctx["lead_id"]) 
            context["need_lead_resolution"] = False
            context["router_blocked"] = False
            return {"slots": slots, "lead_atual": lead_ctx, "context": context}

    # 4) Último fallback: tenta resolver por nome/empresa a partir de slots auxiliares
    term_parts: List[str] = []
    for key in ("nome", "empresa"):
        if slots.get(key):
            term_parts.append(str(slots[key]))
    term = " ".join(term_parts).strip()
    if term:
        try:
            resp = crm_tools.resolve_lead.invoke({"ref": term})
        except Exception:
            resp = None
        if resp and resp.get("data") and resp["data"].get("lead_id"):
            lead_id = str(resp["data"]["lead_id"]) 
            lead_ctx = get_lead_context(lead_id)
            if lead_ctx:
                slots["lead_ref_ou_id"] = lead_id
                context["need_lead_resolution"] = False
                context["router_blocked"] = False
                return {"slots": slots, "lead_atual": lead_ctx, "context": context}
        if resp and resp.get("error") and resp["error"].get("matches"):
            context["matches"] = resp["error"]["matches"]
            context["need_lead_resolution"] = False
            context["router_blocked"] = True
            return {"context": context}

    # 5) Não resolvido: preserve o slot textual (se houver) para tools que aceitam referência natural
    #    e não bloqueie o fluxo — o próximo nó (ex.: handle_notes) poderá resolver via tool.
    context["need_lead_resolution"] = False
    context["router_blocked"] = False
    return {"context": context, "slots": slots}




def handle_notes(state: AgentState) -> AgentState:
    """Nó: handle_notes
    - Implementa intents de notas (adicionar/listar) diretamente via tools
    - Requer `lead_ref_ou_id` (usa `lead_atual` se disponível)
    """
    intent = state["intent"]
    slots = state["slots"]
    lead_atual = state.get("lead_atual")

    # Usa id do lead_atual quando disponível; caso contrário, preserva a referência textual.
    lead_ref = slots.get("lead_ref_ou_id") or (lead_atual.get("lead_id") if lead_atual else None)
    if not lead_ref:
        return {"messages": AIMessage(content="Preciso de um lead para trabalhar com notas.")}
    
    if intent == "nota_adicionar":
        texto = slots.get("texto")
        if not texto:
            return {"messages": AIMessage(content="Preciso do texto da nota.")}
        result = crm_tools.add_note_to_lead.invoke({"lead_ref_ou_id": str(lead_ref), "texto": texto})
        if result.get("error"):
            msg = result["error"].get("message", "Erro ao processar nota.")
            ctx = ensure_context(state)
            push_ai_response(ctx, intent, msg, ok=False, data={"lead_ref": str(lead_ref)})
            return {"context": ctx, "messages": AIMessage(content=msg)}
        # sucesso
        ctx = ensure_context(state)
        push_ai_response(ctx, intent, "Nota registrada com sucesso.", ok=True, data={"lead_ref": str(lead_ref)})
        return {"context": ctx, "messages": AIMessage(content="Nota registrada com sucesso.")}
    
    elif intent == "nota_listar":
        result = crm_tools.list_notes.invoke({"lead_ref_ou_id": str(lead_ref)})
        if result.get("error"):
            msg = result["error"].get("message", "Erro ao processar nota.")
            ctx = ensure_context(state)
            push_ai_response(ctx, intent, msg, ok=False, data={"lead_ref": str(lead_ref)})
            return {"context": ctx, "messages": AIMessage(content=msg)}
        items = result.get("data", {}).get("items", [])
        msg = f"Encontrei {len(items)} notas."
        ctx = ensure_context(state)
        push_ai_response(ctx, intent, msg, ok=True, data={"lead_ref": str(lead_ref), "total": len(items)})
        return {"context": ctx, "messages": AIMessage(content=msg)}
    else:
        return {"messages": AIMessage(content=f"Intent não suportado: {intent}")}



def handle_tasks(state: AgentState) -> AgentState:
    """Nó: handle_tasks
    - Implementa intents de tarefas (criar/concluir/listar) diretamente via tools
    - Normaliza dados opcionais (tipo, data_limite) e tem fallback para concluir última tarefa aberta
    """
    intent = state["intent"]
    slots = state["slots"]
    lead_atual = state.get("lead_atual")
    
    if intent == "tarefa_criar":
        lead_ref = slots.get("lead_ref_ou_id") or (lead_atual.get("lead_id") if lead_atual else None)
        if not lead_ref:
            return {"messages": AIMessage(content="Preciso de um lead para criar a tarefa.")}
        titulo = slots.get("titulo")
        if not titulo:
            return {"messages": AIMessage(content="Preciso do título da tarefa.")}
        # Normaliza campos opcionais antes de chamar a tool
        payload = {"lead_ref_ou_id": str(lead_ref), "titulo": titulo}
        if slots.get("tipo"):
            payload["tipo"] = slots["tipo"]
        # data_limite aceita ISO (YYYY-MM-DD). Evita enviar placeholders (ex.: "YYYY-MM-DD").
        data_limite = slots.get("data_limite")
        if isinstance(data_limite, str):
            s = data_limite.strip().lower()
            # converte palavras comuns
            from datetime import date, timedelta
            if s in ("amanha", "amanhã"):
                payload["data_limite"] = (date.today() + timedelta(days=1)).isoformat()
            elif s == "hoje":
                payload["data_limite"] = date.today().isoformat()
            else:
                # aceita apenas ISO real (4-2-2 dígitos); se vier placeholder ou inválido, não envia
                import re as _re
                if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                    payload["data_limite"] = s
                # caso contrário, omite para evitar erro no banco
        result = crm_tools.create_task.invoke(payload)
        if result.get("error"):
            msg = result["error"].get("message", "Erro ao criar tarefa.")
            ctx = ensure_context(state)
            push_ai_response(ctx, intent, msg, ok=False, data=payload)
            return {"context": ctx, "messages": AIMessage(content=msg)}
        tarefa_id = result.get("data", {}).get("tarefa_id")
        msg = f"Tarefa criada com sucesso. ID: {tarefa_id}"
        ctx = ensure_context(state)
        push_ai_response(ctx, intent, "Tarefa criada", ok=True, data={"tarefa_id": str(tarefa_id)})
        return {"context": ctx, "messages": AIMessage(content=msg)}
    
    elif intent == "tarefa_concluir":
        tarefa_id = slots.get("tarefa_id")
        if not tarefa_id:
            # Heurística: concluir a última tarefa aberta do lead, se possível
            lead_ref = slots.get("lead_ref_ou_id") or (lead_atual.get("lead_id") if lead_atual else None)
            if lead_ref:
                listar = crm_tools.list_tasks.invoke({"lead_ref_ou_id": str(lead_ref), "status": "aberta"})
                if not listar.get("error"):
                    items = listar.get("data", {}).get("items", [])
                    if items:
                        tarefa_id = str(items[0]["tarefa_id"])  # já vem ordenado por criado_em desc
            if not tarefa_id:
                return {"messages": AIMessage(content="Preciso do ID da tarefa para concluir.")}
        result = crm_tools.complete_task.invoke({"tarefa_id": str(tarefa_id)})
        if result.get("error"):
            msg = result["error"].get("message", "Erro ao concluir tarefa.")
            ctx = ensure_context(state)
            push_ai_response(ctx, intent, msg, ok=False, data={"tarefa_id": str(tarefa_id)})
            return {"context": ctx, "messages": AIMessage(content=msg)}
        ctx = ensure_context(state)
        push_ai_response(ctx, intent, "Tarefa concluída", ok=True, data={"tarefa_id": str(tarefa_id)})
        return {"context": ctx, "messages": AIMessage(content="Tarefa concluída com sucesso.")}
    
    elif intent == "tarefa_listar":
        payload = {}
        lead_ref = slots.get("lead_ref_ou_id") or (lead_atual.get("lead_id") if lead_atual else None)
        if lead_ref:
            payload["lead_ref_ou_id"] = str(lead_ref)
        if slots.get("status"):
            payload["status"] = slots["status"]
        result = crm_tools.list_tasks.invoke(payload)
        if result.get("error"):
            # Fallback: se lead não for encontrado, tente listar sem filtro de lead
            msg = result["error"].get("message", "Erro ao listar tarefas.")
            if payload.get("lead_ref_ou_id") and msg.lower().startswith("lead não encontrado"):
                fallback_payload = {}
                if payload.get("status"):
                    fallback_payload["status"] = payload["status"]
                result = crm_tools.list_tasks.invoke(fallback_payload)
                if not result.get("error"):
                    items = result.get("data", {}).get("items", [])
                    msg2 = f"Encontrei {len(items)} tarefas."
                    ctx = ensure_context(state)
                    push_ai_response(ctx, intent, msg2, ok=True, data={"total": len(items)})
                    return {"context": ctx, "messages": AIMessage(content=msg2)}
            ctx = ensure_context(state)
            push_ai_response(ctx, intent, msg, ok=False, data=payload)
            return {"context": ctx, "messages": AIMessage(content=msg)}
        items = result.get("data", {}).get("items", [])
        msg = f"Encontrei {len(items)} tarefas."
        ctx = ensure_context(state)
        push_ai_response(ctx, intent, msg, ok=True, data={"total": len(items)})
        return {"context": ctx, "messages": AIMessage(content=msg)}
    
    else:
        return {"messages": AIMessage(content=f"Intent não suportado: {intent}")}




# === Nó: atualização de contexto ===
def update_context(state: AgentState) -> AgentState:
    """Nó: update_context
    - Consolida `lead_atual` a partir de resultados de ferramentas/mensagens
    - Injeta `lead_ref_ou_id` nos slots quando possível
    - Padroniza `context.pending_actions`
    - Replaneja (LLM) UMA vez no turno para extrair ações restantes (sem heurísticas regex)
    """
    slots = dict(state.get("slots") or {})
    tool_result = state.get("tool_result") or {}
    data = tool_result.get("data") or {}
    new_lead = None
    ctx = ensure_context(state)
    # Normaliza estrutura de pendências (sempre plural)
    if ctx.get("pending_actions") is None:
        ctx["pending_actions"] = []

    # 1) Prioriza qualquer tool_result explícito
    primary_ref = data.get("lead_id") or data.get("lead_ref_ou_id")
    if primary_ref:
        new_lead = get_lead_context(primary_ref)

    # 2) Caso não haja tool_result no estado, varre as ToolMessages recentes para extrair contexto e segmentos
    if not new_lead:
        messages = state.get("messages") or []
        # Chaves para evitar segmentos duplicados
        existing_keys = set()
        try:
            for seg in (ctx.get("ai_responses") or []):
                data = (seg or {}).get("data") or {}
                intent_name = (seg or {}).get("intent") or ""
                k = None
                if intent_name == "lead_criar" and data.get("lead_id"):
                    k = f"lead:{data.get('lead_id')}"
                elif intent_name == "proposta_rascunhar" and data.get("proposta_id"):
                    k = f"draft:{data.get('proposta_id')}"
                elif intent_name == "proposta_adicionar_item" and data.get("proposta_id") and data.get("item_id"):
                    k = f"item:{data.get('proposta_id')}:{data.get('item_id')}"
                elif intent_name == "proposta_calcular_totais" and data.get("proposta_id"):
                    k = f"totals:{data.get('proposta_id')}:{data.get('subtotal')}:{data.get('total')}"
                if k:
                    existing_keys.add(k)
        except Exception:
            pass
        # Varre ToolMessages mais recentes (reversed)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                raw = msg.content
                try:
                    parsed = None
                    if isinstance(raw, dict):
                        parsed = raw
                    elif isinstance(raw, str):
                        # Remove UUID('...') para permitir literal_eval
                        clean = re.sub(r"UUID\('([^']*)'\)", r"'\1'", raw)
                        parsed = ast.literal_eval(clean)
                    if isinstance(parsed, dict):
                        d = parsed.get("data") or {}
                        lead_ref = d.get("lead_id") or d.get("lead_ref_ou_id")
                        if lead_ref and not new_lead:
                            new_lead = get_lead_context(lead_ref)
                            if new_lead:
                                # injeta lead_ref_ou_id no slot se não houver
                                slots.setdefault("lead_ref_ou_id", str(new_lead.get("lead_id")))
                        # Cria segmentos didáticos para o finalizer com base na tool usada
                        tool_name = getattr(msg, "name", "") or ""
                        try:
                            if tool_name == "create_lead" and d.get("lead_id"):
                                key = f"lead:{d.get('lead_id')}"
                                if key not in existing_keys:
                                    push_ai_response(
                                        ctx,
                                        "lead_criar",
                                        f"Lead criado (id: {d.get('lead_id')})",
                                        ok=True,
                                        data={"lead_id": d.get("lead_id")},
                                    )
                                    existing_keys.add(key)
                            elif tool_name == "draft_proposal" and d.get("proposta_id"):
                                key = f"draft:{d.get('proposta_id')}"
                                if key not in existing_keys:
                                    push_ai_response(
                                        ctx,
                                        "proposta_rascunhar",
                                        f"Proposta rascunhada (id: {d.get('proposta_id')})",
                                        ok=True,
                                        data={"proposta_id": d.get("proposta_id"), "lead_id": d.get("lead_id")},
                                    )
                                    existing_keys.add(key)
                            elif tool_name == "add_proposal_item" and d.get("proposta_id") and d.get("item_id"):
                                key = f"item:{d.get('proposta_id')}:{d.get('item_id')}"
                                if key not in existing_keys:
                                    push_ai_response(
                                        ctx,
                                        "proposta_adicionar_item",
                                        "Item adicionado",
                                        ok=True,
                                        data={"proposta_id": d.get("proposta_id"), "item_id": d.get("item_id")},
                                    )
                                    existing_keys.add(key)
                            elif tool_name == "calculate_proposal_totals" and d.get("proposta_id") is not None:
                                key = f"totals:{d.get('proposta_id')}:{d.get('subtotal')}:{d.get('total')}"
                                if key not in existing_keys:
                                    push_ai_response(
                                        ctx,
                                        "proposta_calcular_totais",
                                        f"Totais atualizados: subtotal={d.get('subtotal')} total={d.get('total')}",
                                        ok=True,
                                        data={
                                            "proposta_id": d.get("proposta_id"),
                                            "subtotal": d.get("subtotal"),
                                            "total": d.get("total"),
                                            "desconto_pct": d.get("desconto_pct"),
                                        },
                                    )
                                    existing_keys.add(key)
                        except Exception:
                            # ignora falhas de push de segmento
                            pass
                except Exception:
                    # Se parsing falhar, segue para a próxima mensagem/tool
                    continue

    # 3) Se ainda não houver, usa o slot existente
    if not new_lead and slots.get("lead_ref_ou_id"):
        new_lead = get_lead_context(slots["lead_ref_ou_id"])

    # 3.1) Se já temos um lead consolidado e existem ações pendentes, injete a referência
    #      de lead nos slots das ações que exigirem lead e ainda não a possuírem.
    try:
        pending_now = list(ctx.get("pending_actions") or [])
        if new_lead and pending_now:
            lead_id_str = str(new_lead.get("lead_id")) if new_lead.get("lead_id") else None
            changed = False
            for act in pending_now:
                intent_a = act.get("intent")
                slots_a = dict(act.get("slots") or {})
                if intent_a in INTENTS_REQUIRING_LEAD and lead_id_str and not slots_a.get("lead_ref_ou_id"):
                    slots_a["lead_ref_ou_id"] = lead_id_str
                    act["slots"] = slots_a
                    changed = True
                # também garante moeda padrão para propostas
                if intent_a == "proposta_rascunhar" and "moeda" not in slots_a:
                    slots_a["moeda"] = "BRL"
                    act["slots"] = slots_a
                    changed = True
            if changed:
                ctx["pending_actions"] = pending_now
    except Exception:
        # não bloqueia fluxo se normalização falhar
        pass

    # 4) Replanejamento LLM (sem heurísticas):
    #    Se não houver ações pendentes e ainda não replanejamos neste turno,
    #    pedimos ao planner LLM para (re)extrair ações restantes a partir do texto do usuário.
    #    - Motivo: linguagem natural varia; o primeiro planejamento pode não capturar tudo.
    #    - Proteções: marca `context.replanned=True` para evitar loop; deduplica contra a fila atual;
    #                 injeta `lead_ref_ou_id` quando já temos `lead_atual`.
    try:
        pending_now = list(ctx.get("pending_actions") or [])
        already_replanned = bool(ctx.get("replanned"))
        if not pending_now and not already_replanned:
            # 4.1) Coleta o último texto do usuário (base para replanejamento)
            user_text = ""
            for _m in reversed(state.get("messages") or []):
                if isinstance(_m, HumanMessage):
                    user_text = extract_text_content(_m) or ""
                    break
            # 4.2) Invoca o planejador LLM SOMENTE com a última HumanMessage,
            #      precedida de um contexto explícito de replanejamento para
            #      extrair as DEMAIS ações do pedido original.
            planner_msgs = [SystemMessage(content=PLAN_SYSTEM_PROMPT)]
            lead_ctx = new_lead or state.get("lead_atual")
            if lead_ctx and lead_ctx.get("lead_id"):
                repl_ctx = (
                    "Contexto de replanejamento: o lead já foi criado (lead_id={lid}). "
                    "Extraia as demais ações pedidas originalmente (ex.: nota, tarefa, proposta), "
                    "mantendo os slots citados. Não inclua a criação do lead novamente."
                ).format(lid=str(lead_ctx["lead_id"]))
                planner_msgs.append(HumanMessage(content=repl_ctx))
            if user_text:
                planner_msgs.append(HumanMessage(content=user_text))
            else:
                # fallback conservador: usa o fluxo atual
                planner_msgs.extend(state.get("messages") or [])
            plan = plan_llm.invoke(planner_msgs)
            # 4.3) Normaliza e filtra ações replanejadas
            new_actions: List[Dict[str, Any]] = []
            current_intent = state.get("intent")
            lead_ctx = new_lead or state.get("lead_atual")
            for a in (plan.actions or []):
                intent_a: IntentLiteral = a.intent  # type: ignore
                raw_slots = getattr(a, "slots", None)
                if isinstance(raw_slots, dict):
                    slots_a = dict(raw_slots)
                else:
                    slots_a = slots_strings_to_dict(list(raw_slots or []))
                # ignora repetir a intenção já em execução
                if current_intent and intent_a == current_intent:
                    continue
                # evita propor lead_criar novamente quando já existe lead no contexto
                if intent_a == "lead_criar" and (lead_ctx and lead_ctx.get("lead_id")):
                    continue
                # default de moeda para proposta
                if intent_a == "proposta_rascunhar" and "moeda" not in slots_a:
                    slots_a["moeda"] = "BRL"
                # injeta referência de lead quando necessário e disponível
                if intent_a in INTENTS_REQUIRING_LEAD and not slots_a.get("lead_ref_ou_id"):
                    if lead_ctx and lead_ctx.get("lead_id"):
                        slots_a["lead_ref_ou_id"] = str(lead_ctx["lead_id"])
                new_actions.append({"intent": intent_a, "slots": slots_a})
            # 4.4) Deduplica contra a fila atual e contra ações já executadas neste turno
            def _k(x: Dict[str, Any]):
                return (x.get("intent"), tuple(sorted((x.get("slots") or {}).items())))
            keys = { _k(x) for x in pending_now }
            executed_intents = set()
            try:
                for seg in (ctx.get("ai_responses") or []):
                    name = (seg or {}).get("intent")
                    if name:
                        executed_intents.add(name)
            except Exception:
                pass
            for act in new_actions:
                if act.get("intent") in executed_intents:
                    continue
                if _k(act) not in keys:
                    pending_now.append(act)
                    keys.add(_k(act))
            # 4.5) Atualiza o contexto com a fila e marca replanejamento
            ctx["pending_actions"] = pending_now
            ctx["replanned"] = True
    except Exception:
        # Em qualquer falha do replanejamento, seguimos sem interromper o fluxo
        pass

    updates: AgentState = {"slots": slots, "context": ctx}
    if new_lead:
        updates["lead_atual"] = new_lead
    return updates

# Heurísticas auxiliares foram movidas para app.agent.parsers

# === Nó: resposta final ao usuário ===
def respond_final(state: AgentState) -> AgentState:
    """Nó: respond_final
    - Gera a resposta final ao usuário quando não há mais ações pendentes
    - Se `context.ai_responses` tiver 1 item, encaminha-o direto
    - Se tiver >1, agrega todos em uma resposta coerente
    - Para intents específicos (listar_*), pode chamar tools e padronizar o texto
    """
    ctx = ensure_context(state)
    intent = state.get("intent")
    ai_resps = list(ctx.get("ai_responses") or [])

    # Caso especial: lookup de status de lead (determinístico)
    if intent == "listar_status_lead":
        try:
            res = crm_tools.list_lead_status.invoke({})
            if res.get("error"):
                msg = res["error"].get("message", "Erro ao listar status de lead.")
                push_ai_response(ctx, intent, msg, ok=False)
                return {"context": ctx, "messages": AIMessage(content=msg)}
            items = (res.get("data") or {}).get("items", [])
            codigos = ", ".join([str(it.get("codigo")) for it in items]) if items else "nenhum"
            total = (res.get("data") or {}).get("total")
            msg = f"Status de lead válidos ({total}): {codigos}."
            push_ai_response(ctx, intent, msg, ok=True, data={"total": total})
            return {"context": ctx, "messages": AIMessage(content=msg)}
        except Exception:
            msg = "Não foi possível listar os status de lead agora."
            push_ai_response(ctx, intent, msg, ok=False)
            return {"context": ctx, "messages": AIMessage(content=msg)}

    # Finalização via LLM: monta um resumo natural com base SOMENTE nas ações executadas
    # Prepara contexto do usuário e ações
    user_text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            user_text = extract_text_content(m)
            break
    actions_txt = []
    errors_txt = []
    for seg in ai_resps:
        txt = (seg.get("text") or "").strip()
        if txt:
            if not seg.get("ok", True):
                errors_txt.append(txt)
            else:
                actions_txt.append(txt)

    # Se não houver ações, devolve um eco sucinto do prompt (fallback mínimo)
    if not actions_txt and not errors_txt:
        last_message = state["messages"][-1]
        text_content = extract_text_content(last_message)
        return {"context": ctx, "messages": AIMessage(content=(text_content or ""))}

    # Monta input para a LLM finalizadora
    parts = []
    if user_text:
        parts.append(f"Prompt do usuário: {user_text}")
    if actions_txt:
        parts.append("Ações executadas:\n" + "\n".join(f"- {t}" for t in actions_txt))
    if errors_txt:
        parts.append("Falhas:\n" + "\n".join(f"- {t}" for t in errors_txt))
    final_input = "\n\n".join(parts)

    final_msg = finalizer_model.invoke([SystemMessage(content=FINALIZER_SYSTEM_PROMPT), HumanMessage(content=final_input)])
    final_text = extract_text_content(final_msg) or ""
    # Limpa para o próximo turno
    ctx["ai_responses"] = []
    return {"context": ctx, "messages": AIMessage(content=final_text)}

# === Montagem do grafo ===
graph = StateGraph(AgentState)
# Nó: classifica a intenção e extrai slots
graph.add_node("parse_and_classify", parse_and_classify)
# Nó: valida e prepara slots/contexto para roteamento
graph.add_node("router", router_node)
# Nó: resolve lead quando necessário (fast‑path + fallback)
graph.add_node("resolve_lead", resolve_lead)
# Nó: executa intents de notas
graph.add_node("handle_notes", handle_notes)
# Nó: executa intents de tarefas
graph.add_node("handle_tasks", handle_tasks)

# === Subgrafos ReAct ===
# Nó: subgrafo ReAct para LEADS (usa tools de leads)
leads_agent_graph = new_react.create_react_executor(
    model=model,
    tools=[
        crm_tools.create_lead,
        crm_tools.get_lead,
        crm_tools.search_leads,
        crm_tools.list_leads,
        crm_tools.update_lead,
        crm_tools.resolve_lead,
    ],
    prompt=LEAD_REACT_PROMPT,
    prompt_vars=["intent", "slots", "lead_atual"],
)
graph.add_node("leads_agent", leads_agent_graph)

# Nó: subgrafo ReAct para PROPOSTAS (usa tools de propostas)
proposals_agent_graph = new_react.create_react_executor(
    model=proposals_model,
    tools=[
        crm_tools.draft_proposal,
        crm_tools.add_proposal_item,
        crm_tools.calculate_proposal_totals,
        crm_tools.list_proposals,
        crm_tools.export_proposal,
        crm_tools.update_proposal_body,
        crm_tools.resolve_lead,
    ],
    prompt=PROPOSAL_REACT_PROMPT,
    prompt_vars=["intent", "slots", "lead_atual"],
)
graph.add_node("proposals_agent", proposals_agent_graph)

# Nó: consolida contexto e extrai ações adicionais (nota/tarefa/proposta)
graph.add_node("update_context", update_context)
# Nó: produz a resposta final
graph.add_node("respond_final", respond_final)

# === Nó: execução de pendências ===
def execute_pending(state: AgentState) -> AgentState:
    """Nó: execute_pending
    - Remove a próxima ação de `context.pending_actions` e injeta em `intent/slots`
    - Em seguida, o fluxo retorna ao `router` para executar a tool apropriada
    """
    ctx = ensure_context(state)
    pending_actions = ctx.get("pending_actions") or []
    if not pending_actions:
        return {}
    action = pending_actions.pop(0)
    act_intent = action.get("intent")
    act_slots = dict(action.get("slots") or {})
    # injeta lead_ref do lead_atual se faltar
    if not act_slots.get("lead_ref_ou_id"):
        la = state.get("lead_atual")
        if la and la.get("lead_id"):
            act_slots["lead_ref_ou_id"] = str(la["lead_id"])
    ctx["pending_actions"] = pending_actions
    return {"context": ctx, "intent": act_intent, "slots": act_slots}

graph.add_node("execute_pending", execute_pending)

graph.add_edge(START, "parse_and_classify")
# Nó: planeja ações do turno e antecipa pré-requisitos
graph.add_node("prepare_plan", prepare_plan)
graph.add_edge("parse_and_classify", "prepare_plan")
graph.add_edge("prepare_plan", "router")

graph.add_conditional_edges(
    "router",
    route_intent,
    {
        "resolve_lead": "resolve_lead",
        "handle_leads": "leads_agent",
        "handle_notes": "handle_notes",
        "handle_tasks": "handle_tasks",
        "handle_proposals": "proposals_agent",
        "respond_final": "respond_final",
    },
)

graph.add_edge("resolve_lead", "router")
graph.add_edge("leads_agent", "update_context")
graph.add_edge("handle_notes", "update_context")
graph.add_edge("handle_tasks", "update_context")
graph.add_edge("proposals_agent", "update_context")

# Após atualizar contexto, execute ações pendentes em loop até acabar
graph.add_conditional_edges(
    "update_context",
    has_pending,
    {"execute": "execute_pending", "done": "respond_final"},
)
# Após buscar a próxima ação, volte para o roteador para executar a tool correta
graph.add_edge("execute_pending", "router")
graph.add_edge("respond_final", END)

compiled_graph = graph.compile(checkpointer=MemorySaver())
