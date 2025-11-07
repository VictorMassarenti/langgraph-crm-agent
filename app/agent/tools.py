import os
import re
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()



def get_db_url() -> str:
    # Monta a URL a partir de variáveis explícitas (didático para os alunos).
    user = os.getenv("DB_USER")
    pwd = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    if not all([user, pwd, host, name]):
        raise RuntimeError("Defina DB_USER, DB_PASSWORD, DB_HOST, DB_NAME e opcionalmente DB_PORT/DB_SSLMODE")
    sslmode = os.getenv("DB_SSLMODE", "require")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}?sslmode={sslmode}"


def get_conn():
    return psycopg.connect(get_db_url())


UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def normalize_phone(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def is_uuid(s: str) -> bool:
    return bool(s) and bool(UUID_RE.match(s))


def resolve_lead_id_by_ref(cur, ref: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Resolve um lead a partir de uma referência natural (uuid/email/telefone/nome+empresa).
    Retorna (lead_id_ou_none, matches_para_desambiguacao).
    """
    if not ref:
        return None, []
    ref = ref.strip()
    if is_uuid(ref):
        cur.execute("select id, nome, email, empresa from public.leads where id = %s", (ref,))
        row = cur.fetchone()
        return (row[0], []) if row else (None, [])
    if "@" in ref:
        cur.execute(
            "select id, nome, email, empresa from public.leads where lower(email)=lower(%s)",
            (ref,),
        )
        row = cur.fetchone()
        return (row[0], []) if row else (None, [])
    digits = normalize_phone(ref)
    if len(digits) >= 8:
        cur.execute(
            "select id, nome, email, empresa from public.leads where regexp_replace(telefone, '[^0-9]', '', 'g')=%s",
            (digits,),
        )
        row = cur.fetchone()
        return (row[0], []) if row else (None, [])
    # Nome/empresa (busca broad)
    like = f"%{ref.lower()}%"
    cur.execute(
        """
        select id, nome, email, empresa
        from public.leads
        where lower(nome) like %s or lower(empresa) like %s
        order by criado_em desc
        limit 5
        """,
        (like, like),
    )
    rows = cur.fetchall() or []
    if len(rows) == 1:
        return rows[0][0], []
    return None, [
        {"lead_id": r[0], "nome": r[1], "email": r[2], "empresa": r[3]} for r in rows
    ]


@tool
def create_lead(
    nome: str,
    email: Optional[str] = None,
    telefone: Optional[str] = None,
    empresa: Optional[str] = None,
    origem: Optional[str] = None,
    status_codigo: str = "novo",
) -> Dict[str, Any]:
    """Cria um lead. Campos: nome (obrigatório), email, telefone, empresa, origem. 
    Usa status_codigo (default: novo)."""
    if not nome:
        return {"error": {"message": "Campo 'nome' é obrigatório."}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            # valida status
            cur.execute("select 1 from public.status_lead where codigo=%s", (status_codigo,))
            if not cur.fetchone():
                return {"error": {"message": f"status_codigo inválido: {status_codigo}"}}
            # checa unicidade (email/telefone)
            if email:
                cur.execute(
                    "select 1 from public.leads where lower(email)=lower(%s)", (email,)
                )
                if cur.fetchone():
                    return {"error": {"message": "Já existe lead com este email."}}
            if telefone:
                cur.execute(
                    "select 1 from public.leads where regexp_replace(telefone,'[^0-9]','','g')=%s",
                    (normalize_phone(telefone),),
                )
                if cur.fetchone():
                    return {"error": {"message": "Já existe lead com este telefone."}}
            cur.execute(
                """
                insert into public.leads (nome, email, telefone, empresa, origem, status_codigo)
                values (%s,%s,%s,%s,%s,%s)
                returning id
                """,
                (nome, email, telefone, empresa, origem, status_codigo),
            )
            lead_id = cur.fetchone()[0]
    return {"message": "Lead criado", "data": {"lead_id": lead_id, "nome": nome, "email": email, "empresa": empresa}}

# (renomeado: função exposta como create_lead)


@tool
def get_lead(ref: str) -> Dict[str, Any]:
    """Obtém um lead por referência (uuid/email/telefone/nome ou empresa). 
    Retorna lead_id e dados básicos."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, ref)
            if lead_id:
                cur.execute(
                    "select id, nome, email, telefone, empresa, status_codigo from public.leads where id=%s",
                    (lead_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"error": {"message": "Lead não encontrado"}}
                return {
                    "message": "Lead encontrado",
                    "data": {
                        "lead_id": row[0],
                        "nome": row[1],
                        "email": row[2],
                        "telefone": row[3],
                        "empresa": row[4],
                        "status_codigo": row[5],
                    },
                }
            if matches:
                return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
            return {"error": {"message": "Lead não encontrado"}}

# (renomeado: função exposta como get_lead)


@tool
def search_leads(consulta: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Procura leads por nome/empresa (LIKE). 
    Aceita paginação por limit/offset."""
    like = f"%{(consulta or '').lower()}%"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select id, nome, empresa from public.leads where lower(nome) like %s or lower(empresa) like %s order by criado_em desc limit %s offset %s",
                (like, like, limit, offset),
            )
            rows = cur.fetchall() or []
            cur.execute(
                "select count(*) from public.leads where lower(nome) like %s or lower(empresa) like %s",
                (like, like),
            )
            total = cur.fetchone()[0]
    items = [{"lead_id": r[0], "nome": r[1], "empresa": r[2]} for r in rows]
    return {"message": f"{len(items)} resultados", "data": {"items": items, "total": total}}

# (renomeado: função exposta como search_leads)


@tool
def list_leads(limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Lista leads recentes com paginação."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select id, nome, email, empresa, status_codigo from public.leads order by criado_em desc limit %s offset %s",
                (limit, offset),
            )
            rows = cur.fetchall() or []
            cur.execute("select count(*) from public.leads")
            total = cur.fetchone()[0]
    items = [
        {
            "lead_id": r[0],
            "nome": r[1],
            "email": r[2],
            "empresa": r[3],
            "status_codigo": r[4],
        }
        for r in rows
    ]
    return {"message": f"{len(items)} leads", "data": {"items": items, "total": total}}

# (renomeado: função exposta como list_leads)


@tool
def update_lead(
    lead_id: str,
    email: Optional[str] = None,
    telefone: Optional[str] = None,
    empresa: Optional[str] = None,
    status_codigo: Optional[str] = None,
    qualificado: Optional[bool] = None,
    ultimo_contato_em: Optional[str] = None,
    proxima_acao_em: Optional[str] = None,
) -> Dict[str, Any]:
    """Atualiza campos do lead. 
    Aceita: email, telefone, empresa, status_codigo, qualificado, ultimo_contato_em, proxima_acao_em."""
    if not is_uuid(lead_id):
        return {"error": {"message": "lead_id inválido"}}
    sets = []
    vals: List[Any] = []
    if email is not None:
        sets.append("email=%s")
        vals.append(email)
    if telefone is not None:
        sets.append("telefone=%s")
        vals.append(telefone)
    if empresa is not None:
        sets.append("empresa=%s")
        vals.append(empresa)
    if status_codigo is not None:
        sets.append("status_codigo=%s")
        vals.append(status_codigo)
    if qualificado is not None:
        sets.append("qualificado=%s")
        vals.append(qualificado)
    if ultimo_contato_em is not None:
        sets.append("ultimo_contato_em=%s")
        vals.append(ultimo_contato_em)
    if proxima_acao_em is not None:
        sets.append("proxima_acao_em=%s")
        vals.append(proxima_acao_em)
    if not sets:
        return {"error": {"message": "Nenhum campo para atualizar"}}
    sets.append("atualizado_em=now()")
    with get_conn() as conn:
        with conn.cursor() as cur:
            # valida status se informado
            if status_codigo is not None:
                cur.execute("select 1 from public.status_lead where codigo=%s", (status_codigo,))
                if not cur.fetchone():
                    return {"error": {"message": f"status_codigo inválido: {status_codigo}"}}
            sql = f"update public.leads set {' ,'.join(sets)} where id=%s returning id"
            cur.execute(sql, (*vals, lead_id))
            row = cur.fetchone()
            if not row:
                return {"error": {"message": "Lead não encontrado"}}
    return {"message": "Lead atualizado", "data": {"lead_id": lead_id}}

# (renomeado: função exposta como update_lead)


@tool
def add_note_to_lead(lead_ref_ou_id: str, texto: str) -> Dict[str, Any]:
    """Adiciona nota ao lead (ref: uuid/email/telefone/nome/empresa)."""
    if not texto:
        return {"error": {"message": "Texto é obrigatório"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
            if not lead_id:
                if matches:
                    return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                return {"error": {"message": "Lead não encontrado"}}
            cur.execute(
                "insert into public.notas_lead (lead_id, texto) values (%s,%s) returning id",
                (lead_id, texto),
            )
            note_id = cur.fetchone()[0]
    return {"message": "Nota adicionada", "data": {"note_id": note_id, "lead_id": lead_id}}

# (renomeado: função exposta como add_note_to_lead)


@tool
def list_notes(lead_ref_ou_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
    """Lista notas de um lead (ref: uuid/email/telefone/nome/empresa)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
            if not lead_id:
                if matches:
                    return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                return {"error": {"message": "Lead não encontrado"}}
            cur.execute(
                "select id, texto, criado_em from public.notas_lead where lead_id=%s order by criado_em desc limit %s offset %s",
                (lead_id, limit, offset),
            )
            rows = cur.fetchall() or []
            cur.execute(
                "select count(*) from public.notas_lead where lead_id=%s",
                (lead_id,),
            )
            total = cur.fetchone()[0]
    items = [{"note_id": r[0], "texto": r[1], "criado_em": r[2].isoformat()} for r in rows]
    return {"message": f"{len(items)} notas", "data": {"items": items, "total": total}}

# (renomeado: função exposta como list_notes)


@tool
def create_task(
    lead_ref_ou_id: str,
    titulo: str,
    data_limite: Optional[str] = None,
    tipo: str = "tarefa",
) -> Dict[str, Any]:
    """Cria tarefa para um lead (tipos: ligacao/email/reuniao/tarefa)."""
    if not titulo:
        return {"error": {"message": "Título é obrigatório"}}
    if tipo not in ("ligacao", "email", "reuniao", "tarefa"):
        return {"error": {"message": "tipo inválido"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
            if not lead_id:
                if matches:
                    return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                return {"error": {"message": "Lead não encontrado"}}
            cur.execute(
                """
                insert into public.tarefas_lead (lead_id, tipo, titulo, status, data_limite)
                values (%s,%s,%s,'aberta',%s) returning id
                """,
                (lead_id, tipo, titulo, data_limite),
            )
            tarefa_id = cur.fetchone()[0]
    return {"message": "Tarefa criada", "data": {"tarefa_id": tarefa_id, "lead_id": lead_id}}

# (renomeado: função exposta como create_task)


@tool
def complete_task(tarefa_id: str) -> Dict[str, Any]:
    """Conclui uma tarefa (status='concluida')."""
    if not is_uuid(tarefa_id):
        return {"error": {"message": "tarefa_id inválido"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update public.tarefas_lead set status='concluida', concluido_em=now() where id=%s returning id",
                (tarefa_id,),
            )
            row = cur.fetchone()
            if not row:
                return {"error": {"message": "Tarefa não encontrada"}}
    return {"message": "Tarefa concluída", "data": {"tarefa_id": tarefa_id, "status": "concluida"}}

# (renomeado: função exposta como complete_task)


@tool
def list_tasks(
    lead_ref_ou_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """Lista tarefas, opcionalmente filtrando por lead e status (aberta|concluida|cancelada)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id = None
            if lead_ref_ou_id:
                lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
                if not lead_id:
                    if matches:
                        return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                    return {"error": {"message": "Lead não encontrado"}}
            clauses = []
            vals: List[Any] = []
            if lead_id:
                clauses.append("lead_id=%s")
                vals.append(lead_id)
            if status:
                clauses.append("status=%s")
                vals.append(status)
            where = ("where " + " and ".join(clauses)) if clauses else ""
            cur.execute(
                f"select id, lead_id, titulo, status, data_limite, criado_em from public.tarefas_lead {where} order by criado_em desc limit %s offset %s",
                (*vals, limit, offset),
            )
            rows = cur.fetchall() or []
            cur.execute(
                f"select count(*) from public.tarefas_lead {where}",
                tuple(vals),
            )
            total = cur.fetchone()[0]
    items = [
        {
            "tarefa_id": r[0],
            "lead_id": r[1],
            "titulo": r[2],
            "status": r[3],
            "data_limite": r[4].isoformat() if r[4] else None,
            "criado_em": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]
    return {"message": f"{len(items)} tarefas", "data": {"items": items, "total": total}}

# (renomeado: função exposta como list_tasks)


@tool
def draft_proposal(lead_ref_ou_id: str, titulo: str, moeda: str = "BRL") -> Dict[str, Any]:
    """Cria uma proposta em rascunho para um lead (moeda padrão BRL)."""
    if not titulo:
        return {"error": {"message": "Título é obrigatório"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
            if not lead_id:
                if matches:
                    return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                return {"error": {"message": "Lead não encontrado"}}
            cur.execute(
                """
                insert into public.propostas (lead_id, titulo, moeda, subtotal, desconto_pct, total, status)
                values (%s,%s,%s,0,0,0,'rascunho') returning id
                """,
                (lead_id, titulo, moeda),
            )
            proposta_id = cur.fetchone()[0]
    return {"message": "Proposta criada", "data": {"proposta_id": proposta_id, "lead_id": lead_id}}

# (renomeado: função exposta como draft_proposal)


@tool
def add_proposal_item(
    proposta_id: str, descricao: str, quantidade: float, preco_unitario: float
) -> Dict[str, Any]:
    """Adiciona item a uma proposta e recalcula totais."""
    if not is_uuid(proposta_id):
        return {"error": {"message": "proposta_id inválido"}}
    if not descricao or quantidade is None or preco_unitario is None:
        return {"error": {"message": "Campos obrigatórios: descricao, quantidade, preco_unitario"}}
    total_linha = round((quantidade or 0) * (preco_unitario or 0), 2)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "insert into public.itens_proposta (proposta_id, descricao, quantidade, preco_unitario, total) values (%s,%s,%s,%s,%s) returning id",
                (proposta_id, descricao, quantidade, preco_unitario, total_linha),
            )
            item_id = cur.fetchone()[0]
            # recalcula
            recalc_proposal_totals_cur(cur, proposta_id)
    return {
        "message": "Item adicionado",
        "data": {"proposta_id": proposta_id, "item_id": item_id},
    }

# (renomeado: função exposta como add_proposal_item)


def recalc_proposal_totals_cur(cur, proposta_id: str) -> None:
    cur.execute(
        "select coalesce(sum(total),0) from public.itens_proposta where proposta_id=%s",
        (proposta_id,),
    )
    subtotal = float(cur.fetchone()[0] or 0)
    # desconto_pct está armazenado na proposta
    cur.execute(
        "select desconto_pct from public.propostas where id=%s",
        (proposta_id,),
    )
    row = cur.fetchone()
    desconto_pct = float(row[0] or 0) if row else 0
    total = round(subtotal * (1 - desconto_pct / 100.0), 2)
    cur.execute(
        "update public.propostas set subtotal=%s, total=%s, atualizado_em=now() where id=%s",
        (subtotal, total, proposta_id),
    )


@tool
def calculate_proposal_totals(proposta_id: str) -> Dict[str, Any]:
    """Recalcula e retorna totais de uma proposta (subtotal/total)."""
    if not is_uuid(proposta_id):
        return {"error": {"message": "proposta_id inválido"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            recalc_proposal_totals_cur(cur, proposta_id)
            cur.execute(
                "select subtotal, desconto_pct, total from public.propostas where id=%s",
                (proposta_id,),
            )
            row = cur.fetchone()
            if not row:
                return {"error": {"message": "Proposta não encontrada"}}
    return {
        "message": "Totais atualizados",
        "data": {"proposta_id": proposta_id, "subtotal": float(row[0]), "desconto_pct": float(row[1]), "total": float(row[2])},
    }

# (renomeado: função exposta como calculate_proposal_totals)


@tool
def list_proposals(lead_ref_ou_id: str) -> Dict[str, Any]:
    """Lista propostas de um lead (ref: uuid/email/telefone/nome/empresa)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, lead_ref_ou_id)
            if not lead_id:
                if matches:
                    return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
                return {"error": {"message": "Lead não encontrado"}}
            cur.execute(
                "select id, titulo, total, status from public.propostas where lead_id=%s order by criado_em desc",
                (lead_id,),
            )
            rows = cur.fetchall() or []
    items = [
        {"proposta_id": r[0], "titulo": r[1], "total": float(r[2] or 0), "status": r[3]}
        for r in rows
    ]
    return {"message": f"{len(items)} propostas", "data": {"items": items}}

# (renomeado: função exposta como list_proposals)


@tool
def update_proposal_body(proposta_id: str, corpo_md: str) -> Dict[str, Any]:
    """Atualiza o corpo textual (Markdown) de uma proposta existente."""
    if not is_uuid(proposta_id):
        return {"error": {"message": "proposta_id inválido"}}
    if corpo_md is None or str(corpo_md).strip() == "":
        return {"error": {"message": "corpo_md é obrigatório"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update public.propostas set corpo_md=%s, atualizado_em=now() where id=%s returning id",
                (corpo_md, proposta_id),
            )
            row = cur.fetchone()
            if not row:
                return {"error": {"message": "Proposta não encontrada"}}
    return {"message": "Corpo da proposta atualizado", "data": {"proposta_id": proposta_id}}

# (renomeado: função exposta como update_proposal_body)


@tool
def export_proposal(proposta_id: str, formato: str = "markdown") -> Dict[str, Any]:
    """Exporta proposta em markdown ou json (conteúdo retornado em data.conteudo)."""
    if not is_uuid(proposta_id):
        return {"error": {"message": "proposta_id inválido"}}
    formato = (formato or "markdown").lower()
    if formato not in ("markdown", "json"):
        return {"error": {"message": "Formato inválido (use markdown|json)"}}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select p.id, p.titulo, p.moeda, p.subtotal, p.desconto_pct, p.total, p.corpo_md, l.nome, l.empresa from public.propostas p join public.leads l on l.id=p.lead_id where p.id=%s",
                (proposta_id,),
            )
            head = cur.fetchone()
            if not head:
                return {"error": {"message": "Proposta não encontrada"}}
            cur.execute(
                "select descricao, quantidade, preco_unitario, total from public.itens_proposta where proposta_id=%s",
                (proposta_id,),
            )
            itens = cur.fetchall() or []
    corpo_md = head[6]
    lead_nome = head[7]
    lead_empresa = head[8]
    if formato == "json":
        return {
            "message": "Proposta exportada",
            "data": {
                "proposta_id": head[0],
                "titulo": head[1],
                "moeda": head[2],
                "subtotal": float(head[3] or 0),
                "desconto_pct": float(head[4] or 0),
                "total": float(head[5] or 0),
                "corpo_md": corpo_md,
                "lead": {"nome": lead_nome, "empresa": lead_empresa},
                "itens": [
                    {
                        "descricao": r[0],
                        "quantidade": float(r[1] or 0),
                        "preco_unitario": float(r[2] or 0),
                        "total": float(r[3] or 0),
                    }
                    for r in itens
                ],
            },
        }
    # markdown
    if corpo_md and str(corpo_md).strip():
        return {"message": "Proposta exportada", "data": {"proposta_id": proposta_id, "formato": "markdown", "conteudo": str(corpo_md)}}
    linhas = [f"# {head[1]}", f"Cliente: {lead_nome} ({lead_empresa})", "", "## Itens:"]
    for r in itens:
        linhas.append(f"- {r[0]} — {r[1]} x {r[2]} = {r[3]}")
    linhas += [
        "",
        f"Subtotal: {head[3]}",
        f"Desconto: {head[4]}%",
        f"Total: {head[5]} {head[2]}",
    ]
    return {"message": "Proposta exportada", "data": {"proposta_id": proposta_id, "formato": "markdown", "conteudo": "\n".join(linhas)}}

# (renomeado: função exposta como export_proposal)


@tool
def list_lead_status() -> Dict[str, Any]:
    """Lista os códigos de status de lead válidos (lookup)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select codigo, rotulo
                from public.status_lead
                order by array_position(array['novo','qualificado','desqualificado','negociacao','ganho','perdido'], codigo), codigo
                """
            )
            rows = cur.fetchall() or []
            cur.execute("select count(*) from public.status_lead")
            total = cur.fetchone()[0]
    items = [{"codigo": r[0], "rotulo": r[1]} for r in rows]
    return {"message": f"{len(items)} status", "data": {"items": items, "total": total}}

# (renomeado: função exposta como list_lead_status)


@tool
def resolve_lead(ref: str) -> Dict[str, Any]:
    """Resolve um lead por referência (uuid/email/telefone/nome/empresa). Retorna lead_id, ou lista para desambiguação."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            lead_id, matches = resolve_lead_id_by_ref(cur, ref)
            if lead_id:
                return {"message": "Lead resolvido", "data": {"lead_id": lead_id}}
            if matches:
                return {"error": {"message": "Mais de um lead corresponde", "matches": matches}}
            return {"error": {"message": "Lead não encontrado"}}

# (renomeado: função exposta como resolve_lead)


@tool
def respond_message(mensagem: str, intent: str, dados: Optional[dict] = None) -> Dict[str, Any]:
    """Padroniza a saída do agente: preencha mensagem, intent e dados opcionais."""
    return {"message": mensagem, "intent": intent, "data": dados or {}}

# (renomeado: função exposta como respond_message)
