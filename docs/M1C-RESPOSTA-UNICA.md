# Resposta Única para Multi‑intents

Este documento descreve como o agente agrega várias ações em um turno e produz uma resposta única, coerente e previsível para o usuário.

## Objetivo
Em interações com múltiplas intenções (ex.: “Cadastre o Lennon; adicione nota; crie tarefa; rascunhe a proposta…”), vários nós são executados. Deixar cada nó “falar” pode gerar respostas desconexas. A solução: centralizar a “narração” no nó `respond_final`.

## Como funciona
- Durante a execução, cada ação concluída gera um “segmento” — uma mini‑mensagem curta e determinística — e o agente acumula esses segmentos em `context.ai_responses`.
- Ao final do turno, `respond_final`:
  - Se a lista tiver 1 item, encaminha essa mensagem ao usuário.
  - Se houver mais de 1, cria uma resposta agregada do tipo “Ações realizadas: …” (e “Ocorreram falhas: …” se necessário).

## Onde os segmentos são criados
- `handle_notes` e `handle_tasks`: após cada tool call (sucesso/erro/listagem). Usam o helper `push_ai_response(ctx, intent, text, ok=True, data=None)`.
- `update_context`: varre as `ToolMessage` recentes (sem interromper na primeira) para extrair contexto e criar segmentos equivalentes. Regras de detecção principais:
  - Lead criado: quando a `ToolMessage.name` é `create_lead`/`criar_lead` ou o texto da mensagem contém “lead” e “criado”. O segmento gerado é “Lead criado: {nome} (id: {lead_id})”.
  - Proposta rascunhada/criada: quando a `ToolMessage.name` é `draft_proposal`/`rascunhar_proposta` ou o texto contém “proposta” e “criada/rascunho/rascunhada”. O segmento gerado é “Proposta rascunhada: {titulo} (id: {proposta_id})”.
  - A primeira referência de lead encontrada nas `ToolMessage` é usada para consolidar `lead_atual` e injetar `slots.lead_ref_ou_id` quando ausente.

## Planejamento (LLM‑first)
- O planejamento de multi‑intents é feito via LLM, com saída JSON `{"actions": [{intent, slots: ["{...}"]}, ...]}` (PlanAction).
- `prepare_plan` normaliza slots (strings de objeto → dict), deduplica e popula `context.pending_actions`.
- `update_context` replaneja no máximo 1× por turno caso a fila esteja vazia e ainda haja ações implícitas no pedido.

## Padrões de formatação
- Segmentos são curtos, determinísticos e com dados essenciais (ids, datas). Exemplos:
  - `lead_criar` → “Lead criado: {nome} (id: {lead_id})”
  - `nota_adicionar` → “Nota registrada: ‘{texto}’”
  - `tarefa_criar` → “Tarefa criada: {titulo} (id: {tarefa_id}, prazo: {data})”
  - `proposta_rascunhar` → “Proposta rascunhada: {titulo} (id: {proposta_id})”

## Separação de responsabilidades
- Execução: nós e tools (chamadas, validações, persistência)
- Narração: `respond_final` (usa `context.ai_responses`)

## Benefícios
- Resposta final sempre coerente com tudo que foi feito no turno.
- Previsibilidade (útil para testes) e clareza didática (separação execução vs. narração).
- Mantém rastreabilidade (cada segmento pode conter `data` com ids/detalhes).

## Onde ver no código
- `app/agent/helpers.py` — utilitários (`push_ai_response`, etc.)
- `app/agent/workflow.py` — nós do grafo (comentados em PT‑BR), `respond_final` agrega a resposta. Em `update_context`, a varredura de `ToolMessage` consolida `lead_atual`, injeta `lead_ref_ou_id` e cria segmentos de proposta (rascunho, item adicionado, totais atualizados).
- `app/agent/prompt.py` — prompts do planner (JSON), finalizer e ReAct.
- `app/agent/tools.py` — ferramentas do Mini‑CRM (nomes em inglês)

## Notas de implementação
- As `ToolMessage` podem trazer payloads como string que incluem `UUID('...')`. O código limpa o texto e usa `ast.literal_eval` com fallback seguro.
- A detecção de eventos não depende apenas do texto da mensagem: também considera o nome da tool (por exemplo, `create_lead` e `draft_proposal`), para maior robustez.
- Para propostas, além do rascunho, também criamos segmentos ao adicionar itens e ao recalcular totais. O segmento de rascunho inclui ao menos o `proposta_id`.
