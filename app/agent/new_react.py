from typing import Iterable, Annotated, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage





# =============================
# Estado MANUAL (TypedDict)
# =============================
class AppState(TypedDict, total=False):
    # histórico de mensagens acumulado pelo add_messages
    messages: Annotated[List[AnyMessage], add_messages]
    # prompt resolvido/opcional a ser injetado como SystemMessage
    react_prompt: str




# =============================
# Grafo ReAct com estado manual
# =============================
def create_react_executor(
    model,
    tools: Iterable,
    prompt: Optional[str] = None,
    prompt_vars: Optional[List[str]] = None,
):
    """Executor ReAct: agent -> tools -> agent (até não haver tool_calls)."""
    tool_list = list(tools)
    tool_node = ToolNode(tool_list)
    bound_model = model.bind_tools(tool_list)

    def prepare(state: AppState):
        if not prompt:
            return {}
        prompt_text = prompt
        if prompt_vars:
            try:
                values = {k: state.get(k) for k in prompt_vars}
                prompt_text = prompt.format(**values)
            except Exception:
                prompt_text = prompt
        return {"react_prompt": prompt_text}

    def call_model(state: AppState):
        # invoque com o histórico atual; retorne SÓ a nova mensagem
        msgs = state["messages"]
        prompt_text = state.get("react_prompt") if isinstance(state, dict) else None
        if prompt_text:
            msgs = [SystemMessage(content=prompt_text), *msgs]
        response = bound_model.invoke(msgs)
        return {
            "messages": [response],           # add_messages cuida do append
        }

    def should_continue(state: AppState) -> str:
        last: BaseMessage = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "continue"
        return "end"

    graph = StateGraph(AppState)
    graph.add_node("agent", call_model)
    graph.add_node("prepare", prepare)
    graph.add_node("tools", tool_node)

    # Se houver prompt dinâmico, entra pelo nó de preparação
    graph.set_entry_point("prepare" if prompt else "agent")
    graph.add_edge("prepare", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    # Finaliza no nó agent para retornar a última resposta do modelo
    graph.set_finish_point("agent")
    return graph.compile()
