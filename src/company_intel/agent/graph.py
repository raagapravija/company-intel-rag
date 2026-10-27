"""LangGraph wiring. Node bodies are filled in phase by phase; the shape is final.

    question -> route -> retrieve -> rerank -> generate -> verify -> END
                  ^                                          |
                  +------------ retry (not grounded) --------+
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from company_intel.agent.state import AgentState

MAX_ATTEMPTS = 2


def route(state: AgentState) -> AgentState:
    """Classify the question and pick retrievers. Phase 1: always hybrid."""
    return {
        "question_type": "lookup",
        "retrievers": ["hybrid"],
        "attempts": state.get("attempts", 0) + 1,
    }


def retrieve(state: AgentState) -> AgentState:
    raise NotImplementedError("Phase 1: call hybrid_search / text2cypher / graph_search")


def rerank_node(state: AgentState) -> AgentState:
    raise NotImplementedError("Phase 1: call retrieval.rerank.rerank")


def generate(state: AgentState) -> AgentState:
    raise NotImplementedError("Phase 1: answer with citations from reranked contexts")


def verify(state: AgentState) -> AgentState:
    raise NotImplementedError("Phase 5: check each claim is supported; set grounded/missing_info")


def after_verify(state: AgentState) -> str:
    if state.get("grounded", True) or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END
    return "route"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, ["route", END])
    return g.compile()
