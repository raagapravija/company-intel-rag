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
    """Run each retriever the router chose. Phase 2/3 add text2cypher and graph."""
    from company_intel.retrieval.hybrid import hybrid_search

    contexts = []
    for name in state.get("retrievers", ["hybrid"]):
        if name == "hybrid":
            contexts += hybrid_search(state["question"])
        else:
            raise NotImplementedError(f"retriever {name!r} arrives in a later phase")
    return {"contexts": contexts}


def rerank_node(state: AgentState) -> AgentState:
    from company_intel.retrieval.rerank import rerank

    # Contexts accumulate across retries; drop duplicates before reranking.
    seen, unique = set(), []
    for c in state.get("contexts", []):
        key = c.metadata.get("chunk_id") or c.text
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return {"reranked": rerank(state["question"], unique)}


def generate(state: AgentState) -> AgentState:
    from company_intel.agent.answer import generate_answer

    answer, citations = generate_answer(state["question"], state.get("reranked", []))
    return {"answer": answer, "citations": citations}


def verify(state: AgentState) -> AgentState:
    """Phase 5 replaces this with an LLM claim-by-claim grounding check.
    For now: an answer with no citations counts as ungrounded."""
    return {"grounded": bool(state.get("citations"))}


def after_verify(state: AgentState) -> str:
    if state.get("grounded", True) or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END
    # Retrying only helps when the verifier says what to look for differently.
    if not state.get("missing_info"):
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
