# graph_pipeline.py
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, START, END  # :contentReference[oaicite:9]{index=9}

from agents import retrieval_agent, validation_agent, synthesis_agent


class PipelineState(TypedDict, total=False):
    query: str
    retrieved_docs: List[str]
    validated_docs: List[str]
    answer: str
    errors: List[str]


def build_pipeline():
    # define graph over our state
    graph = StateGraph(PipelineState)

    # add nodes
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("validation", validation_agent)
    graph.add_node("synthesis", synthesis_agent)

    # edges
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "validation")
    graph.add_edge("validation", "synthesis")
    graph.add_edge("synthesis", END)

    # compile to an executable graph
    app = graph.compile()
    return app


# tiny manual test
if __name__ == "__main__":
    g = build_pipeline()
    out: Dict[str, Any] = g.invoke({"query": "Summarize research trends in Agentic AI"})
    print(out["answer"])
