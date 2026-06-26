"""LangGraph flow: question -> retrieve -> answer."""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from src.answer import answer as gemini_answer
from src.embedder import embed_query
from src.vector_store import search

class RAGState(TypedDict):
    question: str
    retrieved: list[dict]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Embed the question visually and pull the top matching pages."""
    query_vec = embed_query(state["question"])
    return {"retrieved": search(query_vec)}


def answer_node(state: RAGState) -> dict:
    """Ask Gemini to answer the question using the retrieved pages."""
    ans = gemini_answer(state["question"], state["retrieved"])
    return {"answer": ans}


def build_graph():
    """Compile the retrieve -> answer graph."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile()
