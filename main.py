"""Query CLI: ask a question, get an answer read off the page images."""

import sys

from src.graph import build_graph
from src.vector_store import close_client

def run(question: str) -> None:
    """Run one question through the retrieve -> answer graph and print it."""
    graph = build_graph()
    try:
        result = graph.invoke({"question": question})
    finally:
        close_client()
    print("\n" + "=" * 60 + "\nRETRIEVED PAGES\n" + "=" * 60)
    for hit in result["retrieved"]:
        print(f"{hit['pdf']}- page{hit['page_number']} (score {hit['score']})")
    print("\n" + "=" * 60 + "\nANSWER\n" + "=" * 60)
    print(result["answer"] + "\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: PYTHONPATH=. uv run python main.py "your question"')
        sys.exit(1)
run(" ".join(sys.argv[1:]))