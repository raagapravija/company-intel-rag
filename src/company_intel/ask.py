"""Ask a question from the terminal:  uv run python -m company_intel.ask "your question" """

from __future__ import annotations

import sys

from company_intel.agent.graph import build_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m company_intel.ask "What does NVIDIA say about supply-chain risk?"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    result = build_graph().invoke({"question": question})
    print(result.get("answer", ""))
    if result.get("citations"):
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['n']}] {c['label']}  {c.get('url', '')}")


if __name__ == "__main__":
    main()
