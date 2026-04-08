"""Demo script — CLI interaction with the Forge agent."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

load_dotenv()

from forge_ai import ForgeAgent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Forge AI — ML Assistant Demo")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--provider", type=str, default="groq", help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    print("🔨 Forge AI — ML Engineering Assistant")
    print(f"   Provider: {args.provider}")
    print("=" * 50)

    agent = ForgeAgent(provider=args.provider, model=args.model, verbose=True)

    if args.query:
        # Single query mode
        kwargs = {"file_path": args.dataset} if args.dataset else {}
        result = agent.run(args.query, **kwargs)
        print("\n" + result["output"])
        return

    if args.interactive:
        # Interactive REPL
        print("Type 'quit' to exit, 'clear' to reset memory.\n")
        while True:
            try:
                query = input("\n🧑 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "clear":
                agent.reset_memory()
                print("Memory cleared.")
                continue
            if not query:
                continue

            kwargs = {"file_path": args.dataset} if args.dataset else {}
            result = agent.run(query, **kwargs)
            print(f"\n🔨 Forge: {result['output']}")

    else:
        # Default demo queries
        demo_queries = [
            "What's the best approach for handling missing values in a dataset with 10% nulls?",
            "Explain cross-validation and when I should use stratified K-fold",
            "Generate a classification pipeline for predicting customer churn",
        ]

        for query in demo_queries:
            print(f"\n🧑 You: {query}")
            result = agent.run(query)
            print(f"\n🔨 Forge: {result['output']}")
            print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
