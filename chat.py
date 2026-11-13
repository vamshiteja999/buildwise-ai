import sys
from buildwise.core import health_check, answer_question

def main():
    try:
        health_check()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    print("BuildWise construction-document chat. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue
        try:
            answer, _ = answer_question(question)
            print(f"\nBuildWise: {answer}")
        except Exception as exc:
            print(f"\nERROR: Unable to answer this question: {exc}")

if __name__ == "__main__":
    main()
