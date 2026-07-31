import sys

from src.chatbot import RagChatbot, simple_rag_answer


def run_single(question: str) -> None:
    print(f"\nQ: {question}")
    print(f"A: {simple_rag_answer(question)}\n")


def run_interactive() -> None:
    bot = RagChatbot()
    print("RAG chatbot ready (with memory). Type 'exit' or 'quit' to stop, 'reset' to clear memory.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            bot.reset_memory()
            print("(memory cleared)\n")
            continue
        if not question:
            continue
        print(f"Bot: {bot.ask(question)}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(" ".join(sys.argv[1:]))
    else:
        run_interactive()