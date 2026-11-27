"""Terminal chatbot.  Run:  python chatbot.py"""
from rag import RAGChat

HELP = "Commands: /reset  clear history | /sources  show last sources | /quit  exit"


def main():
    chat = RAGChat()
    print("RAG chatbot ready.", HELP, sep="\n")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            chat.reset()
            print("History cleared.")
            continue
        if q == "/sources":
            print("\n".join(f"- {s}" for s in chat.last_sources) or "No sources yet.")
            continue

        print("Bot: ", end="", flush=True)
        for token in chat.stream(q):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
