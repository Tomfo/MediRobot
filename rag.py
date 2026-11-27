"""Retriever pipeline with conversation history.

Flow per turn:
  1. Rewrite the follow-up question into a standalone question using chat history
  2. Retrieve top-k chunks from Pinecone for the standalone question
  3. Answer with the retrieved context + chat history (streamed)
  4. Append the turn to history (trimmed to the last N turns)
"""
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import config

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the user's latest question, rewrite the question so it "
     "can be understood on its own (resolve pronouns and references). Do NOT answer it. "
     "If it is already standalone, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful medicalassistant that answers questions using the provided context.\n"
     "- Use only the context below. If the answer is not in it, say you don't know.\n"
     "- Cite sources inline like [1], [2] using the numbers in the context.\n"
     "- Be concise.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        loc = f"{d.metadata.get('source', 'unknown')}" + (f", p.{int(page) + 1}" if page is not None else "")
        parts.append(f"[{i}] ({loc})\n{d.page_content}")
    return "\n\n".join(parts)


class RAGChat:
    def __init__(self, namespace: str = config.PINECONE_NAMESPACE, k: int = config.TOP_K):
        self.llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0, streaming=True)
        store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
            namespace=namespace,
        )
        self.retriever = store.as_retriever(search_kwargs={"k": k})
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        self.history: list[BaseMessage] = []
        self.last_sources: list[str] = []
        self.last_standalone_question: str = ""

    def reset(self) -> None:
        self.history.clear()
        self.last_sources = []

    def _standalone(self, question: str) -> str:
        if not self.history:
            return question
        return self.condense_chain.invoke({"history": self.history, "question": question}).strip()

    def stream(self, question: str) -> Iterator[str]:
        standalone = self._standalone(question)
        self.last_standalone_question = standalone

        docs = self.retriever.invoke(standalone)
        self.last_sources = [
            f"{d.metadata.get('source', 'unknown')}"
            + (f" (p.{int(d.metadata['page']) + 1})" if d.metadata.get("page") is not None else "")
            for d in docs
        ]

        messages = ANSWER_PROMPT.format_messages(
            context=format_docs(docs), history=self.history, question=question
        )
        answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            answer += token
            yield token

        self.history += [HumanMessage(question), AIMessage(answer)]
        self.history = self.history[-2 * config.MAX_HISTORY_TURNS :]

    def ask(self, question: str) -> dict:
        answer = "".join(self.stream(question))
        return {"answer": answer, "sources": self.last_sources}
