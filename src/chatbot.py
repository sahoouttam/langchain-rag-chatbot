from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.prompts import CHAT_RAG_PROMPT, SIMPLE_RAG_PROMPT
from src.vector_store import get_vector_db


class RagChatbot:

    def __init__(self):
        self.vector_db = get_vector_db()
        self.retriever = self.vector_db.as_retriever()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        self.chat_history = ChatMessageHistory()

        self.chain = (
                {
                    "retrieved_context": self.retriever,
                    "question": RunnablePassthrough(),
                    "chat_history_messages": RunnableLambda(lambda _: self.chat_history.messages)
                }
                | CHAT_RAG_PROMPT
                | self.llm
        )

    def ask(self, question: str) -> str:
        """Ask a question, updating and using conversational memory."""
        self.chat_history.add_user_message(question)
        answer = self.chain.invoke(question)
        self.chat_history.add_ai_message(answer)
        return answer.content

    def reset_memory(self) -> None:
        self.chat_history.clear()


def simple_rag_answer(question: str) -> str:
    """One-shot RAG query with no conversational memory - useful for scripts /
        single-question use where you don't need a persistent chatbot instance."""
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever()
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    chain = {"context": retriever, "question": RunnablePassthrough()} | SIMPLE_RAG_PROMPT | llm
    return chain.invoke(question).content
