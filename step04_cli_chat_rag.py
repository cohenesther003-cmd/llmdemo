import os
from typing import List, Dict

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI

TABLE_NAME = "demo_chunks"
SYSTEM_PROMPT = (
    "You are a concise assistant. Use provided context when relevant. "
    "If the context is unrelated, answer from general knowledge and say so."
)


def connect_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=os.environ["PGVECTOR_DB"],
        host=os.environ["PGVECTOR_HOST"],
        password=os.environ["PGVECTOR_PASSWORD"],
        port=int(os.environ["PGVECTOR_PORT"]),
        user=os.environ["PGVECTOR_USER"],
        table_name=TABLE_NAME,
    )


def retrieve_context(question: str) -> str:
    store = connect_store()
    storage = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_vector_store(store, storage_context=storage)
    nodes = index.as_retriever(similarity_top_k=2).retrieve(question)
    if not nodes:
        return ""
    chunks = [f"Context {i+1}: {node.node.get_content()}" for i, node in enumerate(nodes)]
    return "\n\n".join(chunks)


def generate_reply(
    client: OpenAI, history: List[Dict[str, str]], user_text: str, context: str
) -> str:
    prompt = user_text
    if context:
        prompt = f"Context:\n{context}\n\nUser question: {user_text}"
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ],
    )
    return response.output[0].content[0].text


def chat():
    client = OpenAI()
    history: List[Dict[str, str]] = []
    print("RAG chat ready. Type 'exit' to quit.\n")
    while True:
        user_text = input("You: ").strip()
        if not user_text or user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        context = retrieve_context(user_text)
        reply = generate_reply(client, history, user_text, context)
        print(f"Assistant: {reply}\n")
        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ]
        )


if __name__ == "__main__":
    chat()