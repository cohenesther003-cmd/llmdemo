import os
from typing import List

from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

TABLE_NAME = "demo_chunks"


def connect_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=os.environ["PGVECTOR_DB"],
        host=os.environ["PGVECTOR_HOST"],
        password=os.environ["PGVECTOR_PASSWORD"],
        port=int(os.environ["PGVECTOR_PORT"]),
        user=os.environ["PGVECTOR_USER"],
        table_name=TABLE_NAME,
    )


def retrieve(query: str) -> List[NodeWithScore]:
    store = connect_store()
    storage = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_vector_store(store, storage_context=storage)
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve(query)
    for idx, node in enumerate(nodes, start=1):
        print(f"Hit {idx} (score {node.score:.3f}):\n{node.node.get_content()}\n")
    return nodes


def answer(query: str):
    nodes = retrieve(query)
    synthesizer = get_response_synthesizer(llm=OpenAI(model="gpt-4o-mini"))
    response = synthesizer.synthesize(query=query, nodes=nodes)
    print("Answer:\n", response.response)


if __name__ == "__main__":
    question = input("Ask a question about the demo topics: ").strip()
    if not question:
        raise SystemExit("No question provided.")
    answer(question)