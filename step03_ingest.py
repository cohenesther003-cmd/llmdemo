from pathlib import Path
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

DOCS_PATH = Path("avatar_docs")
TABLE_NAME = "demo_chunks"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # matches text-embedding-3-small


def connect_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=os.environ["PGVECTOR_DB"],
        host=os.environ["PGVECTOR_HOST"],
        password=os.environ.get("PGVECTOR_PASSWORD", ""),
        port=int(os.environ["PGVECTOR_PORT"]),
        user=os.environ["PGVECTOR_USER"],
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
    )


def ingest():
    if not DOCS_PATH.exists():
        raise SystemExit(f"Directory {DOCS_PATH} missing")

    docs = SimpleDirectoryReader(str(DOCS_PATH)).load_data()
    if not docs:
        raise SystemExit("No documents found to ingest")

    vector_store = connect_vector_store()
    storage = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage,
        embed_model=OpenAIEmbedding(model=EMBED_MODEL),
        llm=OpenAI(model="gpt-4o-mini"),
    )
    print(f"Ingested {len(docs)} documents into table '{TABLE_NAME}'")
    return index


if __name__ == "__main__":
    ingest()