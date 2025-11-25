import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI

TABLE_NAME = "demo_chunks"
app = FastAPI()
client = OpenAI()
TODO_LIST: List[str] = []


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None


class ToolCall(BaseModel):
    tool: str
    args: Dict[str, str]


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


def call_llm(question: str, history: List[Dict[str, str]], context: str) -> str:
    prompt = question
    if context:
        prompt = f"Context:\n{context}\n\nUser question: {question}"
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Use context when helpful."},
            *history,
            {"role": "user", "content": prompt},
        ],
    )
    return response.output[0].content[0].text


def tool_weather(city: str) -> Dict[str, str]:
    samples = {
        "rome": ("22°C", "sunny intervals"),
        "tel aviv": ("27°C", "humid and breezy"),
        "london": ("14°C", "light rain"),
    }
    temp, summary = samples.get(city.lower(), ("20°C", "clear skies"))
    return {"city": city, "temperature": temp, "summary": summary}


def tool_todo(action: str, item: Optional[str] = None) -> Dict[str, List[str]]:
    match action:
        case "add":
            if not item:
                raise HTTPException(400, "Missing item to add.")
            TODO_LIST.append(item)
        case "list":
            pass
        case _:
            raise HTTPException(400, "action must be 'add' or 'list'")
    return {"items": TODO_LIST}


TOOLS = {
    "weather": tool_weather,
    "todo": tool_todo,
}


@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    history = payload.history or []
    context = retrieve_context(payload.question)
    answer = call_llm(payload.question, history, context)
    return {"answer": answer, "context": context}


@app.post("/tools/{tool_name}")
def tool_endpoint(tool_name: str, payload: Dict[str, str]):
    tool = TOOLS.get(tool_name)
    if not tool:
        raise HTTPException(404, f"Unknown tool '{tool_name}'")
    return {"tool": tool_name, "result": tool(**payload)}


@app.post("/tool-call")
def tool_call(payload: ToolCall):
    tool = TOOLS.get(payload.tool)
    if not tool:
        raise HTTPException(404, f"Unknown tool '{payload.tool}'")
    return {"tool": payload.tool, "result": tool(**payload.args)}