import json
import os
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from openai import OpenAI

TABLE_NAME = "demo_chunks"
SYSTEM_PROMPT = (
    "You are Pazu assistant. Use RAG context plus tools to answer."
    "Call tools when the user needs avatar items or actions."
)
PAZU_BASE_URL = os.environ.get("PAZU_BASE_URL", "http://127.0.0.1:8000")
DEBUG_MODE = os.environ.get("PAZU_DEBUG", "0") == "1"


def log_debug(message: str):
    if DEBUG_MODE:
        print(f"[debug] {message}")


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


def pazu_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = httpx.get(f"{PAZU_BASE_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def pazu_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(f"{PAZU_BASE_URL}{path}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


TOOLS = [
    {
        "name": "list_items",
        "type": "function",
        "function": {
            "name": "list_items",
            "description": "List avatar items filtered by character, category, or rarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "category": {"type": "string"},
                    "rarity": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "name": "create_item",
        "type": "function",
        "function": {
            "name": "create_item",
            "description": "Create a new avatar item in Pazu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "character": {"type": "string"},
                    "category": {"type": "string"},
                    "rarity": {"type": "string"},
                    "power": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "character", "category"],
                "additionalProperties": False,
            },
        },
    },
    {
        "name": "wear_item",
        "type": "function",
        "function": {
            "name": "wear_item",
            "description": "Trigger the wear action for a character and item ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "required": ["character", "item_id"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name: str, arguments_payload: Any) -> str:
    if isinstance(arguments_payload, str):
        args = json.loads(arguments_payload or "{}")
    else:
        args = arguments_payload or {}
    match name:
        case "list_items":
            result = pazu_get("/items", params=args)
        case "create_item":
            result = pazu_post("/items", payload=args)
        case "wear_item":
            result = pazu_post("/actions/wear", payload=args)
        case _:
            result = {"error": f"Unknown tool {name}"}
    return json.dumps(result)


def call_model(client: OpenAI, history: List[Dict[str, str]]):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        tools=TOOLS,
        tool_choice="auto",
    )
    return response


def chat():
    client = OpenAI()
    history: List[Dict[str, str]] = []
    print("Pazu tool-aware chat ready. Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text or user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        context = retrieve_context(user_text)
        prompt = user_text if not context else f"Context:\n{context}\n\nUser question: {user_text}"
        log_debug(f"user prompt ready. context_present={bool(context)} len(history)={len(history)}")
        history.append({"role": "user", "content": prompt})

        while True:
            response = call_model(client, history)
            outputs = getattr(response, "output", [])
            log_debug(f"model outputs: {[out.type for out in outputs]}")
            message_chunks: List[str] = []
            handled_tool = False

            for output in outputs:
                if output.type == "function_call":
                    handled_tool = True
                    tool_id = getattr(output, "id", getattr(output, "call_id", "tool-call"))
                    tool_name = output.name
                    arguments = getattr(output, "arguments", "{}")
                    log_debug(f"function_call -> {tool_name} args={arguments}")
                    tool_result = execute_tool(tool_name, arguments)
                    print(f"[Tool {tool_name} executed]")
                    history.append(
                        {
                            "role": "assistant",
                            "content": f"[ToolCall id={tool_id} name={tool_name}] args={arguments}",
                        }
                    )
                    history.append(
                        {
                            "role": "user",
                            "content": f"[ToolResult id={tool_id}] {tool_result}",
                        }
                    )
                    continue

                if output.type != "message":
                    continue
                for block in output.content:
                    block_type = getattr(block, "type", None)
                    if block_type in {"output_text", "text"} and hasattr(block, "text"):
                        message_chunks.append(block.text)
                    elif block_type == "tool_use":
                        handled_tool = True
                        tool_id = block.id
                        tool_name = block.name
                        arguments = block.arguments
                        log_debug(f"message tool_use -> {tool_name} args={arguments}")
                        tool_result = execute_tool(tool_name, arguments)
                        print(f"[Tool {tool_name} executed]")
                        history.append(
                            {
                                "role": "assistant",
                                "content": f"[ToolCall id={tool_id} name={tool_name}] args={arguments}",
                            }
                        )
                        history.append(
                            {
                                "role": "user",
                                "content": f"[ToolResult id={tool_id}] {tool_result}",
                            }
                        )

            if handled_tool:
                log_debug("tool handled; requesting follow-up")
                continue

            if message_chunks:
                message = "\n".join(message_chunks).strip()
                print(f"Assistant: {message}\n")
                history.append({"role": "assistant", "content": message})
                break

            print("Assistant: (no response)\n")
            history.append({"role": "assistant", "content": "(no response)"})
            break


if __name__ == "__main__":
    chat()