# Pazu LLM Learning Track

This repo walks through a step-by-step LLM learning journey: embedding AvatarWorld docs into pgvector, exposing the Pazu backend API, and chatting via a tool-aware CLI. Follow the sections below to reproduce the full environment.

---

## 1. Prerequisites
- macOS/Linux shell with Docker Desktop.
- Python 3.12+ with `python3 -m venv`.
- OpenAI API key with access to `gpt-4o-mini`.
- Git repo cloned at `~/git/<reponame>` (adjust paths if different).

---

## 2. Environment Setup
```
cd ~/git/<reponame>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt  # or install individually (openai, llama-index, fastapi, uvicorn, httpx, psycopg[binary])
```

Export the shared env vars (add to `.env` for convenience):
```
export OPENAI_API_KEY="sk-...your key..."
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5433
export PGVECTOR_DB=rag_demo
export PGVECTOR_USER=postgres
export PGVECTOR_PASSWORD=ragpass
export PAZU_BASE_URL=http://127.0.0.1:8000
```

> Tip: create a `.env` file and `source .env` after activating the venv.

---

## 3. Start pgvector (Step 3 dependency)
Use the existing Docker image `pgvector/pgvector:pg17` (already pulled). If you need to recreate it:
```
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=ragpass \
  -p 5433:5432 \
  pgvector/pgvector:pg17
```

Create the DB once (only if not already created):
```
docker exec -it pgvector psql -U postgres -c "CREATE DATABASE rag_demo;"
```

### Re-ingest AvatarWorld docs into pgvector
```
source .venv/bin/activate
export OPENAI_API_KEY=...  # ensure it’s set
python step03_ingest.py
```
This loads the four files in `avatar_docs/` into the `demo_chunks` table.

> Re-run `step03_ingest.py` whenever you change the docs.

---

## 4. Run the Pazu backend (Step 5)
```
source .venv/bin/activate
uvicorn step05_pazu:app --reload --port 8000
```
The server exposes:
- `GET /items?character=Nova&category=dress`
- `POST /items` to add a new item (requires `name`, `character`, `category`; optional `rarity`, `power`)
- `POST /actions/wear` to equip an item (`{ "character": "...", "item_id": "..." }`)

Leave this server running while using the CLI.

---

## 5. Tool-aware CLI Chat (Step 6)
```
source .venv/bin/activate
PAZU_DEBUG=1 python step06_cli_chat_tools.py
```
The CLI:
1. Retrieves context from pgvector.
2. Lets OpenAI decide whether to call one of the tools:
   - `list_items(character?, category?, rarity?)`
   - `create_item(name, character, category, rarity?, power≥0)`
   - `wear_item(character, item_id)`
3. Executes the tool via the Pazu API and feeds the result back to the model.

### Example prompts
- “Skyhaven has a diplomatic mission tomorrow; what formal items does Nova already have?”
- “Create a Skyhaven accessory for Nova called Luminous Halo with power 18.”
- “Have Nova wear the Luminous Halo we just added.”
- “List any relic-grade armor for Orion, then summarize if it matches his faction rules.”
- “Lyra needs a flora-themed cloak with power 15; add it for her and equip it.”

Watch the logs when `PAZU_DEBUG=1` to see tool calls and results.

---

## 6. Complete Run Checklist
1. `docker ps` shows the `pgvector` container on port 5433.
2. `.venv` activated, env vars exported.
3. `python step03_ingest.py` has been run at least once for the latest docs.
4. `uvicorn step05_pazu:app --reload` running in a separate terminal.
5. `python step06_cli_chat_tools.py` running in another terminal for the interactive chat.

You now have a full AvatarWorld learning environment: embeddings in pgvector, a FastAPI backend acting as tools, and a CLI assistant that performs RAG + tool calls in response to user prompts. Experiment with more docs, new tool endpoints, or turning the CLI into a web UI as next steps.

