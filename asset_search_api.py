"""
asset_search_api.py — Phase 2: FastAPI backend for sprite semantic search

Endpoints:
    POST /search          — semantic search by free text
    GET  /asset/{id}      — fetch one asset's metadata
    GET  /group/{name}    — fetch all parts of a named asset group

Usage:
    uvicorn asset_search_api:app --reload --port 8001

Environment variables (same as rest of project):
    PGVECTOR_HOST / PGVECTOR_PORT / PGVECTOR_DB / PGVECTOR_USER / PGVECTOR_PASSWORD
"""

import os
from pathlib import Path
from typing import Optional

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 20

app = FastAPI(title="Asset Search API")

# Allow the local HTML UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sprite images as static files so the UI can display thumbnails.
# The path /images/* will serve files from the ASSETS_DIR env var.
ASSETS_DIR = os.environ.get("ASSETS_DIR", "")
if ASSETS_DIR and Path(ASSETS_DIR).is_dir():
    app.mount("/images", StaticFiles(directory=ASSETS_DIR), name="images")

# Load embedding model once at startup (cached in memory)
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# ── DB ────────────────────────────────────────────────────────────────────────
def db_connect():
    return psycopg.connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", 5433)),
        dbname=os.environ.get("PGVECTOR_DB", "rag_demo"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "ragpass"),
        row_factory=dict_row,
    )


def row_to_asset(row: dict) -> dict:
    """Convert a DB row to a clean API response dict."""
    filepath = row.get("filepath", "")
    # Build a URL the browser can use to display the thumbnail.
    # Works when ASSETS_DIR is set and StaticFiles is mounted at /images.
    thumbnail_url = None
    if ASSETS_DIR and filepath.startswith(ASSETS_DIR):
        rel = filepath[len(ASSETS_DIR):].lstrip("/")
        thumbnail_url = f"/images/{rel}"

    return {
        "id": row["id"],
        "filename": row["filename"],
        "filepath": filepath,
        "thumbnail_url": thumbnail_url,
        "group_name": row["group_name"],
        "is_part": row["is_part"],
        "part_number": row["part_number"],
        "description": row["description"],
        "tags": row["tags"] or [],
        "score": row.get("score"),
    }


# ── Request / Response models ─────────────────────────────────────────────────
class SearchRequest(BaseModel):
    q: str
    top_k: int = DEFAULT_TOP_K
    group_results: bool = True   # if True, group parts under their parent asset


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/search")
def search(payload: SearchRequest):
    """
    Semantic search over all ingested sprites.

    When group_results=True (default), results are returned as:
      {
        "results": [
          {
            "group_name": "closet",
            "is_part": true,
            "parts": [ {...asset...}, {...asset...} ],   # individual parts
            "best_score": 0.92
          },
          ...
        ]
      }

    Each item also includes standalone assets (is_part=False) as single-item groups.
    """
    q = payload.q.strip()
    if not q:
        raise HTTPException(400, "Query cannot be empty")

    embedder = get_embedder()
    query_vec = embedder.encode(q, normalize_embeddings=True).tolist()

    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT
                id, filename, filepath, group_name, is_part, part_number,
                description, tags,
                1 - (embedding <=> %s::vector) AS score
            FROM assets
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, payload.top_k),
        ).fetchall()

    if not rows:
        return {"results": [], "query": q}

    assets = [row_to_asset(r) for r in rows]

    if not payload.group_results:
        return {"results": assets, "query": q}

    # Group assets by group_name, collecting parts together
    groups: dict[str, dict] = {}
    for asset in assets:
        gname = asset["group_name"]
        if gname not in groups:
            groups[gname] = {
                "group_name": gname,
                "is_part": asset["is_part"],
                "parts": [],
                "best_score": 0.0,
            }
        groups[gname]["parts"].append(asset)
        if (asset["score"] or 0) > groups[gname]["best_score"]:
            groups[gname]["best_score"] = asset["score"]

    # Sort parts within each group by part_number
    for g in groups.values():
        g["parts"].sort(key=lambda a: (a["part_number"] is None, a["part_number"] or 0))

    # Sort groups by best score descending
    sorted_groups = sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)

    return {"results": sorted_groups, "query": q}


@app.get("/asset/{asset_id}")
def get_asset(asset_id: int):
    """Fetch a single asset by its DB id."""
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, filename, filepath, group_name, is_part, part_number,
                   description, tags
            FROM assets WHERE id = %s
            """,
            (asset_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Asset {asset_id} not found")
    return row_to_asset(row)


@app.get("/group/{group_name}")
def get_group(group_name: str):
    """Fetch all assets (parts or standalone) in a named group."""
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, filepath, group_name, is_part, part_number,
                   description, tags
            FROM assets
            WHERE group_name = %s
            ORDER BY part_number NULLS LAST
            """,
            (group_name,),
        ).fetchall()
    if not rows:
        raise HTTPException(404, f"Group '{group_name}' not found")
    return {"group_name": group_name, "parts": [row_to_asset(r) for r in rows]}


@app.get("/health")
def health():
    return {"status": "ok"}
