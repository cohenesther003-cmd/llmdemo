"""
asset_search_api.py — Phase 2: FastAPI search API backed by ChromaDB

No Docker needed. ChromaDB reads from the local ./chroma_db folder.

Endpoints:
    POST /search          — semantic search by free text
    GET  /asset/{id}      — fetch one asset by its filepath ID
    GET  /group/{name}    — fetch all parts of a named asset group

Usage:
    ASSETS_DIR=/path/to/sprites uvicorn asset_search_api:app --reload --port 8001

Environment variables:
    ASSETS_DIR   — absolute path to sprite folder (used to serve thumbnail images)
    CHROMA_DIR   — ChromaDB data folder (default: ./chroma_db)
"""

import os
from pathlib import Path
from urllib.parse import quote

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import HashingVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIMS = 384
COLLECTION_NAME = "assets"
DEFAULT_TOP_K = 20

# Must match the vectorizer used in asset_ingest.py
_local_embedder = HashingVectorizer(
    n_features=EMBED_DIMS,
    norm="l2",
    analyzer="word",
    ngram_range=(1, 2),
)


def local_embed(text: str) -> list:
    return _local_embedder.transform([text]).toarray()[0].tolist()

app = FastAPI(title="Asset Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sprite images so the browser can display thumbnails
ASSETS_DIR = os.environ.get("ASSETS_DIR", "").rstrip("/")
if ASSETS_DIR and Path(ASSETS_DIR).is_dir():
    app.mount("/images", StaticFiles(directory=ASSETS_DIR), name="images")

# ── Singletons loaded once at startup ────────────────────────────────────────
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        chroma_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Helpers ───────────────────────────────────────────────────────────────────
def meta_to_asset(meta: dict, doc_id: str, score: float | None = None) -> dict:
    """Convert ChromaDB metadata row into a clean API response dict."""
    filepath = meta.get("filepath", "")
    thumbnail_url = None
    if ASSETS_DIR and filepath.startswith(ASSETS_DIR):
        rel = filepath[len(ASSETS_DIR):].lstrip("/")
        thumbnail_url = f"/images/{quote(rel)}"

    is_part_str = meta.get("is_part", "False")
    part_number = meta.get("part_number", -1)

    return {
        "id": doc_id,
        "filename": meta.get("filename", ""),
        "filepath": filepath,
        "thumbnail_url": thumbnail_url,
        "group_name": meta.get("group_name", ""),
        "is_part": is_part_str == "True",
        "part_number": part_number if part_number != -1 else None,
        "description": meta.get("description", ""),
        "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
        "score": round(score, 4) if score is not None else None,
    }


# ── Request model ─────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    q: str
    top_k: int = DEFAULT_TOP_K
    group_results: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/search")
def search(payload: SearchRequest):
    """
    Semantic search over all ingested sprites.

    Returns results grouped by asset name. For example, searching "closet"
    returns one group called "closet" containing all its individual part sprites.
    """
    q = payload.q.strip()
    if not q:
        raise HTTPException(400, "Query cannot be empty")

    query_vec = local_embed(q)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(payload.top_k, collection.count() or 1),
        include=["metadatas", "distances", "documents"],
    )

    ids       = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance (lower = more similar)

    if not ids:
        return {"results": [], "query": q}

    # Convert distance → similarity score (0–1, higher = better)
    assets = [
        meta_to_asset(meta, doc_id, score=1 - dist)
        for doc_id, meta, dist in zip(ids, metadatas, distances)
    ]

    if not payload.group_results:
        return {"results": assets, "query": q}

    # Group by group_name
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

    sorted_groups = sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)
    return {"results": sorted_groups, "query": q}


@app.get("/asset/{asset_id:path}")
def get_asset(asset_id: str):
    """Fetch a single asset by its filepath (used as ID)."""
    collection = get_collection()
    result = collection.get(ids=[asset_id], include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(404, f"Asset not found: {asset_id}")
    return meta_to_asset(result["metadatas"][0], asset_id)


@app.get("/group/{group_name}")
def get_group(group_name: str):
    """Fetch all sprites belonging to a named group (e.g. 'closet')."""
    collection = get_collection()
    result = collection.get(
        where={"group_name": group_name},
        include=["metadatas"],
    )
    if not result["ids"]:
        raise HTTPException(404, f"Group '{group_name}' not found")
    parts = [
        meta_to_asset(meta, doc_id)
        for doc_id, meta in zip(result["ids"], result["metadatas"])
    ]
    parts.sort(key=lambda a: (a["part_number"] is None, a["part_number"] or 0))
    return {"group_name": group_name, "parts": parts}


@app.get("/health")
def health():
    collection = get_collection()
    return {"status": "ok", "total_assets": collection.count()}
