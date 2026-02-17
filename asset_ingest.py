"""
asset_ingest.py — Phase 1: Scan sprites → Gemini description → embed → store in pgvector

Usage:
    GEMINI_API_KEY=your_key python asset_ingest.py --assets-dir /path/to/sprites

Environment variables (same as rest of project):
    GEMINI_API_KEY   — Google Gemini API key (get free key at aistudio.google.com)
    PGVECTOR_HOST    — default: localhost
    PGVECTOR_PORT    — default: 5433
    PGVECTOR_DB      — default: rag_demo
    PGVECTOR_USER    — default: postgres
    PGVECTOR_PASSWORD— default: ragpass
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import google.generativeai as genai
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, free, local
EMBED_DIM = 384
GEMINI_MODEL = "gemini-1.5-flash"

# Regex: detect a trailing numeric suffix like _1, _2, _01 etc.
PART_PATTERN = re.compile(r"^(.+?)_(\d+)$")


# ── Helpers ──────────────────────────────────────────────────────────────────
def db_connect():
    return psycopg.connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", 5433)),
        dbname=os.environ.get("PGVECTOR_DB", "rag_demo"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "ragpass"),
        row_factory=dict_row,
    )


def parse_group(stem: str) -> tuple[str, bool, int | None]:
    """
    Return (group_name, is_part, part_number) for a filename stem.

    Examples:
        closet_1  → ("closet", True, 1)
        closet_02 → ("closet", True, 2)
        sword     → ("sword",  False, None)
    """
    m = PART_PATTERN.match(stem)
    if m:
        return m.group(1), True, int(m.group(2))
    return stem, False, None


def describe_with_gemini(image_path: Path, client: genai.GenerativeModel) -> tuple[str, list[str]]:
    """
    Send the image to Gemini Flash and get back a description + tags.
    Returns (description, tags_list).
    """
    prompt = (
        "You are describing a 2D game sprite for a semantic search index.\n"
        "Look at this image and respond in this exact format:\n\n"
        "DESCRIPTION: <one or two sentences describing the sprite: what it is, "
        "its colors, style, and any notable visual features>\n"
        "TAGS: <comma-separated keywords: object type, colors, style, category, "
        "any identifiable parts or themes>\n\n"
        "Be specific and visual. Do not say 'the image shows' — just describe directly."
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(image_path.suffix.lower(), "image/png")

    response = client.generate_content(
        [{"mime_type": mime, "data": image_data}, prompt]
    )
    text = response.text.strip()

    description, tags = "", []
    for line in text.splitlines():
        if line.startswith("DESCRIPTION:"):
            description = line.removeprefix("DESCRIPTION:").strip()
        elif line.startswith("TAGS:"):
            raw_tags = line.removeprefix("TAGS:").strip()
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

    return description, tags


def already_ingested(conn, filepath: str) -> bool:
    """Check if this file path is already in the DB (allows re-running safely)."""
    row = conn.execute(
        "SELECT id FROM assets WHERE filepath = %s LIMIT 1", (filepath,)
    ).fetchone()
    return row is not None


def insert_asset(conn, *, filename, filepath, group_name, is_part, part_number,
                 description, tags, embedding):
    conn.execute(
        """
        INSERT INTO assets
            (filename, filepath, group_name, is_part, part_number,
             description, tags, embedding)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (filename, filepath, group_name, is_part, part_number,
         description, tags, embedding.tolist()),
    )


# ── Main ingestion pipeline ───────────────────────────────────────────────
def ingest(assets_dir: Path, skip_existing: bool = True):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY environment variable not set.\n"
                 "Get a free key at https://aistudio.google.com/app/apikey")

    # Collect all image files
    image_files = [
        p for p in sorted(assets_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        sys.exit(f"No image files found in {assets_dir}")

    print(f"Found {len(image_files)} image(s) in {assets_dir}")

    # Load models
    print("Loading sentence-transformers embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Connecting to Gemini ({GEMINI_MODEL})...")
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(GEMINI_MODEL)

    print("Connecting to pgvector...")
    conn = db_connect()

    ok = skipped = failed = 0

    for idx, img_path in enumerate(image_files, 1):
        abs_path = str(img_path.resolve())
        stem = img_path.stem                           # e.g. "closet_1"
        group_name, is_part, part_number = parse_group(stem)

        print(f"\n[{idx}/{len(image_files)}] {img_path.name}", end="  ")

        if skip_existing and already_ingested(conn, abs_path):
            print("(already ingested, skipping)")
            skipped += 1
            continue

        # 1. Describe with Gemini
        try:
            description, tags = describe_with_gemini(img_path, gemini)
        except Exception as e:
            print(f"GEMINI ERROR: {e}")
            failed += 1
            time.sleep(2)   # back off before next request
            continue

        if not description:
            print("WARNING: Gemini returned empty description, skipping.")
            failed += 1
            continue

        # 2. Build the text to embed: description + tags + filename
        embed_text = f"{description} Tags: {', '.join(tags)}. File: {stem.replace('_', ' ')}"

        # 3. Embed
        embedding = embedder.encode(embed_text, normalize_embeddings=True)

        # 4. Store
        insert_asset(
            conn,
            filename=img_path.name,
            filepath=abs_path,
            group_name=group_name,
            is_part=is_part,
            part_number=part_number,
            description=description,
            tags=tags,
            embedding=embedding,
        )
        conn.commit()

        print(f"OK  group={group_name!r}  part={part_number}  tags={tags[:4]}")
        ok += 1

        # Gemini free tier: 15 requests/min → add small delay
        time.sleep(1)

    conn.close()
    print(f"\n{'─'*50}")
    print(f"Done. Ingested: {ok}  Skipped: {skipped}  Failed: {failed}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest game sprites into vector DB")
    parser.add_argument(
        "--assets-dir", required=True, type=Path,
        help="Root folder containing your sprite images"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-ingest files that are already in the DB"
    )
    args = parser.parse_args()

    if not args.assets_dir.is_dir():
        sys.exit(f"ERROR: {args.assets_dir} is not a directory")

    ingest(args.assets_dir, skip_existing=not args.no_skip)
