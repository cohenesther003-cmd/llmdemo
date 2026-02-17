"""
asset_ingest.py — Phase 1: Scan sprites → description → embed → store in ChromaDB

No Docker or external services needed. ChromaDB saves to a local folder on disk.

Usage:
    python asset_ingest.py --assets-dir /path/to/sprites

    # With Gemini vision (requires GEMINI_API_KEY and internet access):
    GEMINI_API_KEY=your_key python asset_ingest.py --assets-dir /path/to/sprites --use-gemini

Environment variables:
    GEMINI_API_KEY   — Google Gemini API key (optional, only needed with --use-gemini)
    CHROMA_DIR       — folder where ChromaDB stores its data (default: ./chroma_db)
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import HashingVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
GEMINI_MODEL = "gemini-1.5-flash"
COLLECTION_NAME = "assets"
EMBED_DIMS = 384   # HashingVectorizer output size

# Detects trailing numeric suffix: closet_1, sword_02, etc.
PART_PATTERN = re.compile(r"^(.+?)_(\d+)$")

# ── Local embedder (no download needed) ──────────────────────────────────────
_local_embedder = HashingVectorizer(
    n_features=EMBED_DIMS,
    norm="l2",
    analyzer="word",
    ngram_range=(1, 2),
)


def local_embed(text: str) -> list[float]:
    """Embed text using a hashing trick — deterministic, no model download."""
    return _local_embedder.transform([text]).toarray()[0].tolist()


# ── Fallback descriptions from filename ──────────────────────────────────────
# These are used when --use-gemini is not set or Gemini fails.
_KNOWN_ASSETS: dict[str, tuple[str, list[str]]] = {
    "sword":   ("A shiny silver sword with a metallic blade and golden crossguard.",
                ["sword", "weapon", "blade", "metal", "silver", "combat", "rpg"]),
    "closet":  ("A wooden closet or wardrobe with two doors, used for storing items.",
                ["closet", "wardrobe", "furniture", "storage", "wooden", "doors", "brown"]),
    "tree":    ("A tall green tree with a round leafy canopy and a brown trunk.",
                ["tree", "nature", "green", "plant", "outdoor", "forest", "foliage"]),
    "chest":   ("A golden treasure chest with a decorative lock, used to hold loot.",
                ["chest", "treasure", "gold", "loot", "container", "rpg", "storage"]),
    "coin":    ("A shiny gold coin used as in-game currency for purchasing items.",
                ["coin", "gold", "currency", "money", "collectible", "round", "reward"]),
    "shield":  ("A blue defensive shield used to block attacks and protect the player.",
                ["shield", "defense", "blue", "armor", "protection", "combat", "round"]),
    "potion":  ("A small glass bottle filled with a glowing red healing potion.",
                ["potion", "healing", "red", "bottle", "magic", "rpg", "consumable"]),
    "key":     ("A golden key used to unlock doors, chests, and secret passages.",
                ["key", "gold", "unlock", "door", "item", "rpg", "access"]),
}


def describe_from_filename(stem: str) -> tuple[str, list[str]]:
    """Return (description, tags) using the filename stem as a lookup key."""
    group_name, _, _ = parse_group(stem)
    if group_name in _KNOWN_ASSETS:
        return _KNOWN_ASSETS[group_name]
    readable = stem.replace("_", " ")
    return (
        f"A 2D game sprite: {readable}.",
        [w for w in readable.split() if len(w) > 2],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_group(stem: str) -> tuple[str, bool, int | None]:
    """
    Return (group_name, is_part, part_number) for a filename stem.

    closet_1  → ("closet", True, 1)
    closet_02 → ("closet", True, 2)
    sword     → ("sword",  False, None)
    """
    m = PART_PATTERN.match(stem)
    if m:
        return m.group(1), True, int(m.group(2))
    return stem, False, None



def describe_with_gemini(image_path: Path, model) -> tuple[str, list[str]]:
    """Call Gemini Flash with the sprite image → (description, tags).
    Only used when --use-gemini flag is passed."""
    prompt = (
        "You are describing a 2D game sprite for a semantic search index.\n"
        "Look at this image and respond in this exact format:\n\n"
        "DESCRIPTION: <one or two sentences describing the sprite: what it is, "
        "its colors, style, and any notable visual features>\n"
        "TAGS: <comma-separated keywords: object type, colors, style, category, "
        "any identifiable parts or themes>\n\n"
        "Be specific and visual. Do not say 'the image shows' — just describe directly."
    )

    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime = mime_map.get(image_path.suffix.lower(), "image/png")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = model.generate_content([{"mime_type": mime, "data": image_bytes}, prompt])
    text = response.text.strip()

    description, tags = "", []
    for line in text.splitlines():
        if line.startswith("DESCRIPTION:"):
            description = line.removeprefix("DESCRIPTION:").strip()
        elif line.startswith("TAGS:"):
            raw = line.removeprefix("TAGS:").strip()
            tags = [t.strip().lower() for t in raw.split(",") if t.strip()]

    return description, tags


# ── Main pipeline ─────────────────────────────────────────────────────────────
def ingest(assets_dir: Path, skip_existing: bool = True, use_gemini: bool = False):
    # Set up vision model (optional)
    gemini = None
    if use_gemini:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: --use-gemini requires GEMINI_API_KEY environment variable.\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        print(f"Connecting to Gemini ({GEMINI_MODEL})…")
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(GEMINI_MODEL)
    else:
        print("Using local filename-based descriptions (no Gemini API needed).")

    # Collect image files
    image_files = sorted(
        p for p in assets_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        sys.exit(f"No image files found in {assets_dir}")
    print(f"Found {len(image_files)} image(s) in {assets_dir}")
    print("Using local HashingVectorizer for embeddings (no model download needed).")

    # Open ChromaDB (creates ./chroma_db folder if it doesn't exist)
    chroma_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
    print(f"Opening ChromaDB at {chroma_dir!r}…")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    # Get already-ingested IDs so we can skip them
    existing_ids = set(collection.get(include=[])["ids"])

    ok = skipped = failed = 0

    for idx, img_path in enumerate(image_files, 1):
        abs_path = str(img_path.resolve())
        doc_id = abs_path                     # use full path as unique ID
        stem = img_path.stem
        group_name, is_part, part_number = parse_group(stem)

        print(f"\n[{idx}/{len(image_files)}] {img_path.name}", end="  ")

        if skip_existing and doc_id in existing_ids:
            print("(already ingested, skipping)")
            skipped += 1
            continue

        # 1. Describe the sprite
        if gemini is not None:
            try:
                description, tags = describe_with_gemini(img_path, gemini)
                time.sleep(1)   # Gemini free tier rate limit
            except Exception as e:
                print(f"GEMINI ERROR: {e} — falling back to filename description")
                description, tags = describe_from_filename(stem)
        else:
            description, tags = describe_from_filename(stem)

        if not description:
            print("WARNING: empty description, skipping.")
            failed += 1
            continue

        # 2. Build embed text: description + tags + readable filename
        embed_text = (
            f"{description} "
            f"Tags: {', '.join(tags)}. "
            f"File: {stem.replace('_', ' ')}"
        )

        # 3. Embed locally (HashingVectorizer — no API call, no rate limit)
        embedding = local_embed(embed_text)

        # 4. Store in ChromaDB
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{
                "filename": img_path.name,
                "filepath": abs_path,
                "group_name": group_name,
                "is_part": str(is_part),          # ChromaDB metadata must be str/int/float
                "part_number": part_number if part_number is not None else -1,
                "tags": ", ".join(tags),           # store as comma-joined string
                "description": description,
            }],
        )

        print(f"OK  group={group_name!r}  part={part_number}  tags={tags[:4]}")
        ok += 1

    print(f"\n{'─'*50}")
    print(f"Done. Ingested: {ok}  Skipped: {skipped}  Failed: {failed}")
    print(f"ChromaDB collection '{COLLECTION_NAME}' now has "
          f"{collection.count()} entries.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest game sprites into ChromaDB")
    parser.add_argument("--assets-dir", required=True, type=Path,
                        help="Root folder containing your sprite images")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-ingest files already in the DB")
    parser.add_argument("--use-gemini", action="store_true",
                        help="Use Gemini vision API for descriptions (requires GEMINI_API_KEY)")
    args = parser.parse_args()

    if not args.assets_dir.is_dir():
        sys.exit(f"ERROR: {args.assets_dir} is not a directory")

    ingest(args.assets_dir, skip_existing=not args.no_skip, use_gemini=args.use_gemini)
