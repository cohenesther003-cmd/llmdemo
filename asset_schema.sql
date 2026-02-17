-- Asset Search Schema
-- Run this once to set up the assets table in your pgvector database.
-- Usage: psql -h $PGVECTOR_HOST -p $PGVECTOR_PORT -U $PGVECTOR_USER -d $PGVECTOR_DB -f asset_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS assets (
    id              SERIAL PRIMARY KEY,
    filename        TEXT NOT NULL,             -- e.g. closet_1.png
    filepath        TEXT NOT NULL,             -- absolute local path
    group_name      TEXT NOT NULL,             -- e.g. "closet" (base name without _N suffix)
    is_part         BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE if this is one piece of a multi-part asset
    part_number     INT,                       -- NULL for standalone assets, 1/2/3... for parts
    description     TEXT,                      -- AI-generated natural language description
    tags            TEXT[],                    -- extracted keyword tags e.g. {"furniture","wood","brown"}
    embedding       vector(384),               -- sentence-transformers all-MiniLM-L6-v2 dims
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast cosine-similarity search
CREATE INDEX IF NOT EXISTS assets_embedding_idx
    ON assets
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for group lookups (e.g. "show all parts of closet")
CREATE INDEX IF NOT EXISTS assets_group_idx ON assets (group_name);
