# Asset Search Interface — Manager Briefing

---

## The 30-Second Pitch

**Problem**: Game studios have thousands of sprite assets. Finding the right one means remembering filenames or browsing folders. It wastes artists' and developers' time.

**Solution**: A search engine where you type `"blue defensive item"` and it shows you the shield — even if the file is named `shield_2.png`. It understands **meaning**, not just filenames.

---

## How It Works (Simply)

1. A one-time script reads every image, generates a text description, and stores it in a local database
2. When you search, your query is matched against those descriptions mathematically
3. Results are grouped (so `sword_1` and `sword_2` appear together) and ranked by relevance

---

## What We Built

| Piece | What it is | Status |
|-------|-----------|--------|
| Ingestion pipeline | Script that processes sprites into the database | ✅ Done |
| Search API | Backend server with 4 endpoints | ✅ Done |
| Search UI | Browser-based search interface | ✅ Done |
| AI descriptions | Gemini reads actual images | ⚠️ Built — needs network access to Google API |

---

## Key Technical Decisions

| Decision | What we chose | Why |
|----------|--------------|-----|
| Vector database | ChromaDB (local file) | No cloud account, no cost, runs anywhere |
| AI vision | Google Gemini Flash | Free tier, fast, multimodal |
| Backend | FastAPI (Python) | Industry standard, auto-generates API docs |
| Frontend | Plain HTML/JS | No build step, no framework dependency |
| Deployment | No Docker needed | Runs with two terminal commands |

---

## Current Gap: Demo vs. Production

Right now, descriptions are written from filenames (hardcoded for 8 test sprites).

In production, **Google's Gemini AI looks at each image** and writes a real description — that's what makes the search genuinely smart. The code is already written; it just needs network access to Google's API.

---

## What It Takes to Go to Production

1. **Unblock network access** to `generativelanguage.googleapis.com`
2. **Re-ingest with real Gemini descriptions** — run `python asset_ingest.py --use-gemini` once
3. **Point it at real sprites** — change `ASSETS_DIR` to the actual sprite folder
4. **Host the API** — deploy to any server; the database is a folder you copy with it

> No new code needed. The foundation is complete.

---

## API Endpoints (for technical stakeholders)

| Endpoint | What it does |
|----------|-------------|
| `GET /search?q=...` | Search for assets by description |
| `GET /assets` | List all ingested assets |
| `GET /asset/{id}` | Get details for a single asset |
| `POST /ingest` | Trigger re-ingestion of assets |

---

*Last updated: 2026-02-24*
