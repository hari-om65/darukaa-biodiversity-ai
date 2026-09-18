# Darukaa Biodiversity AI

An API that turns a handful of site variables (soil organic carbon, rainfall,
crop/land-use, region) into evidence-grounded biodiversity recommendations. A
small causal graph identifies which downstream effects are plausible, a
ChromaDB knowledge base supplies the evidence for each effect, and Llama (via Groq)
turns the two into specific, cited recommendations — never generic advice,
never an invented statistic.

## Contents

- [Architecture overview](#architecture-overview)
- [Database & schema](#database--schema)
- [Local setup](#local-setup)
- [How the pipeline works end to end](#how-the-pipeline-works-end-to-end)
- [API reference](#api-reference)
- [CI/CD](#cicd)
- [Deployment](#deployment)
- [Project layout](#project-layout)

## Architecture overview

Three independent pieces feed a single composition step:

```
reasoning/          knowledge/                    app/
┌──────────────┐    ┌─────────────────────┐       ┌───────────────────────┐
│ graph.yaml    │    │ sources/*.md        │       │ routers/               │
│ thresholds.yaml│   │   → ingest.py       │       │  health / chat / whatif│
│   → engine.py │    │   → ChromaDB        │       │ composer.py            │
│  analyze()    │    │  retrieve()         │       │  gather evidence per   │
└──────┬────────┘    └──────────┬──────────┘       │  chain, call Groq,     │
       │                        │                   │  validate with pydantic│
       └───────────┬────────────┘                   └───────────┬───────────┘
                    ▼                                            │
          causal chains + evidence ─────────────────────────────►│
                                                                  ▼
                                                     RecommendationsResponse
                                                     + natural-language reply
                                                     + explain payload
```

- **`reasoning/`** — a small causal graph (soil → microbial diversity → water
  retention → species survival, etc.) plus a set of threshold rules
  (`soil_organic_carbon < 0.5`, `rainfall == "low"`, `crop == "monoculture"`).
  `analyze(inputs)` evaluates the thresholds and walks up to two hops
  downstream through the graph from each one that fires.
- **`knowledge/`** — short reference documents, chunked and embedded with
  `sentence-transformers/all-MiniLM-L6-v2`, stored in a persistent ChromaDB
  collection. `retrieve(query, tags=...)` does similarity search scoped to
  topic tags.
- **`app/composer.py`** — for every causal chain `analyze()` finds, retrieves
  evidence scoped to that chain's topics, builds one prompt containing all of
  it, and asks Groq (Llama) for structured JSON recommendations. Every recommendation
  must cite evidence actually retrieved for this request; the response is
  validated with Pydantic and retried once if it fails.
- **`app/routers/`** — `/chat` and `/chat/structured` run the full pipeline
  and return recommendations plus a natural-language summary; `/whatif` runs
  just the reasoning half, letting you simulate a variable change without
  calling Groq.
- **`frontend/`** — a minimal Streamlit page (currently just a health check;
  not the primary interface).

## Database & schema

### ChromaDB — collection `biodiversity_knowledge`

Persisted on disk at `knowledge/chroma_db/` (override with `CHROMA_DB_DIR`,
see [`.env.example`](.env.example)). One entry per ~500-word chunk of a source
document in `knowledge/sources/`:

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | `"<source-filename-stem>-<chunk-index>"` |
| `embedding` | `float[384]` | from `all-MiniLM-L6-v2` |
| `document` | `str` | the chunk's raw text |
| `metadata.source` | `str` | source filename, e.g. `soil_organic_carbon.md` |
| `metadata.year` | `int` | from the source file's frontmatter |
| `metadata.topic_tags` | `str` | comma-joined, e.g. `"soil,biodiversity"` |
| `metadata.tag_<tag>` | `bool` | one boolean per allowed tag (Chroma metadata values must be scalars, not lists), for `soil`, `water`, `land_use`, `biodiversity`, `climate`, `human_impact` |

A source file declares its metadata in a YAML frontmatter block:

```markdown
---
title: Soil Organic Carbon and Ecosystem Health
year: 2020
topic_tags: [soil, biodiversity]
---

Body text, chunked at ingest time...
```

Run `python -m knowledge.ingestion.ingest` to (re-)ingest everything under
`knowledge/sources/` — it's idempotent (`collection.upsert` keyed by id).

### `reasoning/graph.yaml` — the causal graph

A flat list of directed edges, loaded into a `networkx.DiGraph`:

```yaml
edges:
  - from: soil_organic_carbon
    to: microbial_diversity
    relation: positive   # positive | negative
    strength: high        # low | medium | high
```

`relation` and `strength` compose across a multi-hop chain (relation by sign
multiplication, strength by taking the chain's weakest link) — see
`reasoning/engine.py::_compose_relation` / `_compose_strength`.

### `reasoning/thresholds.yaml` — trigger rules

```yaml
thresholds:
  - variable: soil_organic_carbon   # key to look up in the inputs dict
    operator: "<"                    # < <= > >= == !=
    value: 0.5
    flag: critical_low               # human-readable label
    node: soil_organic_carbon        # graph node to start walking from
```

`node` lets a threshold on one variable point at a differently-named graph
node — e.g. the `crop` threshold triggers on `crop == "monoculture"`, but the
graph node it walks from is `monoculture` (the value), not `crop`.

## Local setup

Requires Python 3.11+.

```bash
# 1. Clone and enter the project
cd darukaa-biodiversity-ai

# 2. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
#   then edit .env and set GROQ_API_KEY

# 5. Populate the knowledge base (idempotent; re-run any time sources change)
python -m knowledge.ingestion.ingest

# 6. Run the API
uvicorn app.main:app --reload
# → http://localhost:8000/docs for interactive OpenAPI docs
# → http://localhost:8000/health should return {"status": "ok"}

# 7. (optional) Run the Streamlit frontend against it
streamlit run frontend/streamlit_app.py
```

Run the test suite with:

```bash
pytest
```

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging
Face (a one-time ~90MB download, cached under `~/.cache/huggingface`
afterward). Tests that call the Groq API for real
(`tests/test_composer.py::test_compose_recommendations_live_semi_arid_wheat`)
are skipped automatically unless `GROQ_API_KEY` is set — everything else
exercises the composer/chat/whatif logic against a fake Groq client, so
the rest of the suite runs free and offline.

Lint with:

```bash
ruff check .
```

## How the pipeline works end to end

Using `/chat` as the walkthrough (`/chat/structured` and `/whatif` are
variations on the same core — see [API reference](#api-reference)):

1. **Request in.** `POST /chat` with `{session_id, message, structured_input?}`.
   Session state (conversation history + known variables) lives in an
   in-memory dict (`app/chat_session.py`) keyed by `session_id`.
2. **Slot-filling.** `app/chat_extraction.py` pulls whichever of
   `soil_organic_carbon`, `rainfall`, `crop`, `region` it can find in the raw
   message text via a small rule-based extractor (deterministic and free —
   no LLM call needed just to fill slots). Anything in `structured_input` is
   merged in directly. Everything found is merged into the session's running
   `variables` dict, so a multi-turn conversation accumulates state.
3. **Clarify if incomplete.** If any of the four required variables are still
   missing, the response comes back with `status: "needs_clarification"` and
   a question covering *only* the missing ones — already-known variables are
   never re-asked about.
4. **Reasoning.** Once all four are known, `reasoning.engine.analyze(variables)`
   evaluates `thresholds.yaml` against them. Each threshold that fires (e.g.
   `soil_organic_carbon < 0.5`) starts a walk from its graph node, up to two
   hops downstream through `graph.yaml`, producing a list of causal chains:
   `{path, affected_metric, relation, strength}`.
5. **Retrieval.** For each chain, `app/composer.py::gather_chain_evidence()`
   retrieves up to 3 evidence chunks from ChromaDB, scoped to the topic tags
   relevant to that chain's nodes (e.g. a chain rooted at `soil_organic_carbon`
   is scoped to the `soil` tag). If a tag-scoped search comes back empty, it
   falls back to an unscoped search rather than sending the model zero evidence.
6. **Composition.** `compose_recommendations_with_evidence()` builds one
   prompt — the site inputs, every chain, and its evidence — and calls the
   Groq API (`client.chat.completions.create`, model
   `llama-3.3-70b-versatile` by default, see `GROQ_MODEL` in `.env.example`)
   with a strict `json_schema` response format built from
   `RecommendationsResponse`'s Pydantic schema, forcing structured JSON that
   Pydantic then validates. The system prompt forbids generic advice and
   invented statistics, and requires every recommendation's mechanism and
   cited sources to trace back to the retrieved evidence.
7. **Grounding check + retry.** A post-hoc check rejects any recommendation
   that cites a source it wasn't actually given as evidence (catches citation
   hallucination that schema validation alone can't). On any parse, Pydantic,
   or grounding failure, the call is retried once with the failure reason
   appended to the prompt; a second failure raises.
8. **Response out.** The API returns the structured `RecommendationsResponse`,
   a natural-language summary built from it, and an `explain` payload: which
   thresholds fired, which graph paths were walked, the exact evidence chunks
   retrieved per chain (source + similarity score), and which chain/chunk
   backs each recommendation's cited sources.

`/whatif` runs only steps 4 and a slice of 8: given a proposed change to one
variable, it re-runs `analyze()` before and after (without touching the
session's stored variables), returns the diff (`removed_chains` /
`added_chains`), and annotates every chain with a rough
short/medium/long-term projection derived from the chain's composed strength
and hop count (`reasoning.engine.project_time_horizon` — documented as a
heuristic proxy, not a calibrated forecast, in its own docstring).

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check → `{"status": "ok"}` |
| `POST /chat` | Conversational entry point; clarifies missing variables, then runs the full pipeline |
| `POST /chat/structured` | Same pipeline, but takes a complete `{session_id, inputs}` payload and skips clarification (422 if incomplete) |
| `POST /whatif` | Simulates one variable change against a session's known variables; reasoning only, no Groq call |

See `/docs` (Swagger UI) on a running instance for full request/response
schemas, or the Pydantic models under `app/schemas/`.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request to `main`, as
two parallel jobs:

- **`test`** — installs `requirements.txt`, then runs `pytest`. The
  `all-MiniLM-L6-v2` model download is cached across runs
  (`~/.cache/huggingface`, keyed by a fixed cache key since the model doesn't
  change). If the `GROQ_API_KEY` repository secret is configured, it's
  passed through as an env var and the one live Groq test runs for real;
  otherwise that single test is skipped and everything else still runs.
- **`lint`** — installs `requirements.txt`, then runs `ruff check .`. Config
  lives in `pyproject.toml` (line length 120, target Python 3.11).

There is no automated deploy step (CD) yet — deploys below are manual/
platform-triggered. See [Deployment](#deployment).

## Deployment

The backend is a single Docker image ([`Dockerfile`](Dockerfile)): install
dependencies, copy the app in, then **run
`python -m knowledge.ingestion.ingest` at build time** so the ChromaDB
knowledge base is baked into the image — the container starts ready to serve
recommendations, with no separate init step at runtime. Because ingestion
re-runs on every build, editing `knowledge/sources/` and redeploying is all
it takes to update the knowledge base.

> This bakes the vector store into an image layer, which is fine for a
> knowledge base this size. If it grows past what's comfortable to rebuild
> into an image on every deploy, attach a persistent disk at the path set by
> `CHROMA_DB_DIR` instead, and run ingestion as a one-off/manual step rather
> than in the Dockerfile.

### Backend on Render

This repo includes a [`render.yaml`](render.yaml) Blueprint that provisions
a Docker-based web service from the `Dockerfile`.

1. Push the repo to GitHub (or GitLab).
2. In the Render dashboard: **New +** → **Blueprint**, and point it at the repo.
   Render reads `render.yaml` and provisions the service.
3. On the service's **Environment** tab, set `GROQ_API_KEY` (required)
   and, if you want a non-default store location, `CHROMA_DB_DIR` — both are
   declared in `render.yaml` with `sync: false`, so Render prompts for them
   rather than expecting a value in the file. `GROQ_MODEL` already
   defaults to `llama-3.3-70b-versatile` there; override it the same way if needed.
4. Deploy. Render builds the `Dockerfile` (ingestion runs as part of the
   build), starts the container, and binds it to the `$PORT` it injects — the
   Dockerfile's `CMD` already reads that.
5. Confirm with `curl https://<your-service>.onrender.com/health`.

### Backend on Railway

Railway auto-detects the `Dockerfile` at the repo root — no extra config
file needed.

1. Push the repo to GitHub.
2. In Railway: **New Project** → **Deploy from GitHub repo**, select this repo.
3. Under **Variables**, set `GROQ_API_KEY` (required), and optionally
   `GROQ_MODEL` / `CHROMA_DB_DIR`.
4. Railway builds the `Dockerfile` (same build-time ingestion as Render) and
   deploys, injecting its own `$PORT`, which the Dockerfile's `CMD` respects.
5. Confirm with `curl https://<your-app>.up.railway.app/health`.

### Frontend on Streamlit Community Cloud

`frontend/streamlit_app.py` reads the backend URL from `st.secrets["API_URL"]`
(falling back to the `API_URL` env var, then to `http://localhost:8000` for
local dev) — set it to point at whichever backend you deployed above.

1. Push the repo to GitHub (the same repo as the backend, or a fork of it).
2. In [share.streamlit.io](https://share.streamlit.io): **New app**, pick the
   repo/branch, and set the main file path to `frontend/streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   API_URL = "https://<your-backend-url>"
   ```
4. Deploy. Streamlit Community Cloud installs `requirements.txt` (shared with
   the backend — it already includes `streamlit`) and runs the app.

## Project layout

```
app/
  routers/            health.py, chat.py, whatif.py
  schemas/             pydantic request/response models
  composer.py           reasoning + retrieval + Groq (Llama) → recommendations
  chat_session.py       in-memory per-session state
  chat_extraction.py    rule-based slot-filling for /chat
  explain_builder.py    builds the explain payload shared by /chat & /whatif
  dependencies.py       FastAPI-injected Groq client
  main.py                FastAPI app + router registration
knowledge/
  sources/               source documents (.md, with YAML frontmatter)
  ingestion/ingest.py    chunk → embed → store; retrieve()
  chroma_db/              persistent ChromaDB store (gitignored)
reasoning/
  graph.yaml              causal edges
  thresholds.yaml         trigger rules
  engine.py               loads both; analyze(); project_time_horizon()
frontend/
  streamlit_app.py        minimal Streamlit page
tests/                    pytest suite (offline-friendly; see Local setup)
.github/workflows/ci.yml  test + lint on push/PR
Dockerfile                 backend image; runs ingest.py at build time
render.yaml                Render Blueprint for the backend
```
