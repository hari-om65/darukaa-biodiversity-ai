# Darukaa Biodiversity AI

Placeholder README.

## Project layout

- `app/` — FastAPI application, routers, and Pydantic schemas
- `knowledge/` — ingestion scripts, source documents, and Chroma vector store
- `reasoning/` — causal graph reasoning engine
- `tests/` — test suite
- `frontend/` — Streamlit app

## Getting started

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Running tests

```bash
pytest
```
