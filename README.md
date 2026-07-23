# Hermes RAG Pipeline

A knowledge-base (RAG) pipeline built for a Hermes agent profile, running on a small
single-VPS Docker stack (Postgres + pgvector, LiteLLM, Hermes). This repo holds the
pipeline scripts, the agent's instructions, and the infrastructure config needed to
stand the stack back up — not the runtime data itself (no database contents, no
credentials, no chat history).

## What's in here

- `rag_ingest.py` — gets information into the knowledge base. Supports ingesting a
  Wikipedia article, an uploaded file (PDF, Word, PowerPoint, text, or image/scan),
  or rows pulled from a database table. Every mode ends the same way: the content is
  classified by its structural shape (prose, FAQ, legal, transcript, procedural,
  sectioned, structured/tabular, email, or a safe fallback for anything else), split
  accordingly, converted to an embedding, and stored.
- `rag_query.py` — searches the knowledge base for the stored content closest to a
  given question, optionally scoped to a specific source.
- `SOUL.md` — the instructions for the `researcher` Hermes profile, including when
  and how it's expected to use the two scripts above.
- `hermes-stack/` — the Docker Compose stack and supporting config:
  - `docker-compose.yml` — the three services (Postgres, LiteLLM, Hermes).
  - `litellm/config.yaml` — model routing config for LiteLLM (no keys committed;
    every key is referenced via an environment variable).
  - `init-scripts/` — scripts that run every time the Hermes container starts.
    `04-rag-deps` in particular reinstalls the OCR/parsing tools this pipeline
    depends on (tesseract, poppler, pypdf, python-docx, python-pptx, etc.), since a
    plain container restart otherwise wipes anything installed outside the
    persistent data volume.

## Setup

1. Copy `.env.example` to `.env` and fill in real values (API keys, DB password,
   messaging tokens). Never commit the real `.env` file.
2. From `hermes-stack/`, run `docker compose up -d`.
3. Copy `rag_ingest.py`, `rag_query.py`, and `SOUL.md` into the running Hermes
   container's data volume (`./data/hermes` on the host, mounted to `/opt/data` in
   the container) under the appropriate profile.

## Notes

- This pipeline was built iteratively against a genuinely small server (3.8GB RAM,
  2 vCPU, 28GB disk) — a few of the design choices here (lightweight parsing
  libraries instead of heavier layout-aware tools, careful memory limits) are a
  direct result of that constraint, not necessarily the ideal choice on bigger
  hardware.
- Not included: the Postgres data volume (the actual embeddings/knowledge base),
  Hermes session state and chat history, and any real credentials. Back those up
  separately if needed — they don't belong in version control.
