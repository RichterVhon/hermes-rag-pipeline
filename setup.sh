#!/usr/bin/env bash
# ============================================================================
# RAG Knowledge-Base System -- automated setup script
#
# Stands up the full stack (Postgres+pgvector, LiteLLM, Hermes) on a fresh
# Ubuntu VPS, creates the database schema the RAG pipeline needs, and installs
# the pipeline scripts into the running Hermes profile.
#
# Safe to re-run: every step checks before it acts (idempotent).
#
# Usage:
#   ./setup.sh                    # interactive, prompts for profile name
#   PROFILE=researcher ./setup.sh # non-interactive
# ============================================================================
set -euo pipefail

STACK_DIR="$HOME/hermes-stack"
PROFILE="${PROFILE:-}"

echo "=== 1/7: Checking for Docker ==="
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group membership to apply."
else
    echo "Docker already installed: $(docker --version)"
fi

echo ""
echo "=== 2/7: Setting up directory structure ==="
mkdir -p "$STACK_DIR"/data/postgres
mkdir -p "$STACK_DIR"/data/hermes
mkdir -p "$STACK_DIR"/litellm
mkdir -p "$STACK_DIR"/init-scripts
echo "Directories ready under $STACK_DIR"

echo ""
echo "=== 3/7: Checking for docker-compose.yml, litellm config, and init scripts ==="
for f in docker-compose.yml litellm/config.yaml init-scripts/03-claude-code-symlinks init-scripts/04-rag-deps; do
    if [ ! -f "$STACK_DIR/$f" ]; then
        echo "MISSING: $STACK_DIR/$f"
        echo "Copy this file from the hermes-rag-pipeline repo's hermes-stack/ folder before continuing."
        MISSING_FILES=1
    fi
done
if [ "${MISSING_FILES:-0}" = "1" ]; then
    echo ""
    echo "One or more required config files are missing. Copy them from the repo, then re-run this script."
    exit 1
fi
echo "All required config files present."

echo ""
echo "=== 4/7: Checking for .env ==="
if [ ! -f "$STACK_DIR/.env" ]; then
    echo ".env not found."
    if [ -f "$STACK_DIR/.env.example" ]; then
        cp "$STACK_DIR/.env.example" "$STACK_DIR/.env"
        echo "Created $STACK_DIR/.env from .env.example -- EDIT IT NOW with real values"
        echo "(at minimum: POSTGRES_PASSWORD and OPENAI_API_KEY), then re-run this script."
        exit 1
    else
        echo "No .env.example found either. Create $STACK_DIR/.env manually (see the setup guide"
        echo "for the required variables), then re-run this script."
        exit 1
    fi
fi
if ! grep -q "^POSTGRES_PASSWORD=.\+" "$STACK_DIR/.env" || ! grep -q "^OPENAI_API_KEY=.\+" "$STACK_DIR/.env"; then
    echo "$STACK_DIR/.env exists but POSTGRES_PASSWORD or OPENAI_API_KEY looks empty."
    echo "Fill those in, then re-run this script."
    exit 1
fi
echo ".env present with required values filled in."

echo ""
echo "=== 5/7: Starting the stack ==="
cd "$STACK_DIR"
docker compose up -d
echo "Waiting for Postgres to become healthy..."
for i in $(seq 1 30); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' postgres 2>/dev/null || echo "starting")
    if [ "$STATUS" = "healthy" ]; then
        echo "Postgres is healthy."
        break
    fi
    sleep 2
done
if [ "$STATUS" != "healthy" ]; then
    echo "Postgres did not become healthy in time -- check 'docker compose logs postgres'."
    exit 1
fi

echo ""
echo "=== 6/7: Creating database schema ==="
docker exec postgres psql -U litellm -d litellm -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_content_tsv
    ON rag_chunks USING GIN (to_tsvector('english', content));

CREATE TABLE IF NOT EXISTS rag_query_cache (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    question_embedding vector(1536) NOT NULL,
    source_filter TEXT,
    quality TEXT NOT NULL,
    results JSONB NOT NULL,
    conflict TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
SQL
echo "Schema ready (rag_chunks, rag_query_cache, and indexes)."

echo ""
echo "=== 7/7: Installing pipeline scripts into the Hermes profile ==="
if [ -z "$PROFILE" ]; then
    read -rp "Profile name to install into [researcher]: " PROFILE
    PROFILE="${PROFILE:-researcher}"
fi
PROFILE_DIR="$STACK_DIR/data/hermes/profiles/$PROFILE"
mkdir -p "$PROFILE_DIR"
for f in rag_ingest.py rag_query.py run_regression_tests.py regression_tests.json SOUL.md; do
    if [ -f "$STACK_DIR/../$f" ]; then
        cp "$STACK_DIR/../$f" "$PROFILE_DIR/$f"
        echo "Installed $f into $PROFILE_DIR"
    else
        echo "NOTE: $f not found next to this script -- copy it into $PROFILE_DIR manually."
    fi
done

echo ""
echo "=== Done ==="
echo "Stack is up. Check status with: docker compose -f $STACK_DIR/docker-compose.yml ps"
echo "Pipeline scripts are installed under: $PROFILE_DIR"
echo "See the setup guide for how to ingest content and start asking questions."
