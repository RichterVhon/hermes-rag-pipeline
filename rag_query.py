import requests
import psycopg2
import os
import sys
import re

CANDIDATE_POOL_SIZE = 15
FINAL_RESULT_CAP = 5

RERANK_PROMPT = """You are judging search results for relevance to a question.

Question: {question}

Candidate passages:
{candidates}

First, assess overall retrieval quality: does at least one passage genuinely and confidently answer the question -- not just superficially similar in wording, but actually relevant? Label it STRONG (at least one passage clearly and directly answers the question), WEAK (some related content exists but nothing confidently answers it), or NONE (nothing relevant at all).

Then list the candidate numbers that are genuinely relevant, ordered from most to least relevant (best first), at most 5, comma-separated. If none are relevant, write NONE.

Respond in EXACTLY this format, nothing else:
QUALITY: <STRONG|WEAK|NONE>
RELEVANT: <comma-separated numbers or NONE>"""

def rerank(question, candidates, model="gpt-5-nano"):
    # Lightweight groundedness/quality check (CRAG-style grading), combined
    # into the same call as reranking to avoid a second AI round-trip. The
    # quality label is an explicit signal Hermes can act on directly, instead
    # of having to interpret raw distance numbers on its own.
    numbered = "\n\n".join(f"[{i+1}] (source: {c[1]})\n{c[0][:600]}" for i, c in enumerate(candidates))
    prompt = RERANK_PROMPT.format(question=question, candidates=numbered)
    try:
        r = requests.post("http://litellm:4000/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
        }, timeout=30)
        raw = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[reranker: call failed ({e}), falling back to plain similarity order]", file=sys.stderr)
        return "WEAK", candidates[:FINAL_RESULT_CAP]

    quality = "WEAK"  # cautious default if parsing fails
    relevant_line = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("QUALITY:"):
            val = stripped.split(":", 1)[1].strip().upper()
            if val in ("STRONG", "WEAK", "NONE"):
                quality = val
        elif stripped.upper().startswith("RELEVANT:"):
            relevant_line = stripped.split(":", 1)[1].strip()

    if not relevant_line or relevant_line.upper() == "NONE":
        return quality, []
    indices = []
    for part in re.findall(r'\d+', relevant_line):
        idx = int(part) - 1
        if 0 <= idx < len(candidates) and idx not in indices:
            indices.append(idx)
    if not indices:
        print("[reranker: unparseable response, falling back to plain similarity order]", file=sys.stderr)
        return quality, candidates[:FINAL_RESULT_CAP]
    return quality, [candidates[i] for i in indices[:FINAL_RESULT_CAP]]

if len(sys.argv) < 2:
    print('Usage: python3 rag_query.py "question" [source_filter]')
    sys.exit(1)
question = sys.argv[1]
source_filter = sys.argv[2] if len(sys.argv) > 2 else None
r = requests.post('http://litellm:4000/v1/embeddings', json={'model': 'embed-small', 'input': question})
question_vector = r.json()['data'][0]['embedding']
conn = psycopg2.connect(host='postgres', dbname='litellm', user='litellm', password=os.environ.get('POSTGRES_PASSWORD'))
cur = conn.cursor()
if source_filter:
    cur.execute(
        'SELECT content, source, embedding <-> %s::vector AS distance FROM rag_chunks WHERE source ILIKE %s ORDER BY distance LIMIT %s',
        (question_vector, f'%{source_filter}%', CANDIDATE_POOL_SIZE)
    )
else:
    cur.execute(
        'SELECT content, source, embedding <-> %s::vector AS distance FROM rag_chunks ORDER BY distance LIMIT %s',
        (question_vector, CANDIDATE_POOL_SIZE)
    )
candidates = cur.fetchall()

# Hybrid search: also run a keyword (full-text) search alongside the vector
# search above, so exact terms (names, IDs, dates) that embeddings sometimes
# fuzzy-match aren't missed. Results are merged and deduped before reranking.
KEYWORD_POOL_SIZE = 15
keyword_candidates = []
try:
    if source_filter:
        cur.execute(
            "SELECT content, source, ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank "
            "FROM rag_chunks WHERE source ILIKE %s AND to_tsvector('english', content) @@ plainto_tsquery('english', %s) "
            "ORDER BY rank DESC LIMIT %s",
            (question, f'%{source_filter}%', question, KEYWORD_POOL_SIZE)
        )
    else:
        cur.execute(
            "SELECT content, source, ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank "
            "FROM rag_chunks WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s) "
            "ORDER BY rank DESC LIMIT %s",
            (question, question, KEYWORD_POOL_SIZE)
        )
    # Negate rank so both candidate sources sort the same way (lower = better),
    # matching the vector search's distance convention.
    keyword_candidates = [(content, source, -rank) for content, source, rank in cur.fetchall()]
except Exception as e:
    print(f"[keyword search failed ({e}), continuing with vector results only]", file=sys.stderr)

seen = set()
combined = []
for content, source, score in list(candidates) + keyword_candidates:
    key = (source, content)
    if key not in seen:
        seen.add(key)
        combined.append((content, source, score))

quality, results = rerank(question, combined) if combined else ("NONE", [])

print(f'Question: {question}')
print(f'[retrieval quality: {quality}]')
print('---')
for content, source, distance in results:
    print(f'[source: {source}] [distance: {distance:.4f}]')
    print(content)
    print('---')
