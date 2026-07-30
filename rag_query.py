import requests
import psycopg2
import os
import sys
import re
import json

CANDIDATE_POOL_SIZE = 15
FINAL_RESULT_CAP = 5

RERANK_PROMPT = """You are judging search results for relevance to a question.

Question: {question}

Candidate passages:
{candidates}

First, assess overall retrieval quality: does at least one passage genuinely and confidently answer the question -- not just superficially similar in wording, but actually relevant? Label it STRONG (at least one passage clearly and directly answers the question), WEAK (some related content exists but nothing confidently answers it), or NONE (nothing relevant at all).

Second, check whether the candidate passages disagree with each other on any concrete fact relevant to the question (e.g. different prices, different dates, contradictory instructions) -- a genuine factual conflict, not just different wording or different levels of detail. If they conflict, briefly describe it and name which candidate numbers disagree. If not, write NO.

Then list the candidate numbers that are genuinely relevant, ordered from most to least relevant (best first), at most 5, comma-separated. If none are relevant, write NONE.

Respond in EXACTLY this format, nothing else:
QUALITY: <STRONG|WEAK|NONE>
CONFLICT: <NO, or a brief description naming which candidates disagree>
RELEVANT: <comma-separated numbers or NONE>"""

def rerank(question, candidates, model="gpt-5-nano"):
    # Lightweight groundedness/quality check (CRAG-style grading) plus a
    # conflict check, both combined into the same call as reranking to avoid
    # extra AI round-trips. The quality and conflict labels are explicit
    # signals Hermes can act on directly, instead of having to notice a
    # factual disagreement between sources on its own.
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
        return "WEAK", "NO", candidates[:FINAL_RESULT_CAP]

    quality = "WEAK"  # cautious default if parsing fails
    conflict = "NO"
    relevant_line = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("QUALITY:"):
            val = stripped.split(":", 1)[1].strip().upper()
            if val in ("STRONG", "WEAK", "NONE"):
                quality = val
        elif stripped.upper().startswith("CONFLICT:"):
            val = stripped.split(":", 1)[1].strip()
            if val and val.upper() != "NO":
                conflict = val
        elif stripped.upper().startswith("RELEVANT:"):
            relevant_line = stripped.split(":", 1)[1].strip()

    if not relevant_line or relevant_line.upper() == "NONE":
        return quality, conflict, []
    indices = []
    for part in re.findall(r'\d+', relevant_line):
        idx = int(part) - 1
        if 0 <= idx < len(candidates) and idx not in indices:
            indices.append(idx)
    if not indices:
        print("[reranker: unparseable response, falling back to plain similarity order]", file=sys.stderr)
        return quality, conflict, candidates[:FINAL_RESULT_CAP]
    return quality, conflict, [candidates[i] for i in indices[:FINAL_RESULT_CAP]]

if len(sys.argv) < 2:
    print('Usage: python3 rag_query.py "question" [source_filter]')
    sys.exit(1)
question = sys.argv[1]
source_filter = sys.argv[2] if len(sys.argv) > 2 else None
r = requests.post('http://litellm:4000/v1/embeddings', json={'model': 'embed-small', 'input': question})
question_vector = r.json()['data'][0]['embedding']
conn = psycopg2.connect(host='postgres', dbname='litellm', user='litellm', password=os.environ.get('POSTGRES_PASSWORD'))
cur = conn.cursor()

# Semantic cache: skip the full pipeline (vector search, keyword search, and
# the rerank/quality AI call) if a near-duplicate question was already
# answered recently. Threshold and TTL calibrated empirically -- paraphrases
# of the same question measured ~0.18 cosine distance apart, genuinely
# different questions ~0.83, so 0.25 leaves comfortable margin on both sides.
# The TTL keeps cached answers from going stale if new data gets ingested.
CACHE_SIMILARITY_THRESHOLD = 0.25
CACHE_TTL_MINUTES = 60
if source_filter:
    cur.execute(
        "SELECT question, quality, results, conflict, question_embedding <=> %s::vector AS distance FROM rag_query_cache "
        "WHERE source_filter = %s AND created_at > now() - make_interval(mins => %s) "
        "ORDER BY distance LIMIT 1",
        (question_vector, source_filter, CACHE_TTL_MINUTES)
    )
else:
    cur.execute(
        "SELECT question, quality, results, conflict, question_embedding <=> %s::vector AS distance FROM rag_query_cache "
        "WHERE source_filter IS NULL AND created_at > now() - make_interval(mins => %s) "
        "ORDER BY distance LIMIT 1",
        (question_vector, CACHE_TTL_MINUTES)
    )
cache_row = cur.fetchone()
if cache_row and cache_row[4] is not None and cache_row[4] <= CACHE_SIMILARITY_THRESHOLD:
    cached_question, cached_quality, cached_results, cached_conflict, cache_distance = cache_row
    print(f'Question: {question}')
    print(f'[retrieval quality: {cached_quality}]')
    if cached_conflict and cached_conflict.upper() != "NO":
        print(f'[conflict detected: {cached_conflict}]')
    print(f'[cache hit: matched "{cached_question}", distance {cache_distance:.4f}]')
    print('---')
    for content, source, distance in cached_results:
        print(f'[source: {source}] [distance: {distance:.4f}]')
        print(content)
        print('---')
    conn.close()
    sys.exit(0)

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

quality, conflict, results = rerank(question, combined) if combined else ("NONE", "NO", [])

print(f'Question: {question}')
print(f'[retrieval quality: {quality}]')
if conflict and conflict.upper() != "NO":
    print(f'[conflict detected: {conflict}]')
print('---')
for content, source, distance in results:
    print(f'[source: {source}] [distance: {distance:.4f}]')
    print(content)
    print('---')

results_json = json.dumps([[c, s, d] for c, s, d in results])
cur.execute(
    'INSERT INTO rag_query_cache (question, question_embedding, source_filter, quality, results, conflict) VALUES (%s, %s, %s, %s, %s::jsonb, %s)',
    (question, question_vector, source_filter, quality, results_json, conflict)
)
# Light housekeeping: drop cache rows old enough that no query could still
# find them via the TTL filter above, so the table doesn't grow forever.
cur.execute("DELETE FROM rag_query_cache WHERE created_at < now() - interval '24 hours'")
conn.commit()
