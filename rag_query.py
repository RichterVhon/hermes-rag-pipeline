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

Which of these passages are actually useful for answering the question above -- not just superficially similar in wording, but genuinely relevant? Respond with ONLY a comma-separated list of the candidate numbers that are relevant, ordered from most to least relevant (best first). Include at most 5 numbers. If none are relevant, respond with: NONE

Answer:"""

def rerank(question, candidates, model="gpt-5-nano"):
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
        return candidates[:FINAL_RESULT_CAP]
    if not raw or raw.upper() == "NONE":
        return []
    indices = []
    for part in re.findall(r'\d+', raw):
        idx = int(part) - 1
        if 0 <= idx < len(candidates) and idx not in indices:
            indices.append(idx)
    if not indices:
        print("[reranker: unparseable response, falling back to plain similarity order]", file=sys.stderr)
        return candidates[:FINAL_RESULT_CAP]
    return [candidates[i] for i in indices[:FINAL_RESULT_CAP]]

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
results = rerank(question, candidates) if candidates else []

print(f'Question: {question}')
print('---')
for content, source, distance in results:
    print(f'[source: {source}] [distance: {distance:.4f}]')
    print(content)
    print('---')
