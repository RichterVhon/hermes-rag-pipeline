import requests
import psycopg2
import os
import sys

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
        'SELECT content, source, embedding <-> %s::vector AS distance FROM rag_chunks WHERE source ILIKE %s ORDER BY distance LIMIT 5',
        (question_vector, f'%{source_filter}%')
    )
else:
    cur.execute(
        'SELECT content, source, embedding <-> %s::vector AS distance FROM rag_chunks ORDER BY distance LIMIT 5',
        (question_vector,)
    )

print(f'Question: {question}')
print('---')
for content, source, distance in cur.fetchall():
    print(f'[source: {source}] [distance: {distance:.4f}]')
    print(content)
    print('---')
