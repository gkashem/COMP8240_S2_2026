"""Phase 3 systems (extractive): LEAD-3 (first 3 sentences) and TextRank (3 sentences, original order).
TextRank: sentence graph, edge weight = word overlap / (log|s1| + log|s2|) (Mihalcea & Tarau, 2004),
PageRank damping 0.85, 50 power iterations. Appends to data/processed/phase3_summaries.jsonl."""
import json, math, os, re
from nltk.tokenize import sent_tokenize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_docs.jsonl'))]
words = lambda s: set(re.findall(r'[a-z0-9]+', s.lower()))
def textrank(sents, k=3, d=0.85, it=50):
    W = [words(s) for s in sents]; n = len(sents)
    sim = [[0.0 if i == j or len(W[i]) < 2 or len(W[j]) < 2 else len(W[i] & W[j]) / (math.log(len(W[i])) + math.log(len(W[j])))
            for j in range(n)] for i in range(n)]
    out = [sum(r) or 1.0 for r in sim]; sc = [1.0] * n
    for _ in range(it):
        sc = [(1 - d) + d * sum(sim[j][i] / out[j] * sc[j] for j in range(n)) for i in range(n)]
    top = sorted(sorted(range(n), key=lambda i: -sc[i])[:k])
    return ' '.join(sents[i] for i in top)
rows = []
for doc in docs:
    s = sent_tokenize(doc['source'])
    for name, summ in [('lead3', ' '.join(s[:3])), ('textrank', textrank(s))]:
        rows.append({'id': doc['id'], 'model_id': name, 'source_type': doc['source_type'],
                     'decoded': summ, 'references': [doc['reference']]})
with open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'), 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('summaries:', len(rows)); print('example lead3:', rows[0]['decoded'][:200]); print('example textrank:', rows[1]['decoded'][:200])
