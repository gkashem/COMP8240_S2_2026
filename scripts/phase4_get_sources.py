"""Phase 4: attach CNN/DM source articles to SummEval's 100 article IDs."""
import json
import pandas as pd

df = pd.read_parquet('data/raw/cnndm/3.0.0/test-00000-of-00001.parquet')
ids = {json.loads(l)['id'] for l in open('data/raw/model_annotations.aligned.jsonl')}
h = {i.split('-test-')[1]: i for i in ids}
m = df[df['id'].isin(h)]
assert len(m) == len(ids) == 100
with open('data/processed/summeval_sources.jsonl', 'w') as f:
    for _, r in m.sort_values('id').iterrows():
        f.write(json.dumps({'id': h[r['id']], 'text': r['article']}) + '\n')
print('written:', len(m))
