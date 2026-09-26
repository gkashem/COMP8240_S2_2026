"""Phase 3 systems (abstractive): facebook/bart-large-cnn (larger) and sshleifer/distilbart-cnn-6-6 (smaller),
model.generate() with each model's own generation_config defaults, truncation=True, CPU. Model choice is a project decision
(proposal: 'two pretrained abstractive models with different capacities'). Adds rows to phase3_summaries.jsonl."""
import json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl')
docs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_docs.jsonl'))]
MODELS = {'bart_large_cnn': 'facebook/bart-large-cnn', 'distilbart_cnn_6_6': 'sshleifer/distilbart-cnn-6-6'}
rows = [r for r in map(json.loads, open(P)) if r['model_id'] not in MODELS]
for name, hf in MODELS.items():
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(hf); model = AutoModelForSeq2SeqLM.from_pretrained(hf).eval()
    for d in docs:
        x = tok(d['source'], truncation=True, max_length=1024, return_tensors='pt')
        with torch.no_grad(): ids = model.generate(**x)
        out = tok.decode(ids[0], skip_special_tokens=True).strip()
        rows.append({'id': d['id'], 'model_id': name, 'source_type': d['source_type'], 'decoded': out, 'references': [d['reference']]})
    print('%s: %d docs in %.0f s | example: %s' % (name, len(docs), time.time() - t0, rows[-1]['decoded'][:150]), flush=True)
with open(P, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('total summaries:', len(rows))
