"""Phase 3 system 5 (deliberately weaker configuration, proposal §5.2): sshleifer/distilbart-cnn-6-6 with
degraded decoding: greedy (num_beams=1), max_new_tokens=30, min_length=0, no_repeat_ngram_size=0.
Settings are a project choice, intended to lower fluency/consistency. Adds rows to phase3_summaries.jsonl."""
import json, os, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl')
NAME, HF = 'distilbart_weak', 'sshleifer/distilbart-cnn-6-6'
docs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_docs.jsonl'))]
rows = [r for r in map(json.loads, open(P)) if r['model_id'] != NAME]
t0 = time.time(); tok = AutoTokenizer.from_pretrained(HF); model = AutoModelForSeq2SeqLM.from_pretrained(HF).eval()
for d in docs:
    x = tok(d['source'], truncation=True, max_length=1024, return_tensors='pt')
    with torch.no_grad():
        ids = model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=30, min_length=0, no_repeat_ngram_size=0)
    rows.append({'id': d['id'], 'model_id': NAME, 'source_type': d['source_type'],
                 'decoded': tok.decode(ids[0], skip_special_tokens=True).strip(), 'references': [d['reference']]})
with open(P, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('%s: %d docs in %.0f s | total summaries: %d' % (NAME, len(docs), time.time() - t0, len(rows)))
for r in rows[-3:]: print('-', r['decoded'])
