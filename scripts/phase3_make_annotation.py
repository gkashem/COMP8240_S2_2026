"""Phase 3 annotation setup (proposal §5.2): blind + randomised. Documents shuffled; the 5 summaries of each
document shuffled within it (source shown once per document group, as in SummEval's interface).
Outputs: annotation/sheet.csv (what the annotator sees: item, doc, source, summary, 4 empty 1-5 columns)
         annotation/key.csv   (item -> doc id + system; not opened while annotating). Seed 8240."""
import csv, json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(8240)
docs = {d['id']: d for d in map(json.loads, open(os.path.join(ROOT, 'data/processed/phase3_docs.jsonl')))}
by_doc = {}
for r in map(json.loads, open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))):
    by_doc.setdefault(r['id'], []).append(r)
order = list(by_doc); rng.shuffle(order)
DIMS = ['coherence', 'consistency', 'fluency', 'relevance']
n = 0
with open(os.path.join(ROOT, 'annotation/sheet.csv'), 'w', newline='') as fs, open(os.path.join(ROOT, 'annotation/key.csv'), 'w', newline='') as fk:
    ws, wk = csv.writer(fs), csv.writer(fk)
    ws.writerow(['item', 'doc', 'source', 'summary'] + DIMS); wk.writerow(['item', 'doc_id', 'model_id', 'source_type'])
    for di, did in enumerate(order, 1):
        items = by_doc[did][:]; rng.shuffle(items)
        for j, r in enumerate(items):
            n += 1
            ws.writerow([n, di, docs[did]['source'] if j == 0 else '(same as above)', r['decoded']] + [''] * 4)
            wk.writerow([n, did, r['model_id'], r['source_type']])
print('items: %d | documents: %d' % (n, len(order)))
