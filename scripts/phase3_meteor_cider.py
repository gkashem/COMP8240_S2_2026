"""Phase 3: METEOR (evaluate_example) + CIDEr (evaluate_batch), authors' toolkit, 1 reference, per summary on the new corpus.
Calls mirror phase2_meteor_cider.py. Empty summaries get 0.0 (excluded from the CIDEr batch).
Writes results/scores/phase3_meteor_cider.jsonl. Not joined to human ratings yet (kept blind)."""
import json, os, time
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.cider_metric import CiderMetric
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))]
ok = [i for i, r in enumerate(recs) if r['decoded'].strip()]
t0 = time.time(); m = MeteorMetric()
met = {i: m.evaluate_example(recs[i]['decoded'], recs[i]['references'])['meteor'] for i in ok}
t1 = time.time()
cid_list = [float(s['cider']) for s in CiderMetric().evaluate_batch(
    [recs[i]['decoded'] for i in ok], [recs[i]['references'] for i in ok], aggregate=False)]
cid = dict(zip(ok, cid_list))
with open(os.path.join(ROOT, 'results/scores/phase3_meteor_cider.jsonl'), 'w') as f:
    for i, r in enumerate(recs):
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'],
                            'meteor': met.get(i, 0.0), 'cider': cid.get(i, 0.0)}) + '\n')
print(len(ok), 'scored,', len(recs) - len(ok), 'empty set to 0.0 | meteor s: %.0f | cider s: %.0f' % (t1 - t0, time.time() - t1))
