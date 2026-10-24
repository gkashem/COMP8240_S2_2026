"""Phase 3: MoverScore v2 (authors' toolkit wrapper, patched moverscore_v2 as in Phase 1, 1 reference, one batch)
per summary on the new corpus. Mirrors phase2_moverscore.py. Empty summaries get 0.0 (excluded from the batch).
Writes results/scores/phase3_moverscore.jsonl. Not joined to human ratings yet (kept blind)."""
import json, os, time
from summ_eval.mover_score_metric import MoverScoreMetric
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))]
ok = [i for i, r in enumerate(recs) if r['decoded'].strip()]
t0 = time.time()
scores = MoverScoreMetric().evaluate_batch([recs[i]['decoded'] for i in ok],
                                           [recs[i]['references'] for i in ok], aggregate=False)
ms = {i: float(s['mover_score']) for i, s in zip(ok, scores)}
with open(os.path.join(ROOT, 'results/scores/phase3_moverscore.jsonl'), 'w') as f:
    for i, r in enumerate(recs):
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'], 'mover_score': ms.get(i, 0.0)}) + '\n')
print(len(ok), 'scored,', len(recs) - len(ok), 'empty set to 0.0 | runtime s: %.0f' % (time.time() - t0))
