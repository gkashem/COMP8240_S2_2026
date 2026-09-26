"""Phase 2: MoverScore v2 (authors' toolkit wrapper, patched moverscore_v2 as in Phase 1, 1 ref, one batch over all 2,400)
on RealSumm -> system-level Kendall tau-b vs LitePyramid recall (24 systems). Mirrors phase1_moverscore.py."""
import json, math, os, time
from collections import defaultdict
from summ_eval.mover_score_metric import MoverScoreMetric
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def tau_b(x, y):
    n = len(x); n0 = n * (n - 1) / 2; s = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            s += (dx > 0) - (dx < 0) if dy > 0 else -((dx > 0) - (dx < 0)) if dy < 0 else 0
            n1 += dx == 0; n2 += dy == 0
    return s / math.sqrt((n0 - n1) * (n0 - n2))

mean = lambda v: sum(v) / len(v)
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/realsumm.jsonl'))]
t0 = time.time()
scores = MoverScoreMetric().evaluate_batch([r['decoded'] for r in recs], [r['references'] for r in recs], aggregate=False)
print('summaries scored:', len(scores), '| runtime s: %.0f' % (time.time() - t0))
sys_m, sys_h = defaultdict(list), defaultdict(list)
with open(os.path.join(ROOT, 'results/scores/realsumm_moverscore.jsonl'), 'w') as f:
    for r, sc in zip(recs, scores):
        s = float(sc['mover_score'])
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'], 'mover_score': s}) + '\n')
        sys_m[r['model_id']].append(s); sys_h[r['model_id']].append(r['litepyramid_recall'])
systems = sorted(sys_h)
out = ['systems: %d | references: 1' % len(systems), 'metric\ttau_litepyramid',
       'moverscore\t%.4f' % tau_b([mean(sys_m[k]) for k in systems], [mean(sys_h[k]) for k in systems])]
print('\n'.join(out))
open(os.path.join(ROOT, 'results/tables/phase2_moverscore_tau.txt'), 'w').write('\n'.join(out) + '\n')
