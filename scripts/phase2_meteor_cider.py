"""Phase 2: METEOR (evaluate_example) + CIDEr (evaluate_batch over all 2,400), authors' toolkit, 1 ref, on RealSumm
-> system-level Kendall tau-b vs LitePyramid recall (24 systems). Calls mirror phase1_meteor.py / phase1_cider.py."""
import json, math, os, time
from collections import defaultdict
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.cider_metric import CiderMetric
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
t0 = time.time(); m = MeteorMetric()
met = [m.evaluate_example(r['decoded'], r['references'])['meteor'] for r in recs]
t1 = time.time()
cid = [float(s['cider']) for s in CiderMetric().evaluate_batch([r['decoded'] for r in recs], [r['references'] for r in recs], aggregate=False)]
print('summaries: %d | meteor s: %.0f | cider s: %.0f' % (len(recs), t1 - t0, time.time() - t1))
sys_m = {'meteor': defaultdict(list), 'cider': defaultdict(list)}; sys_h = defaultdict(list)
with open(os.path.join(ROOT, 'results/scores/realsumm_meteor_cider.jsonl'), 'w') as f:
    for r, a, b in zip(recs, met, cid):
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'], 'meteor': a, 'cider': b}) + '\n')
        sys_m['meteor'][r['model_id']].append(a); sys_m['cider'][r['model_id']].append(b)
        sys_h[r['model_id']].append(r['litepyramid_recall'])
systems = sorted(sys_h); h = [mean(sys_h[k]) for k in systems]
out = ['summaries: %d | systems: %d | references: 1' % (len(recs), len(systems)), 'metric\ttau_litepyramid']
for k in sys_m: out.append('%s\t%.4f' % (k, tau_b([mean(sys_m[k][s]) for s in systems], h)))
print('\n'.join(out))
open(os.path.join(ROOT, 'results/tables/phase2_meteor_cider_tau.txt'), 'w').write('\n'.join(out) + '\n')
