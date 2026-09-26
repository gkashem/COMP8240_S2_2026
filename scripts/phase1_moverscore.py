"""Phase 1: MoverScore v2 (authors' toolkit wrapper, 11 refs, one batch over all 1,600) -> system-level Kendall tau-b vs expert ratings.
moverscore_v2.py patched for CPU + current transformers/numpy; see docs/phase1_notes.md."""
import json, math, os, time
from collections import defaultdict
from summ_eval.mover_score_metric import MoverScoreMetric

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data/raw/model_annotations.aligned.jsonl')
DIMS = ['coherence', 'consistency', 'fluency', 'relevance']
PAPER = {'coherence': 0.1912, 'consistency': -0.0294, 'fluency': 0.2583, 'relevance': 0.2941}  # SummEval Table 2

def tau_b(x, y):
    n = len(x); n0 = n * (n - 1) / 2; s = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            s += (dx > 0) - (dx < 0) if dy > 0 else -((dx > 0) - (dx < 0)) if dy < 0 else 0
            n1 += dx == 0; n2 += dy == 0
    return s / math.sqrt((n0 - n1) * (n0 - n2))

mean = lambda v: sum(v) / len(v)
recs = [json.loads(l) for l in open(DATA)]
t0 = time.time()
scores = MoverScoreMetric().evaluate_batch([r['decoded'] for r in recs], [r['references'] for r in recs], aggregate=False)
print('summaries scored:', len(scores), '| runtime s: %.0f' % (time.time() - t0))
sys_m, sys_h = defaultdict(list), {d: defaultdict(list) for d in DIMS}
with open(os.path.join(ROOT, 'results/scores/moverscore.jsonl'), 'w') as f:
    for r, sc in zip(recs, scores):
        s = float(sc['mover_score'])
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'], 'mover_score': s}) + '\n')
        sys_m[r['model_id']].append(s)
        for d in DIMS:
            sys_h[d][r['model_id']].append(mean([a[d] for a in r['expert_annotations']]))
systems = sorted(sys_m)
x = [mean(sys_m[k]) for k in systems]
out = ['systems: %d' % len(systems), 'metric\tdim\ttau_11ref\tpaper']
for d in DIMS:
    out.append('moverscore\t%s\t%.4f\t%.4f' % (d, tau_b(x, [mean(sys_h[d][k]) for k in systems]), PAPER[d]))
print('\n'.join(out))
open(os.path.join(ROOT, 'results/tables/phase1_moverscore_tau.txt'), 'w').write('\n'.join(out) + '\n')
