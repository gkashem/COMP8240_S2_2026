"""Phase 2: ROUGE-1/2/L F1 (Google rouge-score, stemmer, 1 ref) on RealSumm -> system-level Kendall tau-b
vs LitePyramid recall (24 systems). Same aggregation + tau-b as Phase 1."""
import json, math, os
from collections import defaultdict
from rouge_score import rouge_scorer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/realsumm.jsonl'))]
def tau_b(x, y):
    n = len(x); n0 = n * (n - 1) / 2; s = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            s += (dx > 0) - (dx < 0) if dy > 0 else -((dx > 0) - (dx < 0)) if dy < 0 else 0
            n1 += dx == 0; n2 += dy == 0
    return s / math.sqrt((n0 - n1) * (n0 - n2))
mean = lambda v: sum(v) / len(v)
sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
M = ['rouge1', 'rouge2', 'rougeL']
sys_m = {m: defaultdict(list) for m in M}; sys_h = defaultdict(list)
for r in recs:
    s = sc.score(r['references'][0], r['decoded'])
    for m in M: sys_m[m][r['model_id']].append(s[m].fmeasure)
    sys_h[r['model_id']].append(r['litepyramid_recall'])
systems = sorted(sys_h); h = [mean(sys_h[k]) for k in systems]
out = ['summaries: %d | systems: %d' % (len(recs), len(systems)), 'metric\ttau_litepyramid']
for m in M: out.append('%s\t%.4f' % (m, tau_b([mean(sys_m[m][k]) for k in systems], h)))
print('\n'.join(out))
os.makedirs(os.path.join(ROOT, 'results/tables'), exist_ok=True)
open(os.path.join(ROOT, 'results/tables/phase2_rouge_tau.txt'), 'w').write('\n'.join(out) + '\n')
