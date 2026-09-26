"""Phase 2: CHRF + BLEU (authors' summ_eval toolkit, evaluate_example, 1 ref) on RealSumm ->
system-level Kendall tau-b vs LitePyramid recall (24 systems). Mirrors phase1_toolkit_metrics.py."""
import json, os, statistics
from collections import defaultdict
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.bleu_metric import BleuMetric
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def kendall_tau_b(x, y):
    c = d = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0: continue
            if dx == 0: tx += 1
            elif dy == 0: ty += 1
            elif dx * dy > 0: c += 1
            else: d += 1
    return (c - d) / ((c + d + tx) * (c + d + ty)) ** 0.5

rows = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/realsumm.jsonl'))]
metrics = {"chrf": ChrfppMetric(), "bleu": BleuMetric()}
scores, human = defaultdict(list), defaultdict(list)
for r in rows:
    for k, metric in metrics.items():
        scores[(r["model_id"], k)].append(metric.evaluate_example(r["decoded"], r["references"])[k])
    human[r["model_id"]].append(r["litepyramid_recall"])
systems = sorted(human)
h = [statistics.mean(human[s]) for s in systems]
out = ["summaries: %d | systems: %d | references: 1" % (len(rows), len(systems)), "metric\ttau_litepyramid"]
for k in metrics:
    out.append("%s\t%.4f" % (k, kendall_tau_b([statistics.mean(scores[(s, k)]) for s in systems], h)))
print("\n".join(out))
open(os.path.join(ROOT, 'results/tables/phase2_toolkit_tau.txt'), 'w').write("\n".join(out) + "\n")
