"""Phase 1 extension: system-level Kendall's tau for CHRF and BLEU via the authors' summ_eval toolkit."""
import json, statistics
from collections import defaultdict
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.bleu_metric import BleuMetric

PATH = "data/raw/model_annotations.aligned.jsonl"
DIMS = ["coherence", "consistency", "fluency", "relevance"]
PAPER = {  # SummEval Table 2, expert, system-level, 11 references
    "chrf": [0.3971, 0.5294, 0.4649, 0.5882],
    "bleu": [0.1176, 0.0735, 0.3321, 0.2206]}

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

rows = [json.loads(l) for l in open(PATH)]
metrics = {"chrf": ChrfppMetric(), "bleu": BleuMetric()}

scores = defaultdict(list)
human = defaultdict(list)
for r in rows:
    m = r["model_id"]
    for k, metric in metrics.items():
        scores[(m, k)].append(metric.evaluate_example(r["decoded"], r["references"])[k])
    for dim in DIMS:
        human[(m, dim)].append(statistics.mean(a[dim] for a in r["expert_annotations"]))

systems = sorted({r["model_id"] for r in rows})
print(f"summaries: {len(rows)}   systems: {len(systems)}   references: 11\n")
print(f"{'metric':<8}{'dimension':<13}{'tau 11ref':>11}{'paper':>9}")
for k in metrics:
    sys_metric = [statistics.mean(scores[(s, k)]) for s in systems]
    for i, dim in enumerate(DIMS):
        h = [statistics.mean(human[(s, dim)]) for s in systems]
        print(f"{k:<8}{dim:<13}{kendall_tau_b(sys_metric, h):>11.4f}{PAPER[k][i]:>9.4f}")
