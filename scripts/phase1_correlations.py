"""Phase 1: system-level Kendall's tau between ROUGE and expert ratings (SummEval Table 2 subset)."""
import json, statistics
from collections import defaultdict
from rouge_score import rouge_scorer

PATH = "data/raw/model_annotations.aligned.jsonl"
DIMS = ["coherence", "consistency", "fluency", "relevance"]
METRICS = ["rouge1", "rouge2", "rougeL"]
PAPER = {  # SummEval Table 2, expert, system-level, 11 references
    ("rouge1", "coherence"): 0.2500, ("rouge1", "consistency"): 0.5294,
    ("rouge1", "fluency"): 0.5240, ("rouge1", "relevance"): 0.4118,
    ("rouge2", "coherence"): 0.1618, ("rouge2", "consistency"): 0.5882,
    ("rouge2", "fluency"): 0.4797, ("rouge2", "relevance"): 0.2941,
    ("rougeL", "coherence"): 0.0735, ("rougeL", "consistency"): 0.1471,
    ("rougeL", "fluency"): 0.2583, ("rougeL", "relevance"): 0.2353}

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
scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)

metric = {"ref0": defaultdict(list), "multi": defaultdict(list)}
human = defaultdict(list)
for r in rows:
    m = r["model_id"]
    per_ref = [scorer.score(ref, r["decoded"]) for ref in r["references"]]
    for k in METRICS:
        metric["ref0"][(m, k)].append(per_ref[0][k].fmeasure)
        metric["multi"][(m, k)].append(max(s[k].fmeasure for s in per_ref))
    for dim in DIMS:
        human[(m, dim)].append(statistics.mean(a[dim] for a in r["expert_annotations"]))

systems = sorted({r["model_id"] for r in rows})
print(f"summaries: {len(rows)}   systems: {len(systems)}\n")
print(f"{'metric':<8}{'dimension':<13}{'tau ref0':>10}{'tau 11ref':>11}{'paper':>9}")
for k in METRICS:
    for dim in DIMS:
        h = [statistics.mean(human[(s, dim)]) for s in systems]
        t = {mode: kendall_tau_b([statistics.mean(metric[mode][(s, k)]) for s in systems], h)
             for mode in metric}
        p = PAPER.get((k, dim))
        print(f"{k:<8}{dim:<13}{t['ref0']:>10.4f}{t['multi']:>11.4f}{(f'{p:.4f}' if p is not None else '-'):>9}")
