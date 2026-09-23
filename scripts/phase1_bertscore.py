"""Phase 1 extension: system-level Kendall's tau for BERTScore P/R/F1.
Settings match the summ_eval toolkit's BertScoreMetric defaults:
bert-base-uncased, num_layers=8, idf=False, no baseline rescaling, 11 references."""
import json, statistics
from collections import defaultdict
from bert_score import score

PATH = "data/raw/model_annotations.aligned.jsonl"
OUT = "results/scores/bertscore.jsonl"
DIMS = ["coherence", "consistency", "fluency", "relevance"]
PAPER = {  # SummEval Table 2, expert, system-level, 11 references
    "bertscore_p": [0.0588, -0.1912, 0.0074, 0.1618],
    "bertscore_r": [0.1471, 0.6618, 0.4945, 0.3088],
    "bertscore_f": [0.2059, 0.0441, 0.2435, 0.4265]}

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
P, R, F = score([r["decoded"] for r in rows], [r["references"] for r in rows],
                model_type="bert-base-uncased", num_layers=8, idf=False,
                batch_size=64, verbose=True)

with open(OUT, "w") as fh:
    for r, p, rc, f in zip(rows, P.tolist(), R.tolist(), F.tolist()):
        fh.write(json.dumps({"id": r["id"], "model_id": r["model_id"],
                             "bertscore_p": p, "bertscore_r": rc, "bertscore_f": f}) + "\n")

vals = {"bertscore_p": P.tolist(), "bertscore_r": R.tolist(), "bertscore_f": F.tolist()}
scores, human = defaultdict(list), defaultdict(list)
for i, r in enumerate(rows):
    m = r["model_id"]
    for k in vals:
        scores[(m, k)].append(vals[k][i])
    for dim in DIMS:
        human[(m, dim)].append(statistics.mean(a[dim] for a in r["expert_annotations"]))

systems = sorted({r["model_id"] for r in rows})
print(f"\nsummaries: {len(rows)}   systems: {len(systems)}   references: 11")
print(f"per-summary scores saved to {OUT}\n")
print(f"{'metric':<13}{'dimension':<13}{'tau 11ref':>11}{'paper':>9}")
for k in vals:
    sys_metric = [statistics.mean(scores[(s, k)]) for s in systems]
    for i, dim in enumerate(DIMS):
        h = [statistics.mean(human[(s, dim)]) for s in systems]
        print(f"{k:<13}{dim:<13}{kendall_tau_b(sys_metric, h):>11.4f}{PAPER[k][i]:>9.4f}")
