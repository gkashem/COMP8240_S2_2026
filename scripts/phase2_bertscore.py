"""Phase 2: BERTScore P/R/F1 on RealSumm -> system-level Kendall tau-b vs LitePyramid recall (24 systems).
Settings mirror phase1_bertscore.py: bert-base-uncased, num_layers=8, idf=False, no rescaling; 1 reference."""
import json, os, statistics
from collections import defaultdict
from bert_score import score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results/scores/realsumm_bertscore.jsonl')

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
P, R, F = score([r["decoded"] for r in rows], [r["references"] for r in rows],
                model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=64, verbose=True)
vals = {"bertscore_p": P.tolist(), "bertscore_r": R.tolist(), "bertscore_f": F.tolist()}
with open(OUT, "w") as fh:
    for i, r in enumerate(rows):
        fh.write(json.dumps({"id": r["id"], "model_id": r["model_id"], **{k: vals[k][i] for k in vals}}) + "\n")
scores, human = defaultdict(list), defaultdict(list)
for i, r in enumerate(rows):
    for k in vals: scores[(r["model_id"], k)].append(vals[k][i])
    human[r["model_id"]].append(r["litepyramid_recall"])
systems = sorted(human); h = [statistics.mean(human[s]) for s in systems]
out = ["summaries: %d | systems: %d | references: 1" % (len(rows), len(systems)), "metric\ttau_litepyramid"]
for k in vals: out.append("%s\t%.4f" % (k, kendall_tau_b([statistics.mean(scores[(s, k)]) for s in systems], h)))
print("\n".join(out))
open(os.path.join(ROOT, 'results/tables/phase2_bertscore_tau.txt'), 'w').write("\n".join(out) + "\n")
