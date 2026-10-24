"""Phase 3: Kendall tau-b between each automatic metric and the annotator's ratings on the new corpus.
Ratings: first pass (annotation/ratings.csv). System-level (5 systems) and summary-level (within-document, averaged)."""
import csv, json, os, statistics
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: os.path.join(ROOT, p)
DIMS = ["coherence", "consistency", "fluency", "relevance"]
METRICS = ["rouge1", "rouge2", "rougeL", "chrf", "bleu", "bertscore_p", "bertscore_r", "bertscore_f",
           "meteor", "cider", "mover_score"]

def tau_b(x, y):
    c = d = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0: continue
            if dx == 0: tx += 1
            elif dy == 0: ty += 1
            elif dx * dy > 0: c += 1
            else: d += 1
    den = ((c + d + tx) * (c + d + ty)) ** 0.5
    return (c - d) / den if den else None

# metric scores keyed by (id, model_id)
ms = defaultdict(dict)
for f in ["rouge", "toolkit", "bertscore", "meteor_cider", "moverscore"]:
    for l in open(P("results/scores/phase3_%s.jsonl" % f)):
        r = json.loads(l)
        ms[(r["id"], r["model_id"])].update({k: v for k, v in r.items() if k in METRICS})
key = list(csv.DictReader(open(P("annotation/key.csv"), encoding="utf-8")))
rat = {r["item"]: r for r in csv.DictReader(open(P("annotation/ratings.csv"), encoding="utf-8"))}
rows = [{"doc": k["doc_id"], "sys": k["model_id"], "m": ms[(k["doc_id"], k["model_id"])],
         "h": {d: int(rat[k["item"]][d]) for d in DIMS}} for k in key]
assert len(rows) == 150 and all(len(r["m"]) == len(METRICS) for r in rows), "join problem"

systems = sorted({r["sys"] for r in rows}); docs = sorted({r["doc"] for r in rows})
fmt = lambda v: "  n/a" if v is None else "%+.3f" % v
out = ["items: %d | documents: %d | systems: %d" % (len(rows), len(docs), len(systems)),
       "metric\t" + "\t".join("sys_" + d[:3] for d in DIMS) + "\t" + "\t".join("sum_" + d[:3] for d in DIMS)]
for m in METRICS:
    sysl, sml = [], []
    for d in DIMS:
        xs = [statistics.mean(r["m"][m] for r in rows if r["sys"] == s) for s in systems]
        hs = [statistics.mean(r["h"][d] for r in rows if r["sys"] == s) for s in systems]
        sysl.append(tau_b(xs, hs))
        per = []
        for doc in docs:
            g = [r for r in rows if r["doc"] == doc]
            t = tau_b([r["m"][m] for r in g], [r["h"][d] for r in g])
            if t is not None: per.append(t)
        sml.append(statistics.mean(per) if per else None)
    out.append(m + "\t" + "\t".join(fmt(v) for v in sysl) + "\t" + "\t".join(fmt(v) for v in sml))
print("\n".join(out))
open(P("results/tables/phase3_correlations.txt"), "w").write("\n".join(out) + "\n")
