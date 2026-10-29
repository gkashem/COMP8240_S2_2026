"""Phase 4: LLM judge vs the annotator's first-pass ratings on the Phase 3 corpus.
Same method and format as scripts/phase3_correlations.py (tau-b; system level over 5 systems;
summary level within document, averaged). Judge dimension compared with the same human dimension."""
import csv, json, os, statistics
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda p: os.path.join(ROOT, p)
DIMS = ["coherence", "consistency", "fluency", "relevance"]

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

judge, calls = {}, 0
for l in open(P("results/scores/phase4_judge_phase3.jsonl")):
    x = json.loads(l)
    if x["run"] == 0:
        calls += 1
        if x["score"] is not None: judge[(x["id"], x["model_id"], x["dim"])] = x["score"]
key = list(csv.DictReader(open(P("annotation/key.csv"), encoding="utf-8")))
rat = {r["item"]: r for r in csv.DictReader(open(P("annotation/ratings.csv"), encoding="utf-8"))}
rows = [{"doc": k["doc_id"], "sys": k["model_id"], "h": {d: int(rat[k["item"]][d]) for d in DIMS}} for k in key]
assert len(rows) == 150, "join problem"

fmt = lambda v: "  n/a" if v is None else "%+.3f" % v
out = ["items: %d | judge calls: %d | valid judge scores: %d (of %d)" % (len(rows), calls, len(judge), 4 * len(rows)),
       "metric\t" + "\t".join("sys_" + d[:3] for d in DIMS) + "\t" + "\t".join("sum_" + d[:3] for d in DIMS)]
sysl, sml, dist = [], [], []
for d in DIMS:
    g = [r for r in rows if (r["doc"], r["sys"], d) in judge]
    J = lambda r: judge[(r["doc"], r["sys"], d)]
    systems = sorted({r["sys"] for r in g}); docs = sorted({r["doc"] for r in g})
    xs = [statistics.mean(J(r) for r in g if r["sys"] == s) for s in systems]
    hs = [statistics.mean(r["h"][d] for r in g if r["sys"] == s) for s in systems]
    sysl.append(tau_b(xs, hs))
    per = []
    for doc in docs:
        gd = [r for r in g if r["doc"] == doc]
        t = tau_b([J(r) for r in gd], [r["h"][d] for r in gd])
        if t is not None: per.append(t)
    sml.append(statistics.mean(per) if per else None)
    cj, ch = Counter(J(r) for r in g), Counter(r["h"][d] for r in g)
    dist.append("%-12s judge %s | human %s | summary-level docs with defined tau: %d/%d" % (
        d, " ".join("%d:%d" % (v, cj[v]) for v in range(1, 6)), " ".join("%d:%d" % (v, ch[v]) for v in range(1, 6)),
        len(per), len(docs)))
out.append("llm_judge\t" + "\t".join(fmt(v) for v in sysl) + "\t" + "\t".join(fmt(v) for v in sml))
out += ["", "score distributions (items with a valid judge score)"] + dist
print("\n".join(out))
open(P("results/tables/phase4_phase3_judge.txt"), "w").write("\n".join(out) + "\n")
