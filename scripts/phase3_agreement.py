"""Phase 3: intra-annotator agreement, first pass vs re-rating of the 20% subset (30 items)."""
import csv, math, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: {r["item"]: r for r in csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8"))}
first, second = rd("annotation/ratings.csv"), rd("annotation/ratings_rerate.csv")
items = sorted(second, key=int)
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
    return (c - d) / den if den else float("nan")

def alpha_interval(a, b):
    n, N = len(a), 2 * len(a)
    v = a + b
    do = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    de = 2 * (N * sum(t * t for t in v) - sum(v) ** 2) / (N * (N - 1))
    return 1 - do / de if de else float("nan")

out = ["items re-rated: %d" % len(items), "dimension\texact\twithin1\ttau_b\talpha"]
for d in DIMS:
    a = [int(first[i][d]) for i in items]; b = [int(second[i][d]) for i in items]
    exact = sum(x == y for x, y in zip(a, b)) / len(a)
    w1 = sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
    out.append("%s\t%.2f\t%.2f\t%.3f\t%.3f" % (d, exact, w1, tau_b(a, b), alpha_interval(a, b)))
print("\n".join(out))
open(os.path.join(ROOT, "results/tables/phase3_agreement.txt"), "w").write("\n".join(out) + "\n")
