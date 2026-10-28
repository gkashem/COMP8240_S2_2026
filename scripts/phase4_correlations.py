"""Phase 4: Kendall tau-b between the LLM judge and SummEval expert ratings.
Same tau-b as Phase 1; system level (16 systems) and summary level (within article, averaged, as Phase 3).
Experts restricted to the articles the judge scored."""
import argparse, json, math, statistics
from collections import defaultdict, Counter

DIMS = ["coherence", "consistency", "fluency", "relevance"]

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
    den = ((c + d + tx) * (c + d + ty)) ** 0.5
    return (c - d) / den if den else math.nan

ap = argparse.ArgumentParser()
ap.add_argument('--judge', default='results/scores/phase4_judge_summeval.jsonl')
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.judge)]
judge = defaultdict(dict)
for x in rows:
    if x['score'] is not None:
        judge[(x['id'], x['model_id'], x['dim'])][x['run']] = x['score']
expert = {}
for l in open('data/raw/model_annotations.aligned.jsonl'):
    r = json.loads(l)
    for dim in DIMS:
        expert[(r['id'], r['model_id'], dim)] = statistics.mean(a[dim] for a in r['expert_annotations'])

docs = sorted({k[0] for k in judge}); systems = sorted({k[1] for k in judge})
keys = {(x['id'], x['model_id'], x['dim'], x['run']) for x in rows}
unres = len(keys - {k + (run,) for k, v in judge.items() for run in v})
print(f"calls: {len(rows)}   parse failures: {sum(x['score'] is None for x in rows)}   unresolved after retry: {unres}")
print(f"articles: {len(docs)}   systems: {len(systems)}   runs per item: {dict(Counter(len(v) for v in judge.values()))}")

multi = {k: v for k, v in judge.items() if len(v) >= 2}
if multi:
    print("\nrun-to-run variation (items with >=2 runs)")
    print(f"{'dimension':<13}{'items':>6}{'all runs equal':>16}{'mean SD':>9}")
    for dim in DIMS:
        vs = [list(v.values()) for k, v in multi.items() if k[2] == dim]
        if vs:
            same = sum(len(set(v)) == 1 for v in vs) / len(vs)
            sd = statistics.mean(statistics.pstdev(v) for v in vs)
            print(f"{dim:<13}{len(vs):>6}{same:>16.2f}{sd:>9.3f}")

def score(k, mode):
    v = judge.get(k)
    if not v: return None
    return v.get(0) if mode == 'run0' else statistics.mean(v.values())

for mode in (['run0', 'mean'] if multi else ['run0']):
    print(f"\njudge score = {mode}")
    print(f"{'dimension':<13}{'system tau':>11}{'summary tau':>13}{'articles':>10}")
    for dim in DIMS:
        sj, sh = [], []
        for s in systems:
            p = [(score((d, s, dim), mode), expert[(d, s, dim)]) for d in docs]
            p = [q for q in p if q[0] is not None]
            if not p: continue
            sj.append(statistics.mean(q[0] for q in p)); sh.append(statistics.mean(q[1] for q in p))
        taus = []
        for d in docs:
            p = [(score((d, s, dim), mode), expert[(d, s, dim)]) for s in systems]
            p = [q for q in p if q[0] is not None]
            if len(p) >= 2:
                t = kendall_tau_b([q[0] for q in p], [q[1] for q in p])
                if not math.isnan(t): taus.append(t)
        st = statistics.mean(taus) if taus else math.nan
        print(f"{dim:<13}{kendall_tau_b(sj, sh):>11.4f}{st:>13.4f}{len(taus):>10}")
