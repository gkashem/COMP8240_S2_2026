"""Phase 4: LLM judge vs ROUGE on the same 50 SummEval articles.
Kendall tau-b against mean expert ratings; system level (16 systems) and summary level
(within article, averaged). ROUGE computed as in Phase 1 (ref0 and max over 11 references)."""
import json, math, statistics
from rouge_score import rouge_scorer

DIMS = ["coherence", "consistency", "fluency", "relevance"]
METRICS = ["rouge1", "rouge2", "rougeL"]

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

judge = {}
for l in open('results/scores/phase4_judge_summeval.jsonl'):
    x = json.loads(l)
    if x['run'] == 0 and x['score'] is not None:
        judge[(x['id'], x['model_id'], x['dim'])] = x['score']
docs = sorted({k[0] for k in judge})
rows = [r for r in map(json.loads, open('data/raw/model_annotations.aligned.jsonl')) if r['id'] in set(docs)]
scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)

score, expert = {}, {}
for r in rows:
    key = (r['id'], r['model_id'])
    per_ref = [scorer.score(ref, r['decoded']) for ref in r['references']]
    for k in METRICS:
        score[key + (k + ' ref0',)] = per_ref[0][k].fmeasure
        score[key + (k + ' 11ref',)] = max(s[k].fmeasure for s in per_ref)
    for dim in DIMS:
        expert[key + (dim,)] = statistics.mean(a[dim] for a in r['expert_annotations'])
        if key + (dim,) in judge:
            score[key + ('judge ' + dim,)] = judge[key + (dim,)]
systems = sorted({r['model_id'] for r in rows})

def taus(name, dim):
    sj, sh = [], []
    for s in systems:
        p = [(score[(d, s, name)], expert[(d, s, dim)]) for d in docs if (d, s, name) in score]
        if p:
            sj.append(statistics.mean(q[0] for q in p)); sh.append(statistics.mean(q[1] for q in p))
    st = []
    for d in docs:
        p = [(score[(d, s, name)], expert[(d, s, dim)]) for s in systems if (d, s, name) in score]
        if len(p) >= 2:
            t = kendall_tau_b([q[0] for q in p], [q[1] for q in p])
            if not math.isnan(t): st.append(t)
    return kendall_tau_b(sj, sh), statistics.mean(st)

print(f"articles: {len(docs)}   summaries: {len(rows)}   systems: {len(systems)}   judge = run 0\n")
print(f"{'dimension':<13}{'metric':<14}{'system tau':>11}{'summary tau':>13}")
for dim in DIMS:
    for name in ['judge'] + [f'{k} {m}' for k in METRICS for m in ('ref0', '11ref')]:
        sy, su = taus('judge ' + dim if name == 'judge' else name, dim)
        print(f"{dim:<13}{name:<14}{sy:>11.4f}{su:>13.4f}")
    print()
