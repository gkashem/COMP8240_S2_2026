"""Feasibility check: load the released SummEval annotations and score a sample.

Evidence for the proposal's 'Justification for Paper Choice' section, which
requires that we download and execute the relevant software successfully.
"""
import json, statistics
from rouge_score import rouge_scorer

PATH = "data/raw/model_annotations.aligned.jsonl"
rows = [json.loads(l) for l in open(PATH)]
print(f"loaded {len(rows)} annotated summaries")
print(f"distinct models:   {len({r['model_id'] for r in rows})}")
print(f"distinct articles: {len({r['id'] for r in rows})}")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
sample = rows[:50]
for key in ["rouge1", "rouge2", "rougeL"]:
    vals = [scorer.score(r["references"][0], r["decoded"])[key].fmeasure for r in sample]
    print(f"mean {key} F1 (n=50): {statistics.mean(vals):.4f}")

for dim in ["coherence", "consistency", "fluency", "relevance"]:
    vals = [statistics.mean(a[dim] for a in r["expert_annotations"]) for r in sample]
    print(f"mean expert {dim:12s}: {statistics.mean(vals):.3f}")
