"""Phase 3: ROUGE-1/2/L F1 (Google rouge-score, stemmer, 1 reference) per summary on the new corpus.
Writes results/scores/phase3_rouge.jsonl. Not joined to human ratings yet (kept blind)."""
import json, os
from rouge_score import rouge_scorer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))]
sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
with open(os.path.join(ROOT, 'results/scores/phase3_rouge.jsonl'), 'w') as f:
    for r in recs:
        s = sc.score(r['references'][0], r['decoded'])
        f.write(json.dumps({'id': r['id'], 'model_id': r['model_id'],
                            **{m: s[m].fmeasure for m in ('rouge1', 'rouge2', 'rougeL')}}) + '\n')
print(len(recs), 'summaries scored -> results/scores/phase3_rouge.jsonl')
