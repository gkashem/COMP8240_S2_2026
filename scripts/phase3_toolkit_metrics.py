"""Phase 3: CHRF + BLEU (authors' summ_eval toolkit, evaluate_example, 1 reference) per summary on the new corpus.
Writes results/scores/phase3_toolkit.jsonl. Not joined to human ratings yet (kept blind)."""
import json, os
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.bleu_metric import BleuMetric
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))]
metrics = {"chrf": ChrfppMetric(), "bleu": BleuMetric()}
with open(os.path.join(ROOT, 'results/scores/phase3_toolkit.jsonl'), 'w') as f:
    for r in recs:
        row = {'id': r['id'], 'model_id': r['model_id']}
        for k, m in metrics.items():
            row[k] = m.evaluate_example(r['decoded'], r['references'])[k]
        f.write(json.dumps(row) + '\n')
print(len(recs), 'summaries scored -> results/scores/phase3_toolkit.jsonl')
