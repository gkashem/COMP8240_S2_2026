"""Phase 3: BERTScore P/R/F1 per summary on the new corpus. Settings mirror phase2_bertscore.py:
bert-base-uncased, num_layers=8, idf=False, no rescaling; 1 reference.
Empty summaries (BERTScore cannot tokenize them) get 0.0, as with ROUGE.
Writes results/scores/phase3_bertscore.jsonl. Not joined to human ratings yet (kept blind)."""
import json, os
from bert_score import score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = [json.loads(l) for l in open(os.path.join(ROOT, 'data/processed/phase3_summaries.jsonl'))]
ok = [i for i, r in enumerate(recs) if r["decoded"].strip()]
P, R, F = score([recs[i]["decoded"] for i in ok], [recs[i]["references"] for i in ok],
                model_type="bert-base-uncased", num_layers=8, idf=False, batch_size=32, verbose=True)
vals = {k: [0.0] * len(recs) for k in ("bertscore_p", "bertscore_r", "bertscore_f")}
for n, i in enumerate(ok):
    vals["bertscore_p"][i], vals["bertscore_r"][i], vals["bertscore_f"][i] = P[n].item(), R[n].item(), F[n].item()
with open(os.path.join(ROOT, 'results/scores/phase3_bertscore.jsonl'), "w") as fh:
    for i, r in enumerate(recs):
        fh.write(json.dumps({"id": r["id"], "model_id": r["model_id"], **{k: vals[k][i] for k in vals}}) + "\n")
print(len(ok), "scored,", len(recs) - len(ok), "empty set to 0.0 ->", "results/scores/phase3_bertscore.jsonl")
