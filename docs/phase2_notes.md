# Phase 2 — SummEval pipeline applied to RealSumm

## Dataset (existing): RealSumm (Bhandari et al., 2020)
- Source: github.com/neulab/REALSumm, `scores_dicts/abs.pkl` + `ext.pkl` (cloned to data/external/, gitignored)
- 100 CNN/DailyMail test documents, 1 reference summary each
- 25 system entries (14 abstractive + 11 extractive); `bart_out` appears in both with identical
  summaries -> kept once -> **24 systems, 2,400 summaries**
- Human score: **LitePyramid recall** (content-unit recall, 0–1), not SummEval's four 1–5 dimensions

## Preprocessing (`scripts/phase2_convert_realsumm.py`)
- Converted to SummEval-style JSONL: `data/processed/realsumm.jsonl`
- Removed `<t> </t>` sentence tags; collapsed whitespace
- NumPy float32 scores cast to float for JSON

## Results — system-level Kendall tau-b vs LitePyramid (24 systems, 1 reference)
| Metric | tau | Script |
|---|---|---|
| ROUGE-1 | 0.3261 | phase2_rouge.py |
| ROUGE-2 | 0.2899 | phase2_rouge.py |
| ROUGE-L | 0.1159 | phase2_rouge.py |
| CHRF | 0.7246 | phase2_toolkit_metrics.py |
| BLEU | 0.0435 | phase2_toolkit_metrics.py |
| BERTScore-P | 0.0145 | phase2_bertscore.py |
| BERTScore-R | 0.7464 | phase2_bertscore.py |
| BERTScore-F | 0.3188 | phase2_bertscore.py |
| METEOR | 0.8406 | phase2_meteor_cider.py |
| CIDEr | -0.4203 | phase2_meteor_cider.py |
| MoverScore | 0.2971 | phase2_moverscore.py |

Same aggregation (mean per system) and tau-b as Phase 1. Tables: `results/tables/phase2_*_tau.txt`.

## Sanity check
Tau using RealSumm's own precomputed ROUGE F1: R1 0.3116, R2 0.2681, RL 0.2246.
R1/R2 close to ours; RL differs. Inference (unverified): their ROUGE-L is sentence-level,
ours runs on one line because sentence tags were stripped.

## Challenges
- 1 reference vs SummEval's 11; different human-judgement protocol (content recall only)
- Duplicate system across abs/ext files; tagged text needed cleaning
- MoverScore requires the Phase 1 patches (CPU, library versions)
