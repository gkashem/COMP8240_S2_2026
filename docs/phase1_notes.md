# Phase 1 — Replication on original data (SummEval Table 2, system-level)

Proposal §6: "Reproduces a defined subset of the paper's correlations using the original data and
extends the number of metrics evaluated. Any metric that cannot be run will be documented with the reason."

## Setup
- Data: `data/raw/model_annotations.aligned.jsonl` — 1,600 records, 100 articles, 16 systems,
  3 expert ratings per summary, 11 references per summary.
- Target: SummEval Table 2 — Kendall's tau, expert annotations, system level, 11 references.
- Aggregation (all scripts): mean of 3 experts per summary -> mean per system over its 100 summaries
  -> Kendall tau-b across 16 systems (hand-written, identical in every script).
- Environment: GitHub Codespace, CPU only. Toolkit venv `/home/vscode/summeval-env`
  (summ-eval installed `--no-deps`, sacrebleu==1.4.14, gin-config, torch 2.14.0+cpu,
  bert-score 0.3.12, moverscore 1.0.3, pyemd 2.0.0, OpenJDK 21.0.12.1).

## Results (tau, 11 refs) vs paper
| Metric | Dim | Ours | Paper | Script |
|---|---|---|---|---|
| ROUGE-1 | coh / con / flu / rel | 0.3667 / 0.3000 / 0.4770 / 0.5667 | 0.2500 / 0.5294 / 0.5240 / 0.4118 | phase1_correlations.py |
| ROUGE-2 | coh / con / flu / rel | 0.2500 / 0.1833 / 0.4603 / 0.4833 | 0.1618 / 0.5882 / 0.4797 / 0.2941 | phase1_correlations.py |
| ROUGE-L | coh / con / flu / rel | 0.3833 / 0.0833 / 0.3598 / 0.4833 | 0.0735 / 0.1471 / 0.2583 / 0.2353 | phase1_correlations.py |
| CHRF | coh / con / flu / rel | 0.3333 / 0.4667 / 0.5439 / 0.5333 | 0.3971 / 0.5294 / 0.4649 / 0.5882 | phase1_toolkit_metrics.py |
| BLEU | coh / con / flu / rel | 0.3833 / -0.0500 / 0.2259 / 0.2833 | 0.1176 / 0.0735 / 0.3321 / 0.2206 | phase1_toolkit_metrics.py |
| BERTScore-P | coh / con / flu / rel | -0.0167 / -0.2167 / 0.0251 / 0.0833 | 0.0588 / -0.1912 / 0.0074 / 0.1618 | phase1_bertscore.py |
| BERTScore-R | coh / con / flu / rel | 0.2000 / 0.6667 / 0.4937 / 0.4333 | 0.1471 / 0.6618 / 0.4945 / 0.3088 | phase1_bertscore.py |
| BERTScore-F | coh / con / flu / rel | 0.1333 / 0.0333 / 0.2762 / 0.4000 | 0.2059 / 0.0441 / 0.2435 / 0.4265 | phase1_bertscore.py |
| METEOR | coh / con / flu / rel | 0.2833 / 0.6500 / 0.6109 / 0.5500 | 0.2353 / 0.6324 / 0.6126 / 0.4265 | phase1_meteor.py |
| CIDEr | coh / con / flu / rel | 0.2000 / -0.2667 / -0.0921 / -0.0333 | 0.1176 / -0.1912 / -0.0221 / 0.1912 | phase1_cider.py |
| MoverScore | coh / con / flu / rel | 0.0167 / 0.1167 / 0.1590 / 0.2833 | 0.1912 / -0.0294 / 0.2583 / 0.2941 | phase1_moverscore.py |

Tables: `results/tables/phase1_*_tau.txt`. Per-summary scores: `results/scores/*.jsonl`.
Reproduced qualitative finding (proposal §2.2): BERTScore recall vs precision split on consistency.

## Metric coverage
| Metric | Status | Reason / fix |
|---|---|---|
| ROUGE-1/2/L | Run (substitute) | Google `rouge-score` (stemming) instead of Perl ROUGE-1.5.5; declared in proposal §4. |
| CHRF, BLEU | Run | Authors' toolkit. |
| BERTScore P/R/F | Run | `bert_score` with toolkit defaults (bert-base-uncased, layer 8, no idf, no rescale). |
| METEOR | Run | Fixed: `data/paraphrase-en.gz` missing from toolkit install; downloaded from Maluuba/nlg-eval. |
| CIDEr | Run | One batch over all 1,600 (IDF needs the full set). |
| MoverScore | Run, all 1,600 summaries (CPU, 3,240 s) | Installed `moverscore`, `pyemd` (unpinned 2.0.0); `stopwords.txt` fetched from Yale-LILY/SummEval; `moverscore_v2.py` patched (CPU, transformers/numpy API changes) — `logs/moverscore_v2_patch.diff`. |
| ROUGE-WE, S3 | Not run | Hard-coded embedding download (u.cs.biu.ac.il) times out. |
| SMS | Not run | `wmd` module missing. |
| SummaQA, BLANC, SUPERT, data stats | Deferred | Need source articles, not in the released jsonl (proposal §4 risk). |

## Choices not stated in SummEval.pdf (assumptions)
1. Combining 11 references: ROUGE = max F1 over refs; BERTScore = bert_score multi-ref;
   CHRF/BLEU/METEOR = toolkit `evaluate_example`; CIDEr = toolkit batch; MoverScore = toolkit batch
   (mean over the 11 refs; IDF from all summaries / all joined refs).
2. ROUGE stemming on.
3. Tau-b (paper says only "Kendall's tau").
4. System-level aggregation details (paper cites Louis and Nenkova 2013 only).
5. Toolkit defaults assumed to be the paper's settings (BERTScore, METEOR, CIDEr, MoverScore).
6. Library versions differ from 2020 (unpinned installs; MoverScore source patched).

## Inferences (not facts)
- Paper's tau values for coherence/consistency/relevance are multiples of 1/136 (17 items), ours of
  1/120 (16 systems). Suggests the paper ranked 17 entries; identity of the 17th unknown.
- Causes of the per-metric gaps are not established.
