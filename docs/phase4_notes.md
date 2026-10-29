# Phase 4 notes: LLM judge

## Setup
- Model: Qwen3-4B, 4-bit GGUF (Q4_K_M, `Qwen/Qwen3-4B-GGUF`), llama-cpp-python 0.3.36, CPU-only Codespace (2 cores), n_ctx 2048, thinking mode off (`/no_think`).
- Local open-weight model as the proposal prefers (no API cost, no data sent to a third party).
- First candidate was Qwen2.5-3B-Instruct (Q4_K_M). On an 8-call test its reasoning mixed up dimensions (fluency penalised for "lacks context"; consistency judged as completeness) and 4/8 scores were within 1 point of the expert mean, vs 7/8 for Qwen3-4B on the same calls. Switched to Qwen3-4B. Indicative only (8 calls).
- Source articles: `model_annotations.aligned.jsonl` has no source text. The 100 article IDs were matched by hash to the CNN/DM 3.0.0 test split (`abisee/cnn_dailymail`); 100/100 matched (`scripts/phase4_get_sources.py`). The CNN/DM text is not committed (SummEval does not redistribute it either); the script rebuilds it.
- Model files (`models/`) and CNN/DM data are git-ignored.

## Prompt and run settings (`scripts/phase4_judge.py`)
- One call per dimension. Source shown first, then the summary, then the SummEval definition of that dimension (Fabbri et al., 2021, Sec. 4.1; the paper's instructions to annotators rephrased as direct instructions, e.g. "Penalize summaries that contain hallucinated facts").
- Scale: "1 (worst) to 5 (best)". No extra score anchors, following the proposal's choice of the paper's instructions over a new rubric.
- The judge gives a one-sentence justification (at most 25 words) before the score; output is JSON constrained by a grammar (score must be 1-5).
- Temperature 0.2; seed 8240 + run index. Order randomised with seed 8240: articles, summaries within an article, dimensions within a summary.
- Additions to the paper's instructions (declared):
  - SummEval: "the summary has been lowercased and tokenized; do not penalize casing or spacing around punctuation" (SummEval summaries are lowercased and tokenized, e.g. "# 42.5 million" for GBP 42.5m; without the note the first model penalised this as errors) and "Judge only {dimension}; ignore all other qualities".
  - Phase 3 corpus: "document" instead of "news article"/"article"; no lowercasing note (summaries are normally cased); "Judge only {dimension}" kept.
- Output cap 80 tokens. 64 SummEval calls hit the cap (JSON incomplete) and were re-run once with a 160-token cap (`--retry_failed --max_tokens 160`); 1 remained without a score and is excluded.
- Resumable: every call is appended to the scores file; restarts skip completed calls.

## Deviations from the proposal (declared)
1. 50 of SummEval's 100 articles (800 of 1,600 summaries), because of runtime on a 2-core CPU (about 24 min per article). The 50 are the first 50 of a seed-8240 shuffle.
2. Repeated runs only on a pilot (3 articles x 16 summaries x 4 dimensions x 3 runs = 576 calls); the full run is a single pass.
   Pilot: all 3 runs identical for 0.69 / 0.83 / 0.79 / 0.56 of items (coherence / consistency / fluency / relevance); mean SD 0.157 / 0.105 / 0.118 / 0.245 (`results/tables/phase4_pilot.txt`).
   On the final 50 articles, averaging the 3 pilot runs instead of using run 0 changes every tau by at most 0.02 (`results/tables/phase4_correlations.txt`).
3. Scores defined only by their end points (see Prompt).
4. A small model (4B, 4-bit) rather than the more capable model the judge guide recommends; limited by hardware.

## SummEval results (50 articles, 16 systems; judge = run 0)
Kendall tau-b against the mean of the 3 expert ratings, experts restricted to the same 50 articles (`results/tables/phase4_compare.txt`).

| dimension | judge system | best ROUGE system | judge summary | best ROUGE summary |
|---|---|---|---|---|
| coherence | 0.4576 | 0.4370 (R-1, 11 refs) | 0.2544 | 0.1947 (R-L, 11 refs) |
| consistency | 0.4915 | 0.3530 (R-1/R-2, 11 refs) | 0.3013 | 0.2092 (R-2, 11 refs) |
| fluency | 0.6891 | 0.3264 (R-1, 11 refs) | 0.4736 | 0.1544 (R-L, 11 refs) |
| relevance | 0.7395 | 0.5546 (R-1, 11 refs) | 0.2838 | 0.2421 (R-1, 11 refs) |

The judge is highest on every dimension at both levels. No significance test; system level rests on 16 systems.
Not comparable directly with SummEval Table 2 (100 articles).

## Phase 3 corpus results
To be added (`scripts/phase4_compare_phase3.py`, `results/tables/phase4_phase3_judge.txt`).

## Known limitations of the judge
- Coarse 1-5 integer scale; ties are common.
- Run-to-run variation measured only on the pilot.
- Position and verbosity bias not tested. Self-enhancement bias is unlikely to apply (Qwen3 did not produce any of the judged summaries), but this was not tested.
- Small quantised model; agreement with a stronger model was not tested.
