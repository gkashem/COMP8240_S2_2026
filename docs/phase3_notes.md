# Phase 3 — Constructed non-news corpus (complete)

## Collection
| Source | Docs | Reference | Source text | Licence | Script |
|---|---|---|---|---|---|
| Wikipedia (random articles) | 15 | lead section | body | CC BY-SA 4.0 | phase3_collect_wikipedia.py |
| arXiv (cs.*) | 15 | abstract | Introduction (arxiv.org/html) | CC BY / BY-SA / CC0 only | phase3_collect_arxiv.py |
Filters (project choice): reference >= 40 words, source >= 400 words. 65 Wikipedia pages and 177 arXiv records checked.

## Preprocessing (`phase3_preprocess.py`)
- Wikipedia: cut at end sections (See also / References / ...), `== headings ==` removed
- Sources truncated at a sentence boundary to <= 656 words = median of the 100 CNN/DailyMail articles
  in RealSumm's src.txt (measured: min 227, median 656, max 1,903)
- Result: all 30 sources 435–656 words. References kept whole.

## Summaries — 5 systems x 30 docs = 150 (`data/processed/phase3_summaries.jsonl`)
| System | Type | Avg words | Novel bigrams | Script |
|---|---|---|---|---|
| lead3 | extractive baseline | 71.8 | 0.0% | phase3_extractive.py |
| textrank | unsupervised extractive | 115.0 | 1.4% | phase3_extractive.py |
| bart_large_cnn | abstractive, larger | 56.2 | 7.6% | phase3_abstractive.py |
| distilbart_cnn_6_6 | abstractive, smaller | 50.3 | 6.7% | phase3_abstractive.py |
| distilbart_weak | deliberately weaker (greedy, 30 tokens) | — | — | phase3_weak.py |
Runtime on CPU: bart-large-cnn 701 s, distilbart 928 s, weak 418 s.

## Annotation — complete
- Guidelines: `annotation/guidelines.md` (SummEval's four definitions, quoted from the paper; the 1/3/5 anchor
  descriptions are my own additions).
- First pass: all 150 items rated 1–5 on four dimensions with `scripts/phase3_rate.py`, blind
  (`annotation/key.csv` not opened until all ratings were finished). Output `annotation/ratings.csv`.
- Re-rating: 20% subset = 30 items from 20 documents (seed 8240, `phase3_make_rerate.py`), rated again six days
  later with `phase3_rerate.py` -> `annotation/ratings_rerate.csv`.
- Intra-annotator agreement (`phase3_agreement.py`, `results/tables/phase3_agreement.txt`), 30 items:

| Dimension | Exact | Within 1 | Kendall tau-b | Krippendorff alpha (interval) |
|---|---|---|---|---|
| coherence | 0.67 | 0.97 | 0.740 | 0.768 |
| consistency | 0.87 | 1.00 | 0.650 | 0.905 |
| fluency | 0.60 | 1.00 | 0.724 | 0.809 |
| relevance | 0.33 | 0.93 | 0.711 | 0.508 |

  Alpha is above the ~0.67 reliability threshold (proposal §2.3) for three dimensions and below it for
  relevance. Alpha treats 1–5 as an interval scale. n = 30, so all figures are approximate.

## Metrics on the new corpus (per-summary scores in `results/scores/phase3_*.jsonl`)
| Metrics | Script |
|---|---|
| ROUGE-1/2/L (Google rouge-score, 1 reference) | phase3_rouge.py |
| CHRF, BLEU (authors' toolkit) | phase3_toolkit_metrics.py |
| BERTScore P/R/F | phase3_bertscore.py |
| METEOR, CIDEr | phase3_meteor_cider.py |
| MoverScore | phase3_moverscore.py |
Settings mirror Phase 2. One summary is empty (`arxiv-2402.01369`, distilbart_weak). BERTScore, METEOR, CIDEr and
MoverScore score it 0.0 (excluded from their batch); ROUGE, CHRF and BLEU scored it directly without error.

## Correlations (`phase3_correlations.py`, `results/tables/phase3_correlations.txt`)
Kendall tau-b of each metric vs the annotator's first-pass ratings, at system level (5 systems) and summary level
(5 summaries per document, averaged over 30 documents).
- Relevance agrees best with the metrics (summary-level roughly 0.3–0.53 for most; system-level 0.6–0.8).
- Coherence agrees worst (summary-level 0.04–0.32), in line with the paper's finding for word-overlap metrics.
- BERTScore recall vs precision split on consistency reproduced (system-level +0.6 vs -0.4).
- The paper found relevance weak; here it is strongest. Cause not established.
- Caveats: 5 systems (each tau step is one system pair), one annotator, relevance had the lowest self-agreement.

## Challenges
- Abstractive models (trained on CNN/DailyMail news) copy heavily: 7.6% / 6.7% novel bigrams
- Reference length varies widely (46–762 words); one Wikipedia reference (762 w) is longer than its truncated source (603 w)
- Wikipedia markup (headings, end sections) needed cleaning; arXiv needs licence filtering and HTML parsing
- Single unpaid annotator vs SummEval's 3 experts -> treated as an internally consistent ranking
- BERTScore failed with an AttributeError (transformers 5.17.0) until the one empty summary was excluded
- MoverScore warned that at least one input exceeded 512 tokens (1,039); its score may reflect truncated text (not checked which)
