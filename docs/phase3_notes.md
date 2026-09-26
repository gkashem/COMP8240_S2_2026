# Phase 3 — Constructed non-news corpus (progress)

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

## Annotation — set up, not started
- `annotation/sheet.csv`: 150 items, system names hidden, documents and summaries shuffled (seed 8240),
  4 dimensions rated 1–5, source shown per document. `annotation/key.csv` maps items to systems.
- Planned (proposal §5.2): 20% subset re-annotated after a delay for intra-annotator agreement.

## Challenges
- Abstractive models (trained on CNN/DailyMail news) copy heavily: 7.6% / 6.7% novel bigrams
- Reference length varies widely (46–762 words); one Wikipedia reference (762 w) is longer than its truncated source (603 w)
- Wikipedia markup (headings, end sections) needed cleaning; arXiv needs licence filtering and HTML parsing
- Single unpaid annotator vs SummEval's 3 experts -> treated as an internally consistent ranking
