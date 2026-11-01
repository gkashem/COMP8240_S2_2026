# Reproducing SummEval on New Data (COMP8240, 2026)

Replication of SummEval (Fabbri et al., 2021, TACL) and extension to RealSumm, a new Wikipedia/arXiv corpus, and an LLM judge.
Report: [`report/final/COMP8240_FinalReport_48728306.pdf`](report/final/COMP8240_FinalReport_48728306.pdf) (LaTeX source in `report/final/`).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | All code, one script per step (`phase1_*` … `phase4_*`) |
| `data/raw/MANIFEST.md` | Source URL and SHA-256 of the SummEval annotation file |
| `data/raw/phase3/` | Raw Wikipedia and arXiv records as collected |
| `data/processed/phase3_docs.jsonl` | **New corpus**: 30 documents (15 Wikipedia, 15 arXiv), with URL, licence and word counts |
| `data/processed/phase3_summaries.jsonl` | **New corpus**: 150 summaries from 5 systems |
| `data/processed/realsumm.jsonl` | RealSumm converted to SummEval-style JSONL (24 systems, 2,400 summaries) |
| `annotation/` | Guidelines, blind rating sheets, ratings (first pass and re-rating), key file |
| `results/scores/` | Per-summary metric scores and every LLM-judge call |
| `results/tables/` | All correlation and agreement tables reported in the paper |
| `docs/phase1_notes.md` … `phase4_notes.md` | Detailed notes, deviations and fixes for each phase |
| `logs/` | Install logs and the MoverScore patch (`moverscore_v2_patch.diff`) |

Not committed (fetched by the steps below): the SummEval annotation file, CNN/DailyMail articles, the RealSumm clone and model files.

## Setup

Tested in a GitHub Codespace (`.devcontainer/`), Debian 13, 2 CPU cores, no GPU, Python 3.12.3.

```bash
# 1. Toolkit environment (summ-eval pins 2020 versions, so install the frozen set without resolving dependencies)
python3 -m venv /home/vscode/summeval-env
source /home/vscode/summeval-env/bin/activate
pip install --no-deps -r requirements-summeval-env.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m nltk.downloader stopwords punkt punkt_tab
sudo apt-get install -y default-jre-headless          # METEOR needs Java

# 2. Google rouge-score runs in the system Python
/opt/conda/bin/python3 -m pip install rouge-score==0.1.2

# 3. Data
wget -P data/raw https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl
git clone https://github.com/neulab/REALSumm data/external/REALSumm
hf download abisee/cnn_dailymail 3.0.0/test-00000-of-00001.parquet --repo-type dataset --local-dir data/raw/cnndm

# 4. LLM judge model (Phase 4 only)
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models
```

Two toolkit fixes are needed (details in `docs/phase1_notes.md`): METEOR needs `paraphrase-en.gz` (from Maluuba/nlg-eval) in the toolkit's `data/` folder, and MoverScore needs the patch in `logs/moverscore_v2_patch.diff`.

## Running

Activate the venv first. Scripts marked **(sys)** run with `/opt/conda/bin/python3`; all others with `python3` from the venv. Run from the repository root.

| Phase | Scripts, in order |
|---|---|
| 1. Replication | `phase1_correlations.py` (sys), `phase1_toolkit_metrics.py`, `phase1_bertscore.py`, `phase1_meteor.py`, `phase1_cider.py`, `phase1_moverscore.py` |
| 2. RealSumm | `phase2_convert_realsumm.py`, `phase2_rouge.py` (sys), `phase2_toolkit_metrics.py`, `phase2_bertscore.py`, `phase2_meteor_cider.py`, `phase2_moverscore.py` |
| 3. New corpus | `phase3_collect_wikipedia.py`, `phase3_collect_arxiv.py`, `phase3_preprocess.py`, `phase3_extractive.py`, `phase3_abstractive.py`, `phase3_weak.py`, `phase3_make_annotation.py`, `phase3_rate.py`, `phase3_make_rerate.py`, `phase3_rerate.py`, `phase3_agreement.py`, `phase3_rouge.py` (sys), `phase3_toolkit_metrics.py`, `phase3_bertscore.py`, `phase3_meteor_cider.py`, `phase3_moverscore.py`, `phase3_correlations.py` |
| 4. LLM judge | `phase4_get_sources.py`, `phase4_judge.py --data summeval --out results/scores/phase4_judge_summeval.jsonl`, `phase4_judge.py --data phase3 --docs 30 --per_doc 5 --out results/scores/phase4_judge_phase3.jsonl` (add `--retry_failed --max_tokens 160` for calls that hit the token cap), `phase4_correlations.py`, `phase4_compare.py` (sys), `phase4_compare_phase3.py` |

Notes:
- The collection scripts sample random Wikipedia pages and recent arXiv records, so rerunning them gives different documents. Use the committed corpus to reproduce the reported results.
- `phase3_rate.py` and `phase3_rerate.py` are the interactive rating tools; the ratings they produced are in `annotation/`.
- The judge takes about 24 minutes per SummEval article on 2 CPU cores. All judge outputs are committed, so the correlation scripts can run without rerunning it.

## Data licences

- New corpus (`data/processed/phase3_*`, `data/raw/phase3/`, `annotation/`): derived from Wikipedia (CC BY-SA 4.0) and arXiv papers licensed CC BY, CC BY-SA or CC0; released under **CC BY-SA 4.0**. Each record keeps its source URL and licence. Summaries are machine-generated and may contain factual errors.
- `data/processed/realsumm.jsonl`: derived from RealSumm (Bhandari et al., 2020), CC BY 4.0.
- CNN/DailyMail articles and the SummEval annotation file are not redistributed here; the commands above fetch them.