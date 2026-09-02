"""Check that the SummEval authors' own evaluation toolkit installs and runs.

Environment (see logs/summ_eval_install_working.log):
    pip install --no-deps summ-eval
    pip install gin-config sacrebleu==1.4.14

The documented `pip install summ-eval` does not resolve on current Python;
see logs/summ_eval_install_attempt.log. Installing the package without its
2020-era dependency pins, then adding the dependencies the metrics below
actually use, does work.

This script runs two of the paper's own 14 metrics, CHRF and BLEU, from the
authors' toolkit over a sample of the released model summaries, scoring each
summary against the full set of references distributed with it.
"""
import json
import os
import urllib.request

DATA = "data/raw/model_annotations.aligned.jsonl"
URL = ("https://storage.googleapis.com/sfr-summarization-repo-research/"
       "model_annotations.aligned.jsonl")
SAMPLE = 50

if not os.path.exists(DATA):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    print("downloading", URL)
    urllib.request.urlretrieve(URL, DATA)

with open(DATA) as fh:
    records = [json.loads(line) for line in fh if line.strip()]

print("data file :", DATA, os.path.getsize(DATA), "bytes")
print("records   :", len(records))
print("fields    :", sorted(records[0].keys()))

import summ_eval
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.bleu_metric import BleuMetric

print("toolkit   :", os.path.dirname(summ_eval.__file__))

chrf_metric = ChrfppMetric()
bleu_metric = BleuMetric()

chrf_scores = []
bleu_scores = []
for record in records[:SAMPLE]:
    candidate = record["decoded"]
    references = record["references"]
    chrf_scores.append(chrf_metric.evaluate_example(candidate, references)["chrf"])
    bleu_scores.append(bleu_metric.evaluate_example(candidate, references)["bleu"])

n = len(chrf_scores)
print()
print("summaries scored :", n)
print("references each  :", len(records[0]["references"]))
print("CHRF mean        : %.4f" % (sum(chrf_scores) / n))
print("BLEU mean        : %.4f" % (sum(bleu_scores) / n))
print()
print("The authors' toolkit was installed and executed successfully.")
