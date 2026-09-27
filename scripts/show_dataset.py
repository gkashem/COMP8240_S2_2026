import json
from collections import Counter

# Load the released SummEval annotation file
rows = [json.loads(l) for l in open("data/raw/model_annotations.aligned.jsonl")]

# Where the articles come from (ID prefix: cnn- or dm-)
articles = {r["id"] for r in rows}
print("Source:", dict(Counter(a.split("-")[0] for a in articles)))

# Dataset size
print("Records:", len(rows))
print("Systems:", len({r["model_id"] for r in rows}))
print("Articles:", len(articles))
print("Experts:", len(rows[0]["expert_annotations"]),
      "| Crowd workers:", len(rows[0]["turker_annotations"]))
print("Dimensions:", list(rows[0]["expert_annotations"][0]))
