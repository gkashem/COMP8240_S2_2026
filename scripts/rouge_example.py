import json, sys
from collections import Counter
from pathlib import Path

PATH = Path("data/raw/model_annotations.aligned.jsonl")
if not PATH.exists():
    sys.exit(f"MISSING {PATH} - re-download it first")

records = [json.loads(line) for line in PATH.open()]
print(f"loaded {len(records)} records\n")

rows = []
for r in records:
    ref = r["references"][0].split()   # the original CNN/DailyMail highlight
    dec = r["decoded"].split()
    overlap = sum((Counter(ref) & Counter(dec)).values())
    if overlap < 3:
        continue
    rows.append((len(ref) + len(dec), overlap, r))

rows.sort(key=lambda t: t[0])

for n, (total, overlap, r) in enumerate(rows[:5], 1):
    ref, dec = r["references"][0], r["decoded"]
    exp = r["expert_annotations"]
    means = {k: round(sum(e[k] for e in exp) / len(exp), 2)
             for k in ("coherence", "consistency", "fluency", "relevance")}
    print("=" * 72)
    print(f"OPTION {n}   id={r['id']}   model={r['model_id']}")
    print(f"REFERENCE ({len(ref.split())} tokens):\n  {ref}\n")
    print(f"CANDIDATE ({len(dec.split())} tokens):\n  {dec}\n")
    print(f"clipped unigram overlap = {overlap}")
    print(f"expert means = {means}")
