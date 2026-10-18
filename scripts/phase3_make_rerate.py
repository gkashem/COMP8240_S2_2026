"""Pick 20% of items (seed 8240) for blind re-rating. Writes annotation/sheet_rerate.csv."""
import csv, random

rows = list(csv.DictReader(open("annotation/sheet.csv", encoding="utf-8")))
source = {}
for r in rows:
    if len(r["source"].split()) > 20:
        source[r["doc"]] = r["source"]

random.seed(8240)
pick = random.sample(rows, round(0.2 * len(rows)))
random.shuffle(pick)
pick.sort(key=lambda r: int(r["doc"]))   # group by document; order within a document stays random

out, seen = [], set()
for r in pick:
    src = source[r["doc"]] if r["doc"] not in seen else "(same source)"
    seen.add(r["doc"])
    out.append({"item": r["item"], "doc": r["doc"], "source": src, "summary": r["summary"],
                "coherence": "", "consistency": "", "fluency": "", "relevance": ""})

with open("annotation/sheet_rerate.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(len(out), "items from", len(seen), "documents -> annotation/sheet_rerate.csv")
