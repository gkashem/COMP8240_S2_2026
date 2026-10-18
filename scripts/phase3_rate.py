"""Interactive rating of annotation/sheet.csv. Saves to annotation/ratings.csv after
every item; rerun to resume. Blind: never reads annotation/key.csv."""
import csv, os, textwrap

SHEET, OUT = "annotation/sheet.csv", "annotation/ratings.csv"
DIMS = ["coherence", "consistency", "fluency", "relevance"]

rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))
done = {}
if os.path.exists(OUT):
    done = {r["item"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"))}

def ask(dim):
    while True:
        v = input(f"  {dim} (1-5): ").strip()
        if v in {"1", "2", "3", "4", "5"}:
            return v

wrap = lambda t: "\n".join(textwrap.wrap(t, 100))
last_doc, source = None, ""
for i, r in enumerate(rows, 1):
    if len(r["source"].split()) > 20:
        source = r["source"]
    if r["item"] in done:
        continue
    if r["doc"] != last_doc:
        print("\n" + "=" * 100 + f"\nSOURCE (doc {r['doc']}):\n" + wrap(source))
        last_doc = r["doc"]
    print("-" * 100 + f"\nItem {i}/{len(rows)}  SUMMARY:\n" + wrap(r["summary"]))
    done[r["item"]] = {"item": r["item"], **{d: ask(d) for d in DIMS}}
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item"] + DIMS)
        w.writeheader()
        w.writerows(done.values())

print(f"\nAll {len(done)} items rated -> {OUT}")
