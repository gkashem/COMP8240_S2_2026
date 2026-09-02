import json, re
from collections import Counter
from pathlib import Path

TARGET = "dm-test-c50d33e9749e7bb484d9b69c4f5fca35a3a50cb5"
MODEL  = "M20"

def tok(text):
    # mirrors rouge-score's default tokenizer: lowercase, non-alphanumeric -> space
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()

rec = next(json.loads(l) for l in Path("data/raw/model_annotations.aligned.jsonl").open()
           if json.loads(l)["id"] == TARGET and json.loads(l)["model_id"] == MODEL)

ref, cand = tok(rec["references"][0]), tok(rec["decoded"])
print(f"REFERENCE tokens ({len(ref)}):\n  {ref}\n")
print(f"CANDIDATE tokens ({len(cand)}):\n  {cand}\n")

inter = Counter(ref) & Counter(cand)
print("UNIGRAM matches (clipped):")
for w, c in sorted(inter.items()):
    print(f"   {w:<12} x{c}")
print(f"  total unigram overlap = {sum(inter.values())}")
print(f"  |reference| = {len(ref)}   |candidate| = {len(cand)}\n")

bg = lambda t: [f"{a} {b}" for a, b in zip(t, t[1:])]
rb, cb = bg(ref), bg(cand)
bi = Counter(rb) & Counter(cb)
print("BIGRAM matches (clipped):")
for w, c in sorted(bi.items()):
    print(f"   {w!r} x{c}")
print(f"  total bigram overlap = {sum(bi.values())}")
print(f"  reference bigrams = {len(rb)}   candidate bigrams = {len(cb)}\n")

# longest common subsequence
m, n = len(ref), len(cand)
d = [[0]*(n+1) for _ in range(m+1)]
for i in range(m):
    for j in range(n):
        d[i+1][j+1] = d[i][j]+1 if ref[i] == cand[j] else max(d[i][j+1], d[i+1][j])
seq, i, j = [], m, n
while i and j:
    if ref[i-1] == cand[j-1]: seq.append(ref[i-1]); i, j = i-1, j-1
    elif d[i-1][j] >= d[i][j-1]: i -= 1
    else: j -= 1
print(f"LCS length = {d[m][n]}   LCS = {' '.join(reversed(seq))}")
print("\nNo scores printed on purpose - compute P, R and F1 yourself.")
