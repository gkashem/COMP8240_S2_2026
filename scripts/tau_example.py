import json, re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

def tok(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).split()

def rouge1_f1(ref, cand):
    r, c = tok(ref), tok(cand)
    o = sum((Counter(r) & Counter(c)).values())
    return 0.0 if o == 0 else 2 * o / (len(r) + len(c))

by_article = defaultdict(list)
for line in Path("data/raw/model_annotations.aligned.jsonl").open():
    rec = json.loads(line)
    exp = rec["expert_annotations"]
    by_article[rec["id"]].append({
        "model": rec["model_id"],
        "rouge": rouge1_f1(rec["references"][0], rec["decoded"]),
        "human": sum(e["coherence"] for e in exp) / len(exp),
    })

def pairs_cd(items):
    C = D = 0
    for a, b in combinations(items, 2):
        s = (a["human"] - b["human"]) * (a["rouge"] - b["rouge"])
        C, D = (C + 1, D) if s > 0 else (C, D + 1)
    return C, D

found = None
for art, items in by_article.items():
    for combo in combinations(items, 4):
        hs = [x["human"] for x in combo]
        rs = [x["rouge"] for x in combo]
        if len(set(hs)) < 4 or len(set(rs)) < 4:
            continue
        C, D = pairs_cd(combo)
        if 1 <= D <= 2:
            found = (art, sorted(combo, key=lambda x: -x["human"]), C, D)
            break
    if found:
        break

art, combo, C, D = found
print(f"ARTICLE: {art}\n")
for lab, x in zip("ABCD", combo):
    print(f"  {lab}  model={x['model']:<4}  expert coherence={x['human']:.2f}  ROUGE-1 F1={x['rouge']:.3f}")
print("\nHuman ranking (best first): " + " > ".join("ABCD"))
print("Metric ranking (best first): " + " > ".join(
    lab for lab, _ in sorted(zip("ABCD", combo), key=lambda p: -p[1]["rouge"])))
print("\nPAIRWISE COMPARISON")
for (la, a), (lb, b) in combinations(list(zip("ABCD", combo)), 2):
    s = (a["human"] - b["human"]) * (a["rouge"] - b["rouge"])
    print(f"  ({la},{lb})  human {a['human']:.2f} vs {b['human']:.2f}   "
          f"metric {a['rouge']:.3f} vs {b['rouge']:.3f}   "
          f"-> {'CONCORDANT' if s > 0 else 'DISCORDANT'}")
print(f"\nC = {C}   D = {D}   total pairs = 6")
print("tau not printed - compute it yourself from Equation eq:tau")
