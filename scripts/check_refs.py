"""Compare custom.bib against the official ACL Anthology BibTeX for each entry.

Downloads https://aclanthology.org/<ID>.bib for every entry whose url points at
the Anthology, and reports any field that differs. Entries without an Anthology
URL are listed at the end for manual checking.

Usage:  python scripts/check_refs.py report/custom.bib
"""
import re
import sys
import urllib.request

FIELDS = ["title", "author", "booktitle", "journal", "volume", "year", "pages", "doi"]


def parse(text):
    """Return {key: {field: value}} for every @entry in text."""
    out = {}
    for block in re.split(r"\n(?=@)", text.strip()):
        head = re.match(r"@(\w+)\s*\{\s*([^,]+),", block)
        if not head:
            continue
        fields = {}
        for name, value in re.findall(r'(\w+)\s*=\s*"(.*?)"\s*,?\s*\n', block, re.S):
            fields[name.lower()] = re.sub(r"\s+", " ", value).strip()
        out[head.group(2).strip()] = fields
    return out


def main(path):
    local = parse(open(path, encoding="utf-8").read())
    manual, checked, problems = [], 0, 0

    for key, entry in sorted(local.items()):
        url = entry.get("url", "")
        if "aclanthology.org" not in url:
            manual.append(key)
            continue
        anth_id = url.rstrip("/").split("/")[-1]
        try:
            with urllib.request.urlopen(f"https://aclanthology.org/{anth_id}.bib") as fh:
                official = parse(fh.read().decode("utf-8"))
        except Exception as exc:
            print(f"[FETCH FAILED] {key}: {exc}")
            problems += 1
            continue
        if not official:
            print(f"[NOT FOUND] {key}: {anth_id} returned no entry")
            problems += 1
            continue
        theirs = next(iter(official.values()))
        diffs = []
        for field in FIELDS:
            mine, ref = entry.get(field), theirs.get(field)
            if mine is None and ref is None:
                continue
            if mine != ref:
                diffs.append((field, mine, ref))
        checked += 1
        if diffs:
            problems += 1
            print(f"\n[DIFFERS] {key}  ({anth_id})")
            for field, mine, ref in diffs:
                print(f"    {field}:")
                print(f"      yours : {mine}")
                print(f"      ACL   : {ref}")
        else:
            print(f"[OK] {key:34} {anth_id}")

    print(f"\n{checked} entries compared against the ACL Anthology, {problems} with problems.")
    if manual:
        print("\nNot on the Anthology, check these by hand:")
        for key in manual:
            print(f"  - {key}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/custom.bib")
