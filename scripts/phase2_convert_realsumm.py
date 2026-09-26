"""Phase 2: convert RealSumm scores_dicts (abs.pkl + ext.pkl) to SummEval-style JSONL.
bart_out.txt appears in both files with identical summaries -> kept once (as abs).
Text cleaning: <t> </t> sentence tags removed, whitespace collapsed."""
import json, os, pickle, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data/external/REALSumm/scores_dicts')
OUT = os.path.join(ROOT, 'data/processed/realsumm.jsonl')
clean = lambda t: re.sub(r'\s+', ' ', re.sub(r'</?t>', ' ', t)).strip()
n, seen = 0, set()
with open(OUT, 'w') as f:
    for kind in ['abs', 'ext']:
        d = pickle.load(open(os.path.join(SRC, kind + '.pkl'), 'rb'))
        for k, doc in d.items():
            ref = doc['ref_summ']; refs = ref if isinstance(ref, list) else [ref]
            for sysname, s in doc['system_summaries'].items():
                if (k, sysname) in seen: continue
                seen.add((k, sysname))
                f.write(json.dumps({'id': str(doc['doc_id']), 'model_id': sysname.replace('.txt', ''), 'type': kind,
                    'decoded': clean(s['system_summary']), 'references': [clean(x) for x in refs],
                    'litepyramid_recall': s['scores']['litepyramid_recall'],
                    'realsumm_scores': s['scores']}, default=float) + '\n'); n += 1
print('records:', n, '| systems:', len({m for _, m in seen}), '| docs:', len({k for k, _ in seen}))
