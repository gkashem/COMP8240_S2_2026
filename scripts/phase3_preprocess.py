"""Phase 3 preprocessing: merge Wikipedia + arXiv raw docs; truncate each source to <= 656 words
(median length of the 100 CNN/DailyMail articles in RealSumm's src.txt, measured), cutting at a sentence
boundary (nltk punkt). References kept whole. Output: data/processed/phase3_docs.jsonl."""
import json, os
from nltk.tokenize import sent_tokenize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 656
docs = []
for name in ['wikipedia_raw.jsonl', 'arxiv_raw.jsonl']:
    docs += [json.loads(l) for l in open(os.path.join(ROOT, 'data/raw/phase3', name))]
with open(os.path.join(ROOT, 'data/processed/phase3_docs.jsonl'), 'w') as f:
    for d in docs:
        out, n = [], 0
        for s in sent_tokenize(d['source']):
            k = len(s.split())
            if n + k > LIMIT and out: break
            out.append(s); n += k
        d['source_words_raw'] = len(d['source'].split()); d['source'] = ' '.join(out)
        d['source_words'] = n; d['reference_words'] = len(d['reference'].split())
        f.write(json.dumps(d) + '\n')
        print('%-24s %-9s source %5d -> %3d w | reference %3d w' % (d['id'][:24], d['source_type'], d['source_words_raw'], n, d['reference_words']))
print('docs:', len(docs))
