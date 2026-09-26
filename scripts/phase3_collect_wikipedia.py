"""Phase 3 collection: random English Wikipedia articles (CC BY-SA).
Reference = lead section, source = body (full text minus lead). Raw, untruncated; truncation happens in preprocessing.
Filters (project choice, to skip stubs): lead >= 40 words, body >= 400 words. Revision ids saved for reproducibility."""
import json, os, time, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data/raw/phase3/wikipedia_raw.jsonl')
API = 'https://en.wikipedia.org/w/api.php'
UA = {'User-Agent': 'COMP8240-student-project (mohammedgolam.kashem@students.mq.edu.au)'}
TARGET = 15
def get(**p):
    p.update(format='json', formatversion=2)
    req = urllib.request.Request(API + '?' + urllib.parse.urlencode(p), headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))
def page(title, intro):
    p = dict(action='query', prop='extracts|info', explaintext=1, titles=title, redirects=1)
    if intro: p['exintro'] = 1
    return get(**p)['query']['pages'][0]
kept, tried = [], 0
while len(kept) < TARGET and tried < 300:
    for r in get(action='query', list='random', rnnamespace=0, rnlimit=20)['query']['random']:
        tried += 1
        lead_p = page(r['title'], True); full_p = page(r['title'], False)
        lead, full = lead_p.get('extract', '').strip(), full_p.get('extract', '').strip()
        body = full[len(lead):].strip() if full.startswith(lead) else ''
        if len(lead.split()) >= 40 and len(body.split()) >= 400:
            kept.append({'id': 'wiki-%d' % full_p['pageid'], 'source_type': 'wikipedia', 'title': full_p['title'],
                         'revid': full_p.get('lastrevid'), 'url': 'https://en.wikipedia.org/?curid=%d' % full_p['pageid'],
                         'license': 'CC BY-SA 4.0', 'reference': lead, 'source': body})
            print('kept %d: %s (lead %d w, body %d w)' % (len(kept), full_p['title'], len(lead.split()), len(body.split())))
        if len(kept) >= TARGET: break
        time.sleep(0.5)
with open(OUT, 'w') as f:
    for k in kept: f.write(json.dumps(k) + '\n')
print('saved %d articles (checked %d) -> %s' % (len(kept), tried, OUT))
