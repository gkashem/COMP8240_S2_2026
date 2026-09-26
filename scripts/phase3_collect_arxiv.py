"""Phase 3 collection: arXiv papers under CC BY / CC BY-SA / CC0 only (redistributable).
Reference = abstract (OAI-PMH arXiv metadata), source = Introduction section (arxiv.org/html full text).
Project choices: cs.* primary category only; records from 2025-01-06..2025-01-10; abstract >= 40 words,
intro >= 400 words (same as Wikipedia); 3 s between requests (arXiv API policy). Raw, untruncated."""
import json, os, re, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data/raw/phase3/arxiv_raw.jsonl')
OAI = 'https://oaipmh.arxiv.org/oai'
UA = {'User-Agent': 'COMP8240-student-project (mohammedgolam.kashem@students.mq.edu.au)'}
NS = {'o': 'http://www.openarchives.org/OAI/2.0/', 'a': 'http://arxiv.org/OAI/arXiv/'}
OK = ('creativecommons.org/licenses/by/', 'creativecommons.org/licenses/by-sa/', 'creativecommons.org/publicdomain/zero/')
TARGET = 15
fetch = lambda url: urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
norm = lambda t: re.sub(r'\s+', ' ', t).strip()
def intro(aid):
    try: soup = BeautifulSoup(fetch('https://arxiv.org/html/' + aid), 'html.parser')
    except Exception: return ''
    for sec in soup.select('section.ltx_section'):
        h = sec.find(class_='ltx_title_section')
        if h and 'introduction' in h.get_text().lower():
            for m in sec.select('.ltx_Math, .ltx_note, .ltx_cite, .ltx_ref'): m.replace_with(' ')
            return norm(' '.join(p.get_text(' ') for p in sec.select('.ltx_para')))
    return ''
kept, seen = [], 0
params = {'verb': 'ListRecords', 'metadataPrefix': 'arXiv', 'from': '2025-01-06', 'until': '2025-01-10'}
while len(kept) < TARGET and params:
    root = ET.fromstring(fetch(OAI + '?' + urllib.parse.urlencode(params))); time.sleep(3)
    for rec in root.iter('{%s}record' % NS['o']):
        md = rec.find('.//a:arXiv', NS)
        if md is None: continue
        seen += 1
        lic = (md.findtext('a:license', '', NS) or '').strip(); cats = md.findtext('a:categories', '', NS).split()
        if not cats or not cats[0].startswith('cs.') or not lic.startswith(('http://', 'https://')) or not any(k in lic for k in OK): continue
        abstract = norm(md.findtext('a:abstract', '', NS)); aid = md.findtext('a:id', '', NS).strip()
        if len(abstract.split()) < 40: continue
        body = intro(aid); time.sleep(3)
        if len(body.split()) < 400: continue
        kept.append({'id': 'arxiv-' + aid, 'source_type': 'arxiv', 'title': norm(md.findtext('a:title', '', NS)),
                     'category': cats[0], 'url': 'https://arxiv.org/abs/' + aid, 'license': lic,
                     'reference': abstract, 'source': body})
        print('kept %d: %s [%s] (abstract %d w, intro %d w)' % (len(kept), aid, cats[0], len(abstract.split()), len(body.split())))
        if len(kept) >= TARGET: break
    tok = root.find('.//o:resumptionToken', NS)
    params = {'verb': 'ListRecords', 'resumptionToken': tok.text} if tok is not None and tok.text else None
with open(OUT, 'w') as f:
    for k in kept: f.write(json.dumps(k) + '\n')
print('saved %d papers (records seen %d) -> %s' % (len(kept), seen, OUT))
