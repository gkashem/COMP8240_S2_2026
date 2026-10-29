"""Phase 4: LLM judge (Qwen3-4B Q4_K_M, llama.cpp) scores SummEval summaries 1-5.
One call per dimension; SummEval definitions; short reasoning then score as JSON; resumable."""
import argparse, json, os, random, time
from llama_cpp import Llama

DEFS = {
    'coherence': 'Coherence - the collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic.',
    'consistency': 'Consistency - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Penalize summaries that contain hallucinated facts.',
    'fluency': 'Fluency - the quality of individual sentences. Sentences in the summary should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read.',
    'relevance': 'Relevance - selection of important content from the source. The summary should include only important information from the source document. Penalize summaries which contain redundancies and excess information.',
}
SYSTEM = 'You will be given a news article and a summary of it. Rate the summary on one quality dimension, from 1 (worst) to 5 (best).'
SCHEMA = {'type': 'json_object', 'schema': {
    'type': 'object',
    'properties': {'reasoning': {'type': 'string'}, 'score': {'type': 'integer', 'enum': [1, 2, 3, 4, 5]}},
    'required': ['reasoning', 'score']}}

def prompt(src, summ, dim):
    return (f'Source article:\n{src}\n\nSummary:\n{summ}\n\n'
            f'Evaluation criterion:\n{DEFS[dim]}\n\n'
            f'Note: the summary has been lowercased and tokenized; do not penalize casing or spacing around punctuation. Judge only {dim}; ignore all other qualities.\n\n'
            f'First explain your judgement of the summary\'s {dim} in one short sentence (at most 25 words), '
            f'then give a score from 1 to 5. Respond in JSON: {{"reasoning": "...", "score": <1-5>}}')

SYSTEM3 = 'You will be given a document and a summary of it. Rate the summary on one quality dimension, from 1 (worst) to 5 (best).'

def prompt3(src, summ, dim):  # Phase 3 corpus: summaries not lowercased; sources not news
    return (f'Source document:\n{src}\n\nSummary:\n{summ}\n\n'
            f'Evaluation criterion:\n{DEFS[dim]}\n\n'
            f'Judge only {dim}; ignore all other qualities.\n\n'
            f'First explain your judgement of the summary\'s {dim} in one short sentence (at most 25 words), '
            f'then give a score from 1 to 5. Respond in JSON: {{"reasoning": "...", "score": <1-5>}}')

ap = argparse.ArgumentParser()
ap.add_argument('--data', choices=['summeval', 'phase3'], default='summeval')
ap.add_argument('--docs', type=int, default=50)
ap.add_argument('--per_doc', type=int, default=16)
ap.add_argument('--repeats', type=int, default=1)
ap.add_argument('--temp', type=float, default=0.2)
ap.add_argument('--seed', type=int, default=8240)
ap.add_argument('--out', required=True)
ap.add_argument('--model', default='models/Qwen3-4B-Q4_K_M.gguf')
ap.add_argument('--think', action='store_true')
ap.add_argument('--max_tokens', type=int, default=80)
ap.add_argument('--retry_failed', action='store_true', help='only re-run calls that have no valid score yet')
args = ap.parse_args()

if args.data == 'summeval':
    recs = [json.loads(l) for l in open('data/raw/model_annotations.aligned.jsonl')]
    src = {json.loads(l)['id']: json.loads(l)['text'] for l in open('data/processed/summeval_sources.jsonl')}
else:
    recs = [json.loads(l) for l in open('data/processed/phase3_summaries.jsonl')]
    src = {json.loads(l)['id']: json.loads(l)['source'] for l in open('data/processed/phase3_docs.jsonl')}
by_doc = {}
for r in recs:
    by_doc.setdefault(r['id'], []).append(r)
rng = random.Random(args.seed)
docs = sorted(by_doc); rng.shuffle(docs); docs = docs[:args.docs]

done, seen = set(), set()
if os.path.exists(args.out):
    for l in open(args.out):
        x = json.loads(l); key = (x['id'], x['model_id'], x['dim'], x['run']); seen.add(key)
        if x['score'] is not None or not args.retry_failed: done.add(key)
todo = seen - done if args.retry_failed else None
if args.retry_failed: print('to retry:', len(todo), flush=True)
print('already done:', len(done), flush=True)

llm = Llama(model_path=args.model, n_ctx=2048, n_threads=2, seed=args.seed, verbose=False)
t0 = time.time(); n = 0
with open(args.out, 'a') as out:
    for d in docs:
        items = sorted(by_doc[d], key=lambda r: r['model_id']); rng.shuffle(items)
        for r in items[:args.per_doc]:
            dims = list(DEFS); rng.shuffle(dims)
            for dim in dims:
                for k in range(args.repeats):
                    if (d, r['model_id'], dim, k) in done:
                        continue
                    if todo is not None and (d, r['model_id'], dim, k) not in todo:
                        continue
                    t = time.time()
                    res = llm.create_chat_completion(
                        messages=[{'role': 'system', 'content': SYSTEM if args.data == 'summeval' else SYSTEM3},
                                  {'role': 'user', 'content': (prompt if args.data == 'summeval' else prompt3)(src[d], r['decoded'], dim) + ('' if args.think else ' /no_think')}],
                        response_format=SCHEMA, max_tokens=args.max_tokens, temperature=args.temp, seed=args.seed + k)
                    txt = res['choices'][0]['message']['content']
                    try:
                        j = json.loads(txt); score, reason = int(j['score']), j['reasoning']
                    except Exception:
                        score, reason = None, None
                    sec = time.time() - t
                    expert = sum(a[dim] for a in r['expert_annotations']) / 3 if 'expert_annotations' in r else float('nan')
                    rec = {'id': d, 'model_id': r['model_id'], 'dim': dim, 'run': k, 'score': score,
                           'reasoning': reason, 'raw': txt, 'prompt_tokens': res['usage']['prompt_tokens'],
                           'completion_tokens': res['usage']['completion_tokens'], 'max_tokens': args.max_tokens, 'sec': round(sec, 1)}
                    out.write(json.dumps(rec) + '\n'); out.flush(); n += 1
                    print(d[:16], r['model_id'], dim, k, score, f'expert={expert:.2f}',
                          f"gen={rec['completion_tokens']}", f'{sec:.1f}s', flush=True)
print(f'new calls {n}, total {time.time() - t0:.0f}s')
