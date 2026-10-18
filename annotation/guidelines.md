# Annotation guidelines (Phase 3)

Rate each summary 1-5 on four dimensions. Read the source first.
Definitions follow SummEval (Fabbri et al., 2021, Sec. 4.1).

## Coherence - the collective quality of all sentences
"The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic."
- 1: disconnected sentences, no clear topic
- 3: understandable but jumpy or poorly ordered
- 5: flows logically as one connected summary

## Consistency - factual alignment with the source
"A factually consistent summary contains only statements that are entailed by the source document." Penalise hallucinated facts.
- 1: major false or invented claims
- 3: mostly correct, one or two unsupported details
- 5: every statement supported by the source

## Fluency - the quality of individual sentences
Sentences "should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read."
- 1: many fragments or errors, hard to read
- 3: readable with some errors or a cut-off sentence
- 5: grammatical and clean throughout

## Relevance - selection of important content
"The summary should include only important information from the source document." Penalise redundancy and excess detail.
- 1: misses the main point
- 3: some key points, some trivial ones
- 5: captures the key points only

## Procedure
- Rate one document group at a time; read the source once per group.
- Do not look at annotation/key.csv until all ratings are done.
- Integers only (1-5). Rate each dimension independently.
