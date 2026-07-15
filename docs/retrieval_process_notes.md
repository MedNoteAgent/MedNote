# The Retrieval Pipeline, Explained: From a Doctor's Sentence to a Billable Code

A plain-language walkthrough of Task 7 — the runtime half of the RAG system. The
ETL (Task 5) wrote the index cards; the indexing job (Task 6) filed them in a
searchable cabinet. This is the part that runs **every time a physician asks**,
and it has to be right in under a second. No code — just the thinking, including
two design changes that measurement forced on us.

---

## 1 · The Core Idea

Everything before this was offline preparation. Task 7 is the payoff query:

> **Given one sentence of clinical assessment, return at most three ICD-10
> codes — each with a confidence score, a citation, and its more-specific
> children — or honestly return nothing.**

The pipeline is a relay of five specialists, each correcting for the weakness
of the one before it:

| Stage | Specialist | Corrects for |
|-------|-----------|--------------|
| Extract | A fast LLM | Doctors don't speak code-book language |
| Retrieve | The hybrid index (Task 6) | The LLM can't search 47,000 cards |
| Re-rank | A cross-encoder | Retrieval has recall but weak precision |
| Gate | A confidence threshold | Everything upstream can be confidently wrong |
| Expand | The specificity checker | Insurers want the *most specific* code |

And one accountant — a cache — because clinic days repeat themselves.

---

## 2 · Never Search with the Transcript

The single most important rule of the runtime design:

> **Do not pass the transcript to the vector database.**

Transcripts are full of noise — greetings, scheduling, the patient's story
about their cat. Worse, they are written in exam-room language, and the index
is written in code-book language. The ETL documented precisely where that gap
is unbridgeable by data alone: "heart attack" appears on *no card*, because
the official Index routes it through a code-less cross-reference.

So the first stage is a **fast, cheap LLM** with one narrow job: read the
assessment and emit a short list of *normalized clinical entities* — "kid has
an ear infection in both ears" becomes *Acute bilateral otitis media*. This is
the bridge the ETL told us we would need. Each entity then becomes one precise,
formal query.

If the LLM's reply can't be parsed, the pipeline does not crash and does not
silently drop the request — it degrades to querying with the raw assessment
text and logs the fallback. A worse query beats no answer.

---

## 3 · Retrieval: Two Readers Vote, Walls Filter First

Each entity is searched two ways against the Task 6 index — the semantic
reader (dense vectors) and the literal reader (BM25 sparse) — and their votes
are blended 70/30 in favor of meaning. One technical subtlety worth knowing:
the two readers score on **incompatible scales** (cosine lives in 0–1; BM25 is
unbounded), so each list is normalized to 0–1 *within itself* before the
weighted blend. Without that, the sparse reader would shout over the dense one
whenever it matched anything at all.

Before either reader scores a single card, the **demographic walls** go up:
codes marked female-only vanish for a male patient; perinatal codes (28-day
cap) vanish for anyone older. Two principles inherited from the ETL apply:

- **Walls, not weights.** A pregnancy code must never be merely *down-ranked*
  for a male patient; it must be unreachable.
- **Unknown never excludes.** If the patient's sex or age isn't known, the
  corresponding wall simply isn't built. Filtering on information you don't
  have is how you hide the right answer.

---

## 4 · Re-ranking: A Lesson in Asking the Right Question

Retrieval is a wide net — the right code is almost always *somewhere* in the
top 15, rarely at rank 1. A cross-encoder re-reads the query and each candidate
*together* and produces a much sharper score, squashed to 0–1 so it can face
the confidence threshold.

### The measured lesson (this is the big recent change)

The plan said: re-rank against "the specific transcript text." Built and
measured, that instruction quietly breaks the system:

- Re-ranking against the raw phrase **"heart attack"** scored the correct code
  I21.9 at **0.004** — a guaranteed false zero-hit. The earlier notebook
  prototype hit this same wall and blamed the cross-encoder model, proposing
  an LLM re-ranker instead.
- Re-ranking the *same candidates* against the **normalized entity** ("Acute
  myocardial infarction") scored I21.9 at **1.0, rank one**. Same model, same
  candidates. The model was never the problem — the *question* was.

A second, subtler trap followed: joining several entities into one re-rank
query ("Shortness of breath; Chest tightness") diluted every candidate's score
— R06.02, whose description *is literally* "Shortness of breath," fell to
0.425, below the gate. So the final design scores **each candidate against the
single entity that retrieved it**, then merges across entities keeping each
code's best confidence.

> **Score candidates against the question that found them.** The entity
> extractor exists to close a vocabulary gap — handing the re-ranker raw
> colloquial text (or a mixed bag of entities) re-opens that gap at the last
> step, after all the work of closing it.

With that change, the plan's six-query acceptance suite went from 4/6 to
**6/6**, with confidences between 0.92 and 1.0 — keeping the small, local,
free cross-encoder and needing no extra LLM call in the hot path.

---

## 5 · The Gate: Refuse to Guess

If the best confidence is below the threshold (0.7, tunable in config), the
pipeline returns **nothing**, and the caller shows:

> *"Insufficient data to suggest an accurate ICD-10 code. Please manually
> assign in EHR."*

In a medical-coding tool, a confidently wrong suggestion is worse than an
honest refusal — a physician who catches one bad code stops trusting all the
good ones. The gate is also why re-rank confidence quality (Section 4)
mattered so much: badly-asked questions produce low scores, and low scores
turn into false refusals.

---

## 6 · Specificity: The Family Links Pay Off

Insurers want the most specific code that the documentation supports. If a
surviving code is an "unspecified" parent (H65.9, *Unspecified otitis media*),
the pipeline attaches its children — right ear / left ear / bilateral — as
options for the physician to choose from.

Two details make this trustworthy:

- The children come from the **family links the ETL collected and deliberately
  kept out of the embedding text** — machinery, not meaning, now doing its job.
- Children are **fetched from the index by exact code, never generated**. A
  child that isn't in the knowledge base simply doesn't appear. Nothing at
  this stage can hallucinate.

Every suggestion leaves the pipeline typed and stamped: a confidence, a source
citation ("ICD-10-CM 2026"), and `pending_confirmation: true` — nothing is a
diagnosis until a physician signs it.

---

## 7 · The Cache, and Why Its Key Is Fatter Than the Plan's

A clinic day repeats itself — "tension headache" comes up four times before
lunch. An LRU cache makes the second lookup a dictionary hit (<5 ms) instead
of an embed-search-rerank round trip (~300–600 ms).

One correctness detail, a quiet change from the plan's sketch: the cache key
is **entity + patient sex + patient age**, not the entity alone. The
demographic walls change what retrieval returns — caching "delivery" for a
female patient and serving that result to a male patient would smuggle a
pregnancy code straight past the wall. **A cache key must contain everything
that changes the answer.**

---

## 8 · What Changed From the Plan, and Why

Three deliberate deviations, each with its reason on record:

1. **Re-rank per entity, not against the transcript** (Section 4). Measured:
   4/6 → 6/6 on the acceptance suite. The plan's wording predates contact with
   the real cross-encoder's behavior on colloquial text.
2. **Demographics arrive as parameters, not an EHR lookup.** The plan's Step
   7.1 has the pipeline read age/sex from the mock EHR — but the mock EHR is a
   later task, and the dependency graph says Task 7 needs only Task 6. The
   pipeline takes `patient_sex` / `patient_age` as inputs; the graph node that
   *does* know the EHR supplies them later. Looser coupling, buildable today.
3. **The typed result schema was created early.** `SuggestedCode` nominally
   belongs to the next task, but it is this pipeline's return type — so it was
   lifted verbatim from the plan's architecture section now. Field names
   mirror the ETL dataclass exactly, so no naming drift can creep in between
   offline and runtime halves.

One non-deviation worth stating: the notebook prototype had concluded the
cross-encoder should be replaced with an LLM re-ranker. The shipped code
**keeps the cross-encoder** — Section 4's finding showed the model was fine
once the query side was fixed, and a local model in the hot path is faster,
free, and quota-immune.

---

## 9 · The Principles, in Summary

1. **Never search with the transcript.** Extract and normalize first; noise in,
   noise out.
2. **Degrade, don't die.** A failed extraction falls back to the raw text with
   a logged warning — worse query beats no answer.
3. **Normalize before you blend.** Two scoring scales must be brought to the
   same range before weights mean anything.
4. **Walls before scores; unknown never excludes.** Demographic filters are
   deterministic and pre-scoring, and absent information builds no wall.
5. **Score candidates against the question that found them.** Don't re-open a
   vocabulary gap in the last step that the first step existed to close.
6. **Refuse to guess.** Below the threshold, return nothing and say so.
7. **Expand, never invent.** Specificity options are fetched by exact code
   from the knowledge base or they don't exist.
8. **A cache key contains everything that changes the answer** — including the
   patient.
9. **Blame the question before the model.** Two "model failures" in this task
   (the notebook's cross-encoder verdict, the joined-entity dilution) were
   both mis-asked questions; fixing the input beat swapping the component.
