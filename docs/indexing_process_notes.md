# The Indexing Job, Explained: From Index Cards to a Searchable Medical Memory

A plain-language walkthrough of what happens between having ~47,000
embedding-ready "index cards" (the ETL's output) and having a database that
can answer "tension headache" with G44.2 in milliseconds. No code — just the
thinking behind each step.

---

## 1 · The Core Idea

The ETL ended with a question deliberately left open:

> **The cards are written. Now — how will they be found?**

Indexing is the act of reading every card once, *very carefully*, and filing
what was understood in a way that makes future lookups instant. The reader is
an embedding model: it converts each card's text into a vector — a point in a
high-dimensional space where **distance means meaning**. Cards about heart
disease cluster together; cards about ear infections cluster somewhere else.
A query becomes a point in the same space, and retrieval is simply: *which
cards are nearest?*

This happens **offline, once**. Every millisecond spent here is spent so that
no doctor ever waits on it. The runtime system inherits a finished, searchable
memory — it never reads the XML, never re-embeds a card, never even opens the
JSONL. If the ETL's principle was "one code = one document," indexing's is:

> **Read each document once, understand it two different ways, and never read
> it again.**

Why *two* ways? Because meaning and spelling are different kinds of matching —
and medicine needs both.

---

## 2 · Two Readers, Because Language Fails in Two Ways

Every card is indexed twice, by two readers with opposite talents.

| Reader | What it produces | What it's good at | What it's blind to |
|--------|------------------|-------------------|--------------------|
| **The semantic reader** (a medical language model) | A dense vector — one point in meaning-space | "heart attack" ≈ "acute myocardial infarction" | Exact strings: acronyms, codes |
| **The literal reader** (a term-frequency scorer) | A sparse vector — a weighted bag of exact words | "COPD" matches *COPD*, exactly, always | Paraphrase: "can't breathe" ≠ "dyspnea" |

### Why the semantic reader must be a *medical* one

A generic embedding model knows English; it does not know that "infarct,
myocardium" and "heart attack" are the same event. The model we use was
trained specifically on pairs of medical synonyms drawn from the giant UMLS
ontology — its entire education was "these two phrases name one concept."
That is precisely the vocabulary gap the ETL identified between exam-room
language and code-book language. Choosing a generic model here would quietly
undo the Index file's synonym harvest.

### Why the literal reader exists at all

Semantic similarity is fuzzy by design — that is its job. But medicine is full
of tokens where fuzziness is a bug: **COPD**, **STEMI**, **E11.9**. A doctor
who types an acronym means *that acronym*. The literal reader scores documents
the way a classic search engine does: rare words count heavily ("COPD" appears
on few cards), common words barely count ("disease" appears on thousands).
One clever division of labor: the document side records how often each term
appears, while the database itself supplies the "how rare is this term across
all cards" half of the formula at query time — so rarity stays honest even as
documents are added.

Neither reader replaces the other. At runtime they will vote together —
weighted about 70% semantic, 30% literal — but that blending is the
retriever's story. Indexing's job is only to ensure **both readings of every
card are on file**.

---

## 3 · One Filing Cabinet, Two Kinds of Cards

The knowledge base is not only ICD-10 codes. The ETL's final step assembled a
second corpus: clinical documentation guidelines — SOAP-note conventions,
red-flag escalation rules, coding-specificity conventions. These answer
questions no code card can ("how do I escalate chest pain with arm
radiation?").

Both kinds go into the **same** filing cabinet, each card stamped with its
kind: *code* or *guideline*. One cabinet, not two, because some physician
queries are code lookups, some are guidance lookups, and some are genuinely
both — the retriever can filter to one kind or search across both with a
single query. Two separate databases would force that decision at build time;
the stamp defers it to query time, where the context to decide actually
exists.

The guidelines are chunked by **section heading**, not by word count — the
same anti-blind-chunking ideology as the ETL. Each section was *written* to be
self-contained (that was a requirement of the corpus, not an accident), so a
section is the natural unit of retrieval. Six sections, six cards.

### What rides along with each card

The vector is what gets *found*; the payload is what gets *used*. Alongside
both vectors, every card carries its full metadata: the code, description,
hierarchy path, family links, and — critically — the demographic tags from
ETL Step 4. One normalization happens at filing time: cards with *no*
restriction are stamped "applies to all" explicitly, rather than left blank.
The reason is subtle and worth stating: the runtime filter asks "is this card
valid for *this* patient?", and a blank answers that question with silence.
Silence gets filtered out. An explicit "all" never does. **The filing format
anticipates the question that will be asked of it.**

---

## 4 · The Machinery Choices, and Their Reasons

### The database runs inside the process

No Docker, no server, nothing to start or stop. The vector database runs
embedded in the indexing script itself and persists straight to a local
folder. For a single-machine prototype this removes an entire category of
failure (is the server up? right port? right version?) at the cost of one
constraint worth respecting: **only one process may touch that folder at a
time**. Build, then validate, then run the app — sequentially, never
concurrently.

### Every card gets a deterministic identity

Each card's database ID is derived from its content identity — the ICD code,
or the guideline section name — not from a counter or a random draw. The
payoff is **idempotency**: filing the same card twice *overwrites* rather than
*duplicates*. Re-running the indexer is always safe. This one property is what
makes everything below possible.

### Work is saved as it happens — a lesson paid for, not predicted

The first version of this job read all 47,000 cards, embedded all of them
(three-quarters of an hour of continuous computation), and only *then* began
filing. It was killed at the 45-minute mark. Everything was lost — the
database was still empty, because filing hadn't started.

The rebuilt version files each small batch the moment it is embedded. An
interruption now loses at most one batch, and — thanks to deterministic IDs —
a restart can ask the cabinet "which cards do you already have?" and skip
straight to the missing ones. The principle, learned the honest way:

> **In any long job, the distance between "work done" and "work saved" is the
> amount you will lose. Keep it one batch wide.**

(A companion rule surfaced by the same incident: a resumed run must only skip
re-embedding if *nothing about the reading changed*. Swap the embedding model
and the old vectors are stale — that demands a fresh cabinet, not a resume.)

---

## 5 · Validation: Interrogate What You Built

The build ends with numbers ("46,887 cards filed") — but counts only prove
the cabinet is full, not that it *works*. So the final step asks it real
questions, one per failure mode it was designed to prevent:

- **A semantic question** — does formal clinical language land on the right
  code, at the top?
- **A literal question** — does a bare acronym ("COPD") snap to its exact
  code?
- **A guideline question** — does an escalation query surface the red-flag
  section, not a code?

### An honest finding: validate the question that will actually be asked

The plan's original acceptance check was: dense search for "heart attack"
must put I21.9 in the top three. Built and measured, it doesn't — I21.9 lands
fourth, behind its own parent and two angina codes. Investigating *why*
led back to a limitation the ETL had already documented: no card contains the
literal phrase "heart attack," because the official Index routes that
colloquialism through a code-less cross-reference. The system's actual design
compensates at runtime — a fast LLM normalizes colloquial queries into formal
terms *before* the search. Query the index with what the retriever will
really receive — "acute myocardial infarction" — and I21.9 is **rank one**.

So the validation was corrected, not the index: the colloquial query must
land the right code *family* in the top three (it does — the semantic reader
alone gets remarkably close), and the normalized query must hit I21.9
exactly. The lesson generalizes:

> **Validate components with the inputs they will actually receive in the
> system — not with inputs a different component exists to transform.**
> Testing the raw colloquialism against the bare index was really testing the
> absence of the normalizer.

---

## 6 · The Principles, in Summary

1. **Pay the cost once, offline.** Every embedding computed here is a
   millisecond no physician waits for later.
2. **Index every document two ways.** Semantic and literal matching fail in
   opposite directions; medicine requires both readers on file.
3. **The semantic reader must speak the domain's language.** A generic model
   would silently discard the synonym bridge the ETL built.
4. **One cabinet, stamped kinds.** Filter at query time, when context exists —
   not at build time, when it doesn't.
5. **The filing format anticipates the question.** "No restriction" is stored
   as an explicit "all," because blanks lose arguments with filters.
6. **Deterministic identity makes re-runs safe.** Same card, same ID,
   overwrite not duplicate.
7. **Save work as it happens.** The gap between computed and persisted is
   exactly what an interruption costs — keep it one batch wide.
8. **Interrogate the artifact, then trust it.** One validation question per
   failure mode the design claims to prevent.
9. **Validate with real inputs.** Test each component with what the system
   will actually hand it — and when a check fails, first ask whether the
   check, not the artifact, is wrong.
