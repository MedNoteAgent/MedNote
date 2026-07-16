# The ICD-10 ETL, Explained: From Raw XML to Embedding-Ready Text

A plain-language walkthrough of what happens between downloading two government
XML files and having ~47,000 documents ready for a medical embedding model.
No code — just the thinking behind each step.

---

## 1 · The Core Idea

The entire ETL exists to answer one question:

> **What exact text should the embedding model see for each ICD-10 code?**

An embedding model is like a reader with no memory and no context. It sees only
the words you hand it, converts them into a vector, and that vector is
everything the search engine will ever know about that code. If the text is
incomplete, the vector is incomplete — forever.

The raw CMS data is the *opposite* of what a context-free reader needs. It is a
deeply nested tree where meaning lives in **position**: a code's XML entry may
literally say just *"without complications"* — which only means something if
you know its parents were "Type 2 diabetes mellitus" under "Diabetes mellitus"
under "Endocrine diseases." The tree knows this; a leaf read alone does not.

So the ETL's job, in one sentence:

> **Flatten a relational tree into ~47,000 self-contained "index cards," each
> carrying its own ancestry, its own synonyms, and its own rules — so that any
> card read in isolation still tells the full story.**

That is exactly how the vector database will read them: in isolation.

### Why not just chunk the file like normal RAG?

Standard RAG advice — "split the document every 500 words" — would **destroy**
this data. A blind splitter would cut a code away from its own synonyms, glue
the tail of one disease family to the head of an unrelated one, and orphan
children from the parents that give them meaning. The unit of meaning in
ICD-10 is not "500 words"; it is **one code**. So the rule is absolute:

> **One code = one document. Never split, never merge.**

### Offline, once — but quality compounds

The ETL runs offline, once per year (ICD-10 updates every October). Nothing
about it is latency-sensitive. But every retrieval the system ever performs is
downstream of these documents, so care spent here compounds: a synonym missed
in ETL is a query that fails at runtime, every time, for every doctor.

---

## 2 · The Two Source Files, and Why We Need Both

CMS publishes two XML files (~9.7 MB each), and they serve opposite directions
of lookup:

| File | Metaphor | Direction | Language style |
|------|----------|-----------|----------------|
| **Tabular** | The dictionary | code → meaning | Formal clinical ("Acute myocardial infarction, unspecified") |
| **Index** | The reverse phonebook | term → code | How people actually talk ("Infarct… myocardium", "Enlargement, prostate") |

Here is the central tension of the whole project: **doctors speak Index
language, but we must retrieve Tabular codes.** A physician says "ear
infection"; the billable truth is "H66.90 Otitis media, unspecified,
unspecified ear." The Tabular file alone would leave a vocabulary gap between
how conditions are described in the exam room and how they are named in the
code book.

The Index file is the bridge — and it is a gift: a synonym dictionary built by
professional human coders over decades. We would be foolish to rely on the
embedding model's intuition for synonyms when the authoritative mapping ships
in the same download.

---

## 3 · The Five Steps

### Step 1 — Acquire the raw data

Download both XML files from CMS.gov and store them alongside the guidelines
corpus. This step is deliberately boring: verify the files exist, skip the
download if they do, and keep the refresh path (new files each October) a
one-line configuration change. Everything downstream assumes these two files
are the untouched, official source of truth.

### Step 2 — Walk the Tabular tree (the dictionary)

We descend the tree top-down — chapter → section → code → sub-code →
sub-sub-code — and at **every** code node we create one card. While walking,
three things are collected:

1. **Identity:** the code itself and its official description.
2. **Breadcrumbs:** the trail of ancestor descriptions accumulated on the way
   down. This becomes the card's *hierarchy path* — e.g. *"Diseases of the
   nervous system → Episodic and paroxysmal disorders → Other headache
   syndromes."* This single line is what makes a leaf self-contained: the
   context that lived in the tree's shape is now written on the card itself.
3. **The rulebook attached to each code:** its "includes" notes (official
   alternative names), "inclusion terms" (more alternative names), the
   excludes rules (conditions that must *not* be coded here), and sequencing
   notes ("code first…", "use additional code…").

We also record each card's **family links** — who its parent is, who its
children are. These links never enter the embedding text; they exist for a
later runtime feature (the specificity check, where an "unspecified" parent
code offers its more precise children to the physician).

Result: **46,881 cards.** (Worth knowing: the "~72,000 codes" figure often
quoted for ICD-10-CM counts billable permutations, not entries in this file.
We trust what we count, not what folklore says.)

### Step 3 — Harvest synonyms from the Index (the phonebook)

The Index nests terms inside terms: *Diabetes → with → amyotrophy → E11.44*.
We flatten every such path into a single natural-language phrase — "Diabetes,
diabetic, with, amyotrophy" — and file it under its code.

Then we merge: every card whose code appears in the phonebook gets those
phrases attached as **index synonyms**. Two guardrails apply:

- **A cap per code** (ten phrases), so a common condition with dozens of index
  entries doesn't drown its own description in synonyms.
- **No invention:** only phrases the human coders actually wrote. About 16,390
  of our 46,881 cards gain synonyms; the rest legitimately have no index entry
  and are passed through untouched.

**An honest limitation, discovered by testing:** the Index routes some famous
colloquialisms through "see …" cross-references that carry no code — "heart
attack" points to *see Infarct, myocardium* rather than to I21 directly. So
"heart attack" never literally lands on a card. This is *not* fixed in ETL; it
is fixed at runtime, where a fast LLM normalizes colloquial queries into
formal terms before searching. Knowing precisely where ETL's coverage ends is
what told us that runtime step was necessary.

### Step 4 — Tag demographic restrictions

Some codes are biologically impossible for some patients: pregnancy codes for
men, prostate codes for women, perinatal codes for adults. The ideology here:

> **Never trust similarity scoring to exclude the impossible. Filter it out
> deterministically, before scoring even happens.**

A pregnancy code might still score "similar" to a male patient's abdominal
symptoms — similarity is fuzzy by design. So each card gets hard metadata
tags, assigned by code prefix: the pregnancy chapter → female-only (1,791
cards), prostate codes → male-only (27 cards), the perinatal chapter →
newborns only (565 cards). At query time the retriever uses these as a hard
wall, not a soft preference.

Like the family links, these tags ride in the card's metadata — they are
filters, not prose, and never enter the embedding text.

### Step 5 — Export the hand-off artifact

All enriched cards are written to a single file, one JSON object per line
(~34.5 MB). This file is the **frozen boundary** between the ETL and the next
offline job (embedding + indexing). It can be inspected line by line,
spot-checked, diffed between annual releases, and re-indexed at will without
ever re-parsing the XML. If retrieval ever misbehaves, this file is where the
audit starts.

A principle enforced throughout the pipeline: **each step produces new data
rather than modifying what it received.** Parse produces cards; enrichment
produces enriched copies; tagging produces tagged copies. At any point you can
compare a card before and after a step — which is exactly what the tests do.

---

## 4 · The Final Product: Composing the Embedding Text

Every card knows how to render itself as the exact text the embedding model
will see. The composition is deliberate, line by line:

```
G44.2: Tension-type headache                          ← identity first
Hierarchy: Diseases of the nervous system → …         ← ancestry (context)
Also known as: tension headache NOS, stress headache… ← ALL synonym sources fused
Excludes: headache NOS (R51.9), …                     ← capped negative signal
```

The reasoning behind each choice:

- **Code + description leads** because it is the primary thing we want queries
  to match against.
- **The hierarchy line** injects the tree's context into the flat text — the
  payoff of Step 2's breadcrumb collection.
- **"Also known as" fuses three synonym streams** — the Tabular's "includes"
  notes, its "inclusion terms," and the Index phrases — into one line. This is
  the recall engine: it is what lets informal phrasings land on formal codes.
- **Excludes are capped** (first five only). A little negative signal helps
  disambiguate sibling conditions; a wall of exclusions would drown the card's
  own identity.
- **Deliberately absent:** parent/child links, sex/age tags, sequencing rules.
  These are *machinery* — for filtering and for the specificity check — not
  *meaning*. Putting them in the prose would only add noise to the vector.

---

## 5 · The Principles, in Summary

1. **One code = one document.** The unit of meaning dictates the unit of
   storage; generic chunking is data destruction here.
2. **Self-containment.** Every card must survive being read with no context,
   because that is the only way it will ever be read.
3. **Flatten the tree into the text.** Ancestry becomes a hierarchy line;
   position becomes prose.
4. **Use the human-built synonym dictionary.** Never rely on model intuition
   for what expert coders already wrote down.
5. **Hard-filter the impossible; similarity-rank the plausible.** Demographics
   are walls, not weights.
6. **Meaning goes in the text; machinery goes in the metadata.**
7. **Freeze the hand-off.** A line-per-record artifact between pipeline stages
   makes everything inspectable, auditable, and re-runnable.
8. **Trust measured numbers over quoted ones** — and document honestly where
   coverage ends (the "heart attack" gap), because that is what tells the
   runtime design what it must compensate for.
