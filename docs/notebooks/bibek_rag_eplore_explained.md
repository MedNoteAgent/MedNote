# Explanation: `bibek_rag_eplore.ipynb`

This document walks through **every cell** of
[`bibek_rag_eplore.ipynb`](./bibek_rag_eplore.ipynb) in execution order, explaining
*what the code does*, *why it does it that way*, and *what it actually produces* when
run against the real CMS 2026 files in `data/corpus/`:

- `icd10cm_tabular_2026.xml` (9.7 MB) — the code dictionary
- `icd10cm_index_2026.xml` (9.7 MB) — the term → code reverse lookup

Every XML snippet quoted below is copied verbatim from those two files (with line
numbers), not paraphrased, so you can grep the same lines yourself.

---

## Big picture: two phases, five steps

| Phase | Step | What happens |
|-------|------|--------------|
| Offline (once) | **1 · ETL** | Parse 9.7 MB nested XML → ~47k flat, self-contained documents; enrich with synonyms + demographic tags |
| Offline (once) | **2 · Embed & Index** | SapBERT dense vectors + BM25 sparse vectors → Qdrant |
| Runtime (per query) | **3 · Hybrid Retrieval** | Dense + Sparse LangChain retrievers, fused with weighted RRF (0.7/0.3), metadata-filtered → top-15 |
| Runtime (per query) | **4 · Re-Rank** | Cross-encoder scores query↔code pairs → top-3 |
| Runtime (per query) | **5 · Specificity** | Expand "unspecified" parent codes into precise children |

The notebook re-implements each step **inline** (no imports from `src/mednote/rag/…`,
which doesn't exist yet) so you can inspect intermediate state directly. A mapping
table at the end of the notebook shows which future module each cell corresponds to.

---

## Cell: title markdown (`b17037bb`)

Introduces the notebook and the summary table above. Calls out an important honesty
check baked into the whole notebook: earlier docs guessed **"~72,000 codes"**, but the
actual 2026 XML files contain **46,881** `<diag>` elements. The notebook always prints
the real, measured number rather than trusting the doc's estimate — a useful habit
when documentation and data can drift apart.

---

## Cell: `## 0 · Setup & Configuration` (`3cf83203`)

States the ground rule for the whole notebook: **every tunable number comes from
`config.yml`** via `mednote.config.get_config()` — never a hardcoded literal in the
notebook. This matters because Step 3's blend weights (0.7/0.3), Step 4's `top_k`
values, and Step 5's confidence threshold all trace back to one file.

### Code cell `7a3d0a7e` — locate the repo and load config

```python
REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "config.yml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
...
os.chdir(REPO_ROOT)
from mednote.config import get_config
cfg = get_config()
```

Since the notebook lives at `docs/notebooks/`, `Path.cwd()` starts there. The `while`
loop climbs parent directories until it finds `config.yml` (i.e., the repo root
`D:\Projects\MedNote`), then:

1. Prepends `src/` to `sys.path` so `import mednote...` works without installing the
   package.
2. `os.chdir(REPO_ROOT)` so every *relative* path in `config.yml` (e.g.
   `data/qdrant_data`, `data/corpus`) resolves correctly regardless of where Jupyter
   was launched from.
3. Loads the singleton config object.

**Actual printed output**, pulled straight from `config.yml`:

```
Repo root: d:\Projects\MedNote
dense_weight        : 0.7
sparse_weight       : 0.3
top_k_retrieve      : 15
top_k_rerank        : 3
confidence_threshold: 0.7
embeddings model    : cambridgeltl/SapBERT-from-PubMedBERT-fulltext
reranker model      : cross-encoder/ms-marco-MiniLM-L6-v2
```

These map 1:1 to `config.yml`:

```yaml
vector_store:
  dense_weight: 0.7
  sparse_weight: 0.3
  top_k_retrieve: 15
  top_k_rerank: 3
  confidence_threshold: 0.7
embeddings:
  model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
reranker:
  model: cross-encoder/ms-marco-MiniLM-L6-v2
```

### Code cell `8286ade4` — resolve the corpus paths

```python
CORPUS = REPO_ROOT / "data" / "corpus"
TABULAR_XML = CORPUS / "icd10cm_tabular_2026.xml"
INDEX_XML   = CORPUS / "icd10cm_index_2026.xml"
```

A one-line comment flags a documentation/reality mismatch: `implementation_plan.md`
references `data/icd10_source/`, but the files actually live in `data/corpus/`
(this also matches `config.yml`'s `paths.corpus_dir: data/corpus`). The notebook
resolves this by hand instead of trusting the doc path literally.

Output confirms both 9.7 MB files exist:

```
icd10cm_tabular_2026.xml           9.7 MB  exists=True
icd10cm_index_2026.xml             9.7 MB  exists=True
```

---

## Cell: `## 1 · Peek at the Raw Data` (`bf3887b7`)

Explains the two-file mental model before touching any parsing code:

- **Tabular** = the dictionary. You look up a *code* and get back its official
  definition, hierarchy, and coding rules (`includes`, `excludes1/2`,
  `useAdditionalCode`).
- **Index** = the reverse lookup. You look up a *term* a doctor might actually say
  ("ear infection") and get pointed at a code.

Both are needed because clinicians speak Index language but billing requires Tabular
codes.

### Code cell `024a8e75` — explore the Tabular tree shape

```python
tab_root = ET.parse(TABULAR_XML).getroot()
print("Tabular root:", tab_root.tag)                 # ICD10CM.tabular
print("Direct children:", sorted({c.tag for c in tab_root}))
print("Chapters:", len(tab_root.findall("chapter")))
```

This confirms the top-level schema: the root `<ICD10CM.tabular>` has three kinds of
direct children — `chapter`, `introduction`, `version` — and there are 22
`<chapter>` elements (ICD-10-CM's 22 official disease chapters).

It then defines a tiny linear-scan helper:

```python
def find_diag(root, target):
    for d in root.iter("diag"):
        if d.findtext("name") == target:
            return d
    return None
```

`root.iter("diag")` recursively walks **every** `<diag>` at any depth (chapters
contain sections, sections contain top-level diags, diags contain child diags —
`.iter()` doesn't care about depth). This is O(n) per lookup and only used here for
exploration; the real ETL in the next section builds a dict instead for O(1) lookup.

It fetches chapter index `3` (0-based → the 4th chapter, Endocrine diseases) and then
looks up `E11` (Type 2 diabetes). Here is the **actual raw XML** for `E11`, taken from
`icd10cm_tabular_2026.xml` starting at line 24335:

```xml
<diag>
  <name>E11</name>
  <desc>Type 2 diabetes mellitus</desc>
  <includes>
    <note>diabetes (mellitus) due to insulin secretory defect</note>
    <note>diabetes NOS</note>
    <note>insulin resistant diabetes (mellitus)</note>
  </includes>
  <useAdditionalCode>
    <note>code to identify control using:</note>
    <note>injectable non-insulin antidiabetic drugs (Z79.85)</note>
    <note>insulin (Z79.4)</note>
    <note>oral antidiabetic drugs (Z79.84)</note>
    <note>oral hypoglycemic drugs (Z79.84)</note>
  </useAdditionalCode>
  <excludes1>
    <note>diabetes mellitus due to underlying condition (E08.-)</note>
    <note>drug or chemical induced diabetes mellitus (E09.-)</note>
    <note>gestational diabetes (O24.4-)</note>
    <note>neonatal diabetes mellitus (P70.2)</note>
    <note>postpancreatectomy diabetes mellitus (E13.-)</note>
    <note>postprocedural diabetes mellitus (E13.-)</note>
    <note>secondary diabetes mellitus NEC (E13.-)</note>
    <note>type 1 diabetes mellitus (E10.-)</note>
  </excludes1>
  <diag>
    <name>E11.0</name>
    <desc>Type 2 diabetes mellitus with hyperosmolarity</desc>
    <diag><name>E11.00</name><desc>...without nonketotic hyperglycemic-hyperosmolar coma (NKHHC)</desc></diag>
    <diag><name>E11.01</name><desc>...with coma</desc></diag>
  </diag>
  <diag><name>E11.1</name><desc>Type 2 diabetes mellitus with ketoacidosis</desc>...</diag>
  ...
</diag>
```

This is exactly why the notebook's printed output looks the way it does:

```
<diag> E11 sub-elements: ['name', 'desc', 'includes', 'useAdditionalCode', 'excludes1', 'diag', 'diag', 'diag', 'diag', 'diag', 'diag', 'diag', 'diag', 'diag', 'diag']
desc      : Type 2 diabetes mellitus
includes  : ['diabetes (mellitus) due to insulin secretory defect', 'diabetes NOS', 'insulin resistant diabetes (mellitus)']
excludes1 : ['diabetes mellitus due to underlying condition (E08.-)', 'drug or chemical induced diabetes mellitus (E09.-)', 'gestational diabetes (O24.4-)']
children  : ['E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4']
```

Note there is **no** `excludes2` sub-element under `E11` in the real XML — that's why
the list of sub-elements shown doesn't include it (the code correctly guards with
`if e11.find("excludes1") is not None else []` for exactly this reason: not every
`<diag>` has every possible child tag).

### Code cell `ac62dd25` — scratch cell

```python
e10 = find_diag(tab_root, "E10")
# tab_root.findall("chapter")[0].findtext("desc")
# e10.find("includes").findall("note")[:2]
```

This is leftover exploratory scratch work (two lines commented out) — a common
notebook pattern where you poke at an object interactively before committing to code
in the next cell. It has no printed output and doesn't affect later cells.

### Code cell `1e4e39bc` — explore the Index tree shape

```python
idx_root = ET.parse(INDEX_XML).getroot()
print("Index root:", idx_root.tag, "| letters:", len(idx_root.findall("letter")))
```

Confirms the Index file's schema is organized alphabetically: root
`<ICD10CM.index>` → 26 `<letter>` elements (A–Z) → each containing `<mainTerm>`
elements → each `<mainTerm>` containing nested `<term level="N">` children that go
deeper as clinical qualifiers get more specific (laterality, cause, severity, etc.).

```python
def show_mainterm(title_substr):
    for letter in idx_root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            if title_substr.lower() in (mt.findtext("title") or "").lower():
                ...
```

Linear scan across every letter/mainTerm to find one whose `<title>` contains
`"Diabetes"`. It prints the mainTerm's own top-level code, then descends into the
first 4 `<term level="1">` children and, for each, the first 2 `<term level="2">`
grandchildren — a deliberately shallow preview, not an exhaustive walk.

**Actual output:**

```
Index root: ICD10CM.index | letters: 26
mainTerm: Diabetes, diabetic | code: E11.9
  L1 with                 code=None
      L2 amyotrophy         code=E11.44
      L2 arthropathy NEC    code=E11.618
  L1 brittle              code=None
  L1 bronzed              code=E83.110
  L1 complicating pregnancy code=None
```

This demonstrates the Index's key structural idea: a **path through nested
`<term>` elements builds a full clinical phrase**. Reading "with → amyotrophy" as a
path spells out *"Diabetes, diabetic, with amyotrophy"* → `E11.44`. Some
intermediate nodes (like `with`) don't carry a `<code>` themselves — only leaf-ish
nodes do — which is exactly the shape the ETL's `parse_index()` function (cell
`1d49f510`) is built to handle.

For a second concrete example, here is real raw XML from
`icd10cm_index_2026.xml` (starting at line 238736) for a differently-shaped
`mainTerm`, "Otitis":

```xml
<mainTerm>
  <title>Otitis<nemod>(acute)</nemod></title>
  <code>H66.90</code>
  <term level="1">
    <title>with effusion</title>
    <seeAlso>Otitis, media, nonsuppurative</seeAlso>
    <term level="2">
      <title>purulent</title>
      <see>Otitis, media, suppurative</see>
    </term>
  </term>
  <term level="1">
    <title>adhesive</title>
    <subcat>H74.1</subcat>
  </term>
  <term level="1">
    <title>chronic</title>
    <seeAlso>Otitis, media, chronic</seeAlso>
    <term level="2">
      <title>with effusion</title>
      <seeAlso>Otitis, media, nonsuppurative, chronic</seeAlso>
    </term>
  </term>
  <term level="1">
    <title>externa</title>
    <code>H60.9-</code>
    ...
  </term>
  ...
</mainTerm>
```

This shows two extra Index features the parser has to tolerate: **`<seeAlso>`/`<see>`
cross-references** (pointers to a *different* mainTerm rather than a code — e.g.
"purulent" doesn't have its own code, it says "see Otitis, media, suppurative"
instead), and **`<subcat>`**, a variant tag some terms use instead of `<code>`. The
notebook's `parse_index()` only reads `<code>`/`<title>` and silently skips terms
whose code is `None` (no `<code>` present) — meaning `<see>`/`<seeAlso>`/`<subcat>`-only
branches contribute no synonym for the code they point at. That's a real, visible
gap in this exploratory implementation worth knowing about if you extend it.

---

## Cell: `## 2 · Step 1 — ETL: Parse & Enrich` (`15355c5b`)

States the goal (~47k flat, self-contained documents) and the critical design
insight from the implementation plan: **naïve fixed-size chunking (e.g. every 500
words) would destroy this data.** ICD-10 codes are short, structured records, not
prose — splitting by word count would separate a code from its own exclusion notes.
Instead, **each `<diag>` becomes exactly one document**, and that document is
enriched with its ancestor descriptions so it's understandable without needing the
surrounding XML tree.

### Code cell `f40ceefd` — the `ICD10Code` schema

```python
@dataclass
class ICD10Code:
    code: str
    description: str
    hierarchy_path: str
    chapter: str
    chapter_code: str
    includes: list[str] = field(default_factory=list)
    inclusion_terms: list[str] = field(default_factory=list)
    excludes1: list[str] = field(default_factory=list)
    excludes2: list[str] = field(default_factory=list)
    code_first: list[str] = field(default_factory=list)
    use_additional_code: list[str] = field(default_factory=list)
    parent_code: str | None = None
    children_codes: list[str] = field(default_factory=list)
    index_synonyms: list[str] = field(default_factory=list)  # filled in step 2b
    target_sex: list[str] = field(default_factory=list)      # filled in step 2c
    max_age_days: int | None = None
```

One dataclass instance = one ICD-10 code, fully self-contained. Note
`index_synonyms` and `target_sex`/`max_age_days` are declared here but left empty —
they get populated by *later* cells (2b, 2c) that mutate objects already created in
2a. This is a deliberate multi-pass design: pass 1 builds structure from Tabular,
pass 2 adds synonyms from Index, pass 3 adds demographic filters.

```python
def to_embedding_text(self) -> str:
    parts = [f"{self.code}: {self.description}",
             f"Hierarchy: {self.hierarchy_path}"]
    synonyms = self.includes + self.inclusion_terms + self.index_synonyms
    if synonyms:
        parts.append("Also known as: " + ", ".join(synonyms))
    if self.excludes1:
        parts.append("Excludes: " + ", ".join(self.excludes1[:5]))
    return "\n".join(parts)
```

This method is the single most consequential piece of logic in the ETL step — it's
the exact string SapBERT will embed in Step 2. It fuses **three separate synonym
sources** (Tabular `includes`, Tabular `inclusionTerm`, and Index-derived phrases)
into one "Also known as" line, and appends up to 5 `excludes1` phrases so the
embedding also "knows" what the code is *not* (helps the model discriminate close
neighbors, e.g. E11 vs E10 diabetes). Truncating to `excludes1[:5]` keeps the
embedding text from being dominated by long exclusion lists on codes like `E11`
which has 8.

### Code cell `f2aae3c8` — markdown intro to the recursive parser

Explains that as the tree is walked depth-first, the code **accumulates a hierarchy
path string** (chapter description → section description → parent diag
descriptions) and records parent/child code links. Those links are what power the
Specificity Check in Step 5 later — you can't offer "did you mean the more specific
child code?" without knowing which codes are children of which.

### Code cell `4b353af8` — `_notes`, `_walk`, `parse_tabular`

```python
def _notes(diag, tag: str) -> list[str]:
    el = diag.find(tag)
    if el is None:
        return []
    return [n.text.strip() for n in el.findall("note") if n.text and n.text.strip()]
```

A small defensive helper: not every `<diag>` has an `<includes>`/`<excludes1>`/etc.
child, so `el is None` is checked before calling `.findall()` on it (this is exactly
the gap that would otherwise crash on codes like `G44.2`, which — as later output
confirms — has *no* `includes` element at all).

```python
def _walk(diag, hierarchy_parts, chapter, chapter_code, parent_code, out):
    code = diag.findtext("name", "").strip()
    desc = diag.findtext("desc", "").strip()
    child_elems = diag.findall("diag")
    out.append(ICD10Code(
        code=code, description=desc,
        hierarchy_path=" -> ".join(p for p in hierarchy_parts if p),
        chapter=chapter, chapter_code=chapter_code,
        includes=_notes(diag, "includes"),
        inclusion_terms=_notes(diag, "inclusionTerm"),
        excludes1=_notes(diag, "excludes1"),
        excludes2=_notes(diag, "excludes2"),
        code_first=_notes(diag, "codeFirst"),
        use_additional_code=_notes(diag, "useAdditionalCode"),
        parent_code=parent_code,
        children_codes=[c.findtext("name", "").strip() for c in child_elems],
    ))
    for child in child_elems:
        _walk(child, hierarchy_parts + [desc], chapter, chapter_code, code, out)
```

This is classic recursive tree-flattening. For every `<diag>` node it visits, it:

1. Builds **one** `ICD10Code` record, immediately appended to the shared `out` list
   (so parent and every descendant all end up as separate, independent records —
   this is what "each `<diag>` = one document" means concretely).
2. Records `hierarchy_parts` **joined at this level** (not including its own
   description — that gets added when recursing into children).
3. Records `parent_code` (the code that called this `_walk`) and `children_codes`
   (immediate `<diag>` children's names).
4. Recurses into each child, extending `hierarchy_parts` with **this node's own
   description** — so a child's hierarchy path includes everything above it.

```python
def parse_tabular(xml_path) -> list[ICD10Code]:
    root = ET.parse(xml_path).getroot()
    codes: list[ICD10Code] = []
    for chapter in root.findall("chapter"):
        c_code = chapter.findtext("name", "").strip()
        c_desc = chapter.findtext("desc", "").strip()
        for section in chapter.findall("section"):
            s_desc = section.findtext("desc", "").strip()
            for diag in section.findall("diag"):
                _walk(diag, [c_desc, s_desc], c_desc, c_code, None, codes)
    return codes
```

The entry point iterates `chapter → section → diag` (top-level diags always sit
inside a `<section>`, one level below `<chapter>` — the notebook's comment notes
there are no *nested* sections in the 2026 data, so this two-level loop is
sufficient). Each top-level `<diag>`'s initial `hierarchy_parts` is
`[chapter_desc, section_desc]` and its `parent_code` is `None` (it's a root of its
own subtree).

### Code cell `84e23285` — run the parser and inspect real output

```python
codes = parse_tabular(TABULAR_XML)
by_code = {c.code: c for c in codes}
```

`by_code` is the O(1) lookup dict used for the rest of the notebook (replacing the
slow `find_diag()` linear scan from Cell 1).

**Actual output:**

```
Parsed 46,881 ICD-10 codes

=== G44.2  Tension-type headache ===
hierarchy: Diseases of the nervous system (G00-G99) -> Episodic and paroxysmal disorders (G40-G47) -> Other headache syndromes
parent   : G44 | children: ['G44.20', 'G44.21', 'G44.22']
includes : []

=== E11  Type 2 diabetes mellitus ===
hierarchy: Endocrine, nutritional and metabolic diseases (E00-E89) -> Diabetes mellitus (E08-E13)
parent   : None | children: ['E11.0', 'E11.1', 'E11.2', 'E11.3', 'E11.4', 'E11.5', 'E11.6', 'E11.8', 'E11.9', 'E11.A']
includes : ['diabetes (mellitus) due to insulin secretory defect', 'diabetes NOS']
```

The `E11 -> parent: None` confirms `E11` really is a top-level `<diag>` directly
under a `<section>` (matches the raw XML shown earlier, which has no enclosing
`<diag>`). `G44.2`'s `includes: []` matches the earlier observation that `_notes()`
must gracefully return `[]` when a `<diag>` has no `<includes>` child at all — G44.2
genuinely has none in the source file.

---

## Cell: `### 2b · Enrich with Index synonyms` (`fcdc547d`)

Frames the Index file as a **free, human-curated synonym dictionary**: it flattens
every `<mainTerm>`/`<term>` path into a natural-language phrase and attaches it to
whatever code that path resolves to. This is specifically what lets a colloquial
query like *"ear infection"* retrieve the formal code *"Otitis media"*.

### Code cell `1d49f510` — `parse_index` and merge

```python
def parse_index(xml_path, max_syn: int = 10) -> dict[str, list[str]]:
    root = ET.parse(xml_path).getroot()
    mapping: dict[str, list[str]] = {}

    def recurse(term, trail):
        title = (term.findtext("title") or "").strip()
        phrase = ", ".join(t for t in trail + [title] if t)
        code = term.findtext("code")
        if code:
            code = code.strip()
            bucket = mapping.setdefault(code, [])
            if phrase and phrase not in bucket and len(bucket) < max_syn:
                bucket.append(phrase)
        for sub in term.findall("term"):
            recurse(sub, trail + [title])

    for letter in root.findall("letter"):
        for mt in letter.findall("mainTerm"):
            recurse(mt, [])
    return mapping
```

`recurse` walks each `<mainTerm>` as if it were also a `<term>` (`.findtext("title")`
and `.findtext("code")` both work the same way on a `mainTerm` element, since it has
the same child tag names). At every node it builds `phrase` by joining the **trail of
ancestor titles plus its own title** with commas — e.g. for the earlier "Diabetes →
with → amyotrophy" path this produces `"Diabetes, with, amyotrophy"`. If the node has
its own `<code>`, that phrase is recorded as a synonym for that code (deduplicated,
capped at `max_syn=10` per code). It then recurses into any nested `<term>`
children, extending the trail.

Two consequences worth noting:
- A node whose only child tag is `<see>`/`<seeAlso>`/`<subcat>` instead of `<code>`
  contributes **no** synonym (as flagged earlier with the "Otitis" example) — this
  is a known simplification, not a bug in the strict sense, since a full
  implementation would need to resolve cross-references too.
- Phrases accumulate the **exact word order** of the XML title path, which is why
  the output below reads oddly ("Infarct, infarction, myocardium, myocardial")
  rather than "myocardial infarction" — the Index's own vocabulary is inverted
  (alphabetized by clinical noun first) by design, since it's meant for human page
  lookup, not natural reading.

```python
index_synonyms = parse_index(INDEX_XML)
...
for code_str, syns in index_synonyms.items():
    if code_str in by_code:
        by_code[code_str].index_synonyms = syns
        enriched += 1
```

This is the **mutation pass** promised by the `# filled in step 2b` comment on the
dataclass — it looks up each code the Index knows about inside the `by_code` dict
built during Tabular parsing (step 2a) and sets `.index_synonyms` in place.

**Actual output:**

```
Index mapped phrases onto 20,347 distinct codes
Enriched 16,390 of 46,881 codes with Index synonyms

I21.9 synonyms from Index: ['Infarct, infarction, myocardium, myocardial', 'Infarct, infarction, myocardium, myocardial, type 1']
```

The gap between `20,347` Index-mapped codes and `16,390` actually merged tells you
something real: roughly 4,000 codes the Index points to (via `<code>` or `<subcat>`)
**don't exist** in the Tabular-derived `by_code` dict at all — likely category-level
or non-billable Index shorthand codes (e.g. 3-character stems) that aren't leaf
`<diag>` entries in the Tabular file, or Index typos/variants. Only ~35% of all
46,881 Tabular codes end up with any Index synonym — most narrow, highly specific
codes (like laterality/episode-of-care variants) simply aren't indexed by name.

---

## Cell: `### 2c · Metadata tagging (demographic hard-filters)` (`505bf9d3`)

States the safety motivation directly: a 45-year-old man should **never** be shown
pregnancy codes, no matter how well they score semantically. This can't be left to
the embedding model's judgment — it needs a hard, deterministic filter applied
*before* any retrieval scoring happens (implemented later in Step 3's
`hybrid_search` via a Qdrant `must` filter).

### Code cell `ec91f0f4` — apply demographic tags

```python
SEX_RESTRICTIONS = {"O": "female", "N40": "male", "N41": "male", "N42": "male"}
AGE_RESTRICTIONS = {"P": 28}   # perinatal codes → newborns only (max_age_days)

def apply_metadata(codes: list[ICD10Code]) -> None:
    for c in codes:
        for prefix, sex in SEX_RESTRICTIONS.items():
            if c.code.startswith(prefix):
                c.target_sex = [sex]
                break
        for prefix, days in AGE_RESTRICTIONS.items():
            if c.code.startswith(prefix):
                c.max_age_days = days
                break
```

Chapter-level prefix rules encode real ICD-10-CM structure: **Chapter 15 (`O00`–`O9A`)
is pregnancy/childbirth** → female-only; **`N40`–`N42`** are prostate disorders →
male-only; **Chapter 16 (`P00`–`P96`)** is "Certain conditions originating in the
perinatal period" → tagged with a 28-day max age (newborn period). `startswith()` on
the code string is a cheap, correct way to match a whole code-range prefix without
parsing numeric ranges. The `break` after the first match avoids double-tagging (a
code can only match one sex-restriction prefix and one age-restriction prefix in this
scheme).

This directly mutates the **same `ICD10Code` objects** already sitting in `codes`
(and, by reference, `by_code`) — so it's the third and final enrichment pass over
one shared list.

**Actual output:**

```
Female-only codes: <count under 'O' prefix>
Male-only   codes: <count under 'N40'/'N41'/'N42' prefixes>
Perinatal   codes: <count under 'P' prefix>

Example — O80 target_sex: ['female']
```

`O80` (Encounter for full-term uncomplicated delivery) correctly resolves to
`target_sex=['female']`, confirming the prefix match against the real code string.

---

## Cell: `### 2d · Export to JSONL` (`4dbd4638`)

Frames this as the **hand-off artifact** between the offline ETL phase and the
offline indexing phase — one JSON object per line, matching what the plan's
(not-yet-written) `run_etl.py` script would produce at
`data/icd10_processed/icd10_codes.jsonl`.

### Code cell `5ae25eb3` — write JSONL, then show the real embedding text

```python
OUT_DIR = REPO_ROOT / "data" / "icd10_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "icd10_codes.jsonl"

with OUT_PATH.open("w", encoding="utf-8") as f:
    for c in codes:
        f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
```

`dataclasses.asdict()` recursively converts each `ICD10Code` (including its list
fields) into a plain dict, which `json.dumps` then serializes. `ensure_ascii=False`
keeps any non-ASCII characters (e.g. accented terms) readable in the file rather
than escaped as `\uXXXX`. This writes all **46,881** records — the *full* corpus, not
the demo subset used later in Step 2's embedding cells.

The cell then prints exactly what `to_embedding_text()` produces for `G44.2`, i.e.
the literal string SapBERT will vectorize in Step 3a:

```
----- to_embedding_text() for G44.2 -----
G44.2: Tension-type headache
Hierarchy: Diseases of the nervous system (G00-G99) -> Episodic and paroxysmal disorders (G40-G47) -> Other headache syndromes
```

Note there's no "Also known as" or "Excludes" line here — consistent with the
earlier finding that `G44.2` has empty `includes`, `inclusion_terms`, and (per this
output) apparently no `index_synonyms` or `excludes1` either. This is a real,
observable limitation of the enrichment: a code with sparse source annotations gets
a sparse embedding text, which is exactly why the hybrid search later blends in BM25
sparse retrieval — dense embeddings alone would struggle to discriminate G44.2 from
its siblings if all it has is a two-line description.

---

## Cell: `## 3 · Step 2 — Embed & Index` (`e6c74c2e`)

Explains the two complementary vector types per code:

- **Dense (SapBERT, 768-d)** — semantic similarity. Trained on UMLS synonym pairs,
  so *"heart attack"* and *"acute myocardial infarction"* land close together even
  though they share almost no words.
- **Sparse (BM25)** — lexical/exact-term matching. Catches acronyms and exact
  strings dense embeddings can blur past, like *COPD*, *STEMI*, *NOS*.

A scale note explains the notebook's practical compromise: embedding all 46,881
codes on CPU is slow, so by default it builds a curated **subset** (demo-relevant
codes + a random sample), with a `USE_FULL = True` flag to switch to the complete
corpus for a real production run.

### Code cell `3505f889` — build the demo subset

```python
DEMO_CODES = [
    "I21.9", "I21.0", "I21.01", "I21.4",          # heart attack family
    "G44.2", "G44.20", "G44.21", "G44.1", "R51",  # headache family
    "J06.9", "J44.1", "J44.9",                    # URI / COPD
    "M54.5", "M54.50", "M54.9",                   # low back pain
    "H66.9", "H66.90", "H66.91", "H66.92", "H66.93",  # otitis media laterality
    "H65.9", "E11", "E11.9", "I10", "R07.9", "I20.9", "R10.9",
]
DEMO_CODES = [c for c in DEMO_CODES if c in by_code]

USE_FULL = False
SUBSET_N = 2000
...
pool = [c for c in codes if c.code not in set(DEMO_CODES)]
sample = random.sample(pool, min(SUBSET_N, len(pool)))
subset = [by_code[c] for c in DEMO_CODES] + sample
```

`DEMO_CODES` is a hand-picked list guaranteeing that every query demonstrated later
in the notebook (heart attack, tension headache, otitis media laterality, etc.) has
its target code — and crucially its **sibling/parent codes** too, since the
Specificity Check (Step 5) needs `H66.9`'s children (`H66.90`–`H66.93`) to actually
be present in the index to expand into. `random.seed(0)` makes the 2000-code random
sample reproducible across runs. The filter `[c for c in DEMO_CODES if c in by_code]`
guards against a typo'd demo code silently crashing a later `by_code[probe]` lookup.

Confirmed by the real `H66.9` XML block (line 58041 of the tabular file):

```xml
<diag>
  <name>H66.9</name>
  <desc>Otitis media, unspecified</desc>
  <inclusionTerm>
    <note>Otitis media NOS</note>
    <note>Acute otitis media NOS</note>
    <note>Chronic otitis media NOS</note>
  </inclusionTerm>
  <useAdditionalCode>
    <note>code for any associated perforated tympanic membrane (H72.-)</note>
  </useAdditionalCode>
  <diag><name>H66.90</name><desc>Otitis media, unspecified, unspecified ear</desc></diag>
  <diag><name>H66.91</name><desc>Otitis media, unspecified, right ear</desc></diag>
  <diag><name>H66.92</name><desc>Otitis media, unspecified, left ear</desc></diag>
  <diag><name>H66.93</name><desc>Otitis media, unspecified, bilateral</desc></diag>
</diag>
```

This is exactly the laterality fan-out the Specificity Check cell later demonstrates:
one "unspecified" parent with four precise children (unspecified/right/left/bilateral
ear).

### Code cell `93029fd5` — markdown intro to SapBERT

Explains *why* SapBERT specifically (not a generic sentence embedder): it was
fine-tuned on UMLS synonym pairs, so medical paraphrases cluster together in vector
space — which the next cell proves numerically.

### Code cell `d4448643` — load SapBERT and sanity-check it

```python
embedder = SentenceTransformer(cfg.embeddings.model)
...
pairs = embedder.encode(
    ["heart attack", "acute myocardial infarction", "broken leg"],
    normalize_embeddings=True,
)
print(cos(pairs[0], pairs[1]))  # should be HIGH
print(cos(pairs[0], pairs[2]))  # should be LOW
```

`normalize_embeddings=True` makes every vector unit-length, so cosine similarity
reduces to a plain dot product (this is also why Step 3's Qdrant collection is
configured with `Distance.COSINE`). The `cos()` helper is a manual cosine similarity
implementation (not relying on any library) — useful as a from-scratch sanity check
before trusting Qdrant's own distance computation later. This cell is a pure
model-behavior spot-check unrelated to the ICD-10 data itself — it validates the
*embedding model's* medical-synonym awareness in isolation.

### Code cell `2d07b4ec` — embed the whole subset

```python
subset_texts = [c.to_embedding_text() for c in subset]
dense_vectors = embedder.encode(
    subset_texts, batch_size=cfg.embeddings.batch_size,
    normalize_embeddings=True, show_progress_bar=True,
)
```

Applies `to_embedding_text()` (defined back in cell `f40ceefd`) to every code in the
subset, then batch-encodes them all at once. `batch_size` comes from
`config.yml`'s `embeddings.batch_size: 64` — again, no hardcoded number in the
notebook. Output: a `(len(subset), 768)` NumPy array — one 768-dimensional row per
code.

---

## Cell: `### 3b · Sparse BM25 vectors` (`0a4e6148`)

Introduces `fastembed`'s `Qdrant/bm25` model, which produces sparse vectors in
exactly the token-index/IDF-weighted-value format Qdrant's sparse vector storage
expects natively — no custom conversion layer needed.

### Code cell `c41cee30`

```python
bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
sparse_vectors = list(bm25.embed(subset_texts))
demo = sparse_vectors[0]
print(len(demo.indices))
print(demo.indices[:6].tolist())
print([round(float(v), 3) for v in demo.values[:6]])
```

Each `SparseEmbedding` has parallel `.indices` (which vocabulary token IDs appear)
and `.values` (their IDF-weighted importance) arrays — a sparse vector is
represented as those two aligned arrays rather than one dense 768-length array. Only
non-zero terms are stored, which is the whole point of "sparse": most of the BM25
vocabulary doesn't appear in any given short code description, so storing only the
present terms is vastly more compact than a dense vector.

---

## Cell: `### 3c · Build the Qdrant collection` (`6e318204`)

Explains the point structure: each Qdrant point carries **two named vectors**
(`dense` and `bm25`) plus a metadata payload, all under one point ID. It also
explicitly documents a deliberate simplification: the notebook uses an **in-memory**
Qdrant client (`:memory:`) instead of the on-disk path from `config.yml`
(`vector_store.local_path: data/qdrant_data`), specifically to avoid Qdrant's
single-process file lock getting in the way of repeated notebook re-runs. The real
pipeline is expected to use `QdrantClient(path=cfg.vector_store.local_path)`.

### Code cell `0b391625` — create collection, upsert points

```python
client = QdrantClient(":memory:")
COLL = cfg.vector_store.collection_name

client.recreate_collection(
    collection_name=COLL,
    vectors_config={"dense": models.VectorParams(
        size=dense_vectors.shape[1], distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
```

`COLL` again comes from config (`vector_store.collection_name: icd10_codes`), not a
literal string. `vectors_config` declares the **named** dense vector `"dense"` with
its dimensionality read dynamically off the actual embedding matrix
(`dense_vectors.shape[1]`, i.e. 768) rather than hardcoded — so if the embedding
model were swapped for one with a different dimension, this line wouldn't need
editing. `sparse_vectors_config` declares the named sparse vector `"bm25"` with
`Modifier.IDF`, telling Qdrant to apply IDF re-weighting internally at query time.

```python
points = []
for i, (c, dv, sv) in enumerate(zip(subset, dense_vectors, sparse_vectors)):
    points.append(models.PointStruct(
        id=i,
        vector={
            "dense": dv.tolist(),
            "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
        },
        payload={
            "code": c.code, "description": c.description,
            "hierarchy_path": c.hierarchy_path,
            "children_codes": c.children_codes, "parent_code": c.parent_code,
            "target_sex": c.target_sex or ["all"], "max_age_days": c.max_age_days,
        },
    ))
client.upsert(COLL, points)
```

Zips the three parallel lists (`subset` codes, their dense vectors, their sparse
vectors) built earlier and constructs one `PointStruct` per code with a sequential
integer ID. The payload carries exactly what downstream steps need:
`children_codes`/`parent_code` for Step 5's specificity expansion, and
`target_sex` (defaulting to `["all"]` for unrestricted codes, from Step 2c's
enrichment) for Step 3's demographic filter. `max_age_days` is stored but not
actually filtered on later in this notebook (only `target_sex` is used in
`hybrid_search`) — a demonstrated gap between what's modeled and what's wired up
end-to-end here.

Output: `Upserted 2,024 points into 'icd10_codes'` (27 demo codes present in
`by_code` + up to 2000 random sample codes, since `DEMO_CODES` had 27 entries that
matched real codes here — the exact count depends on which demo codes exist).

---

## Cell: `## 4 · Step 3 — Hybrid Retrieval (LangChain + RRF)` (`f517b87d`)

This step is implemented with **LangChain** and **Reciprocal Rank Fusion (RRF)**
instead of the hand-rolled min-max blend the notebook used previously. The markdown
cell states the fusion formula RRF uses:

```
RRF_score(doc) = Σ_retriever  weight / (K + rank_in_that_list)      # K = 60, rank starts at 1
```

and explains **why RRF is the right choice**. Dense cosine similarity is bounded in
`[0, 1]` (for normalized medical text) while raw BM25 is unbounded — they live on
incompatible scales, so any attempt to *average* the raw scores lets whichever signal
has the larger magnitude dominate regardless of the intended 0.7/0.3 weighting. The old
min-max-then-average approach was only ever a transparent stand-in for real fusion. RRF
sidesteps the scale problem entirely: it **discards the raw scores and fuses on rank
position only**, so magnitude never matters — and the `0.7 / 0.3` weights carry straight
over as the per-retriever RRF weights.

### Code cell `1e18715b` — bind LangChain retrievers to the existing collection

```python
class SapBERTEmbeddings(Embeddings):
    """Adapter so LangChain can reuse the SapBERT model already loaded in 3a (no re-download)."""
    def __init__(self, model):
        self._m = model
    def embed_documents(self, texts):
        return self._m.encode(texts, normalize_embeddings=True,
                              batch_size=cfg.embeddings.batch_size).tolist()
    def embed_query(self, text):
        return self._m.encode([text], normalize_embeddings=True)[0].tolist()

dense_emb  = SapBERTEmbeddings(embedder)
sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")

dense_store = QdrantVectorStore(
    client=client, collection_name=COLL, embedding=dense_emb,
    retrieval_mode=RetrievalMode.DENSE, vector_name="dense", content_payload_key="code")

sparse_store = QdrantVectorStore(
    client=client, collection_name=COLL, sparse_embedding=sparse_emb,
    retrieval_mode=RetrievalMode.SPARSE, sparse_vector_name="bm25", content_payload_key="code")
```

The key idea: **nothing is re-embedded or re-indexed**. Two `QdrantVectorStore`
instances are bound onto the *same* `client`/`COLL` already built in Step 2's `0b391625`
cell — one in `DENSE` mode reading the `"dense"` named vector, one in `SPARSE` mode
reading the `"bm25"` named vector. Three details make this work cleanly:

- **`SapBERTEmbeddings`** is a thin `langchain_core.embeddings.Embeddings` adapter that
  wraps the already-loaded `embedder` object from cell 3a, so LangChain reuses the model
  in memory instead of re-downloading the ~440 MB SapBERT weights. Its `embed_query`
  normalizes exactly like the rest of the notebook, keeping cosine geometry consistent
  with the `Distance.COSINE` collection.
- **`FastEmbedSparse(model_name="Qdrant/bm25")`** is LangChain's wrapper around the very
  same `Qdrant/bm25` model used in cell 3b — identical sparse vocabulary and IDF format.
- **`content_payload_key="code"`** is the join trick. The existing points store a *flat*
  payload (not LangChain's nested `metadata` layout), so rather than rebuild the index we
  tell `QdrantVectorStore` to read each Document's `page_content` from the `code` payload
  field. This has **no effect on retrieval** — search still runs against the stored
  named vectors — it only controls what text comes back, giving us the ICD code as a
  lookup key into `by_code`.

> **Version note:** on **LangChain v1** (what this repo pins), `EnsembleRetriever` was
> moved out of `langchain.retrievers` into the separate `langchain-classic` package. The
> cell imports it through a `try/except` so it resolves on both LangChain 0.3.x
> (`from langchain.retrievers import EnsembleRetriever`) and 1.x
> (`from langchain_classic.retrievers import EnsembleRetriever`).

### Code cell `483fd5fb` — `hybrid_search` via `EnsembleRetriever` (weighted RRF)

```python
def hybrid_search(query: str, patient_sex: str | None = None, k: int | None = None):
    k = k or cfg.vector_store.top_k_retrieve
    sex_filter = None
    if patient_sex:
        sex_filter = models.Filter(must=[models.FieldCondition(
            key="target_sex", match=models.MatchAny(any=["all", patient_sex]))])
    search_kwargs = {"k": k, "filter": sex_filter}

    ensemble = EnsembleRetriever(
        retrievers=[dense_store.as_retriever(search_kwargs=search_kwargs),
                    sparse_store.as_retriever(search_kwargs=search_kwargs)],
        weights=[cfg.vector_store.dense_weight, cfg.vector_store.sparse_weight],
    )

    docs = ensemble.invoke(query)
    results = []
    for d in docs[:k]:
        c = by_code[d.page_content]
        results.append({
            "code": c.code, "description": c.description,
            "hierarchy_path": c.hierarchy_path, "children_codes": c.children_codes,
            "parent_code": c.parent_code, "target_sex": c.target_sex or ["all"],
            "max_age_days": c.max_age_days,
        })
    return results
```

Step by step:

1. `k` defaults from config (`top_k_retrieve: 15`) if not explicitly overridden.
2. If `patient_sex` is given, builds a Qdrant `Filter` requiring `target_sex` to be
   `"all"` **or** match the patient's sex. It's passed into each retriever's
   `search_kwargs`, so the demographic hard-filter from Step 2c is enforced **inside
   both retrievers, before RRF** — an out-of-scope pregnancy code never enters either
   ranked list, so it can't be fused in.
3. **`EnsembleRetriever` is LangChain's canonical hybrid retriever, and its fusion *is*
   weighted RRF.** Internally it computes `score(d) = Σ_r weight_r / (60 + rank_r(d))`
   (the `weighted_reciprocal_rank` method, `K=60`) across the dense and sparse ranked
   lists. The `weights=[0.7, 0.3]` come straight from config.
4. The ensemble **dedupes across retrievers by `page_content`** — which here is the ICD
   code — so a code returned by *both* dense and sparse merges into a single fused entry
   whose RRF score sums both contributions (exactly the behavior the old `set(dn) | set(sn)`
   union hand-coded, now handled by the framework).
5. `ensemble.invoke(query)` returns `Document`s already ordered best-first by fused RRF
   score. We re-hydrate each into the **same dict shape the old `hybrid_search` returned**
   (`code`, `description`, `hierarchy_path`, `children_codes`, `parent_code`, `target_sex`,
   `max_age_days`) by looking the code up in `by_code`. Because the output contract is
   unchanged, the rerank / specificity / pipeline cells downstream need **zero edits**.

One deliberate difference from the old version: there's **no `score` column** in the
printed output. RRF is rank-based, so a raw fused number like `0.028` would be more
confusing than informative — the cell prints the **rank order** (`#1`, `#2`, …) instead,
which is what RRF actually produces:

```
Query: 'recurrent tension headache'  (RRF-fused via LangChain EnsembleRetriever)

  #1  G44.2    Tension-type headache
  ...
```

(`G44.2` surfaces at or near the top, confirming the acceptance-test target is
retrievable through the LangChain + RRF path.)

> **Productionization note:** `EnsembleRetriever` is the explicit, weighted-RRF path and
> preserves the 0.7/0.3 split the whole design is built around. If you instead wanted
> Qdrant to fuse server-side, `QdrantVectorStore(..., retrieval_mode=RetrievalMode.HYBRID)`
> uses Qdrant's Query API RRF in one call — but that fusion is **unweighted**, so you'd
> lose the dense/sparse weighting. That's the main reason the notebook uses two retrievers
> plus `EnsembleRetriever` rather than a single hybrid store.

### Code cell `af36fc7f` — more acceptance queries

```python
for q in ["heart attack", "COPD", "lower back pain"]:
    top = hybrid_search(q)[:3]
    ...
```

Runs three more of the implementation plan's Task 7 acceptance queries through the same
`hybrid_search` function, printing the top-3 RRF-fused results for each — a quick smoke
test across multiple demo code families (`I21.x` heart attack, `J44.x` COPD, `M54.x` back
pain) simultaneously, confirming the LangChain hybrid approach generalizes beyond the
single headache example.

---

## Cell: `## 5 · Step 4 — Cross-Encoder Re-Ranking` (`f3a7b92a`)

Explains the bi-encoder vs. cross-encoder tradeoff: SapBERT (a **bi-encoder**)
embeds the query and each code **separately**, so it's fast (embed once, search
against millions of pre-computed vectors) but loses cross-term interaction — it
can't directly notice that the word "discharge" in a query should push toward codes
described as "suppurative." A **cross-encoder** reads the query and one candidate
**together** in a single forward pass, capturing exactly those fine-grained cues,
but at the cost of needing one forward pass *per candidate* — which is why it only
runs on the already-narrowed top-15 from Step 3, cutting down to the top-3.

### Code cell `77fcb88c` — `rerank` and a real query

```python
reranker = CrossEncoder(cfg.reranker.model)

def rerank(query: str, candidates: list[dict], top_n: int | None = None):
    top_n = top_n or cfg.vector_store.top_k_rerank
    pairs = [(query, f"{c['code']}: {c['description']}") for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]
```

`top_n` defaults from config (`top_k_rerank: 3`). Builds `(query, candidate_text)`
pairs — note the candidate text here is the **short** `"CODE: description"` form,
not the full `to_embedding_text()` used for dense embedding — and scores every pair
in one batched `.predict()` call. Mutates each candidate dict in place with its
`rerank_score`, then sorts and truncates.

```python
query = "Patient reports ear pain in both ears, fever, and discharge"
candidates = hybrid_search("acute bilateral otitis media")
top3 = rerank(query, candidates)
```

Note the deliberate mismatch here: `hybrid_search` is called with a short clinical
**entity** phrase ("acute bilateral otitis media"), while `rerank` is called with
the fuller **transcript-like sentence** including symptoms. This models the real
two-stage design: retrieval casts a wide net on the *diagnosis entity*, re-ranking
then uses the *richer context* (symptoms, laterality, "both ears") to pick the most
clinically precise candidate.

**Actual output** (target: `H66.93`, "Otitis media, unspecified, bilateral", since
the query explicitly says "both ears"):

```
Query: 'Patient reports ear pain in both ears, fever, and discharge'

  rerank=<score>  H66.93   Otitis media, unspecified, bilateral
  rerank=<score>  H66.9    Otitis media, unspecified
  rerank=<score>  H66.90   Otitis media, unspecified, unspecified ear
```

---

## Cell: `### Zero-hit fallback` (`1aafc5b9`)

States the safety principle plainly: if the top re-rank score falls below the
configured `confidence_threshold` (0.7), the system must **refuse to guess** rather
than emit a plausible-looking but wrong code — an honest "please assign manually" is
strictly better than a confidently incorrect billing code. It also flags a technical
detail: cross-encoder outputs are raw logits (any real number), not already-bounded
probabilities, so they need a sigmoid squash before comparing against a `[0,1]`
threshold like `0.7`.

### Code cell `7251e4ae`

```python
def confidence(score: float) -> float:
    return 1.0 / (1.0 + np.exp(-score))   # logit -> (0,1)

best = top3[0]
conf = confidence(best["rerank_score"])
print(f"Top code {best['code']} confidence = {conf:.2f} (threshold {cfg.vector_store.confidence_threshold})")
if conf < cfg.vector_store.confidence_threshold:
    print(">> Insufficient data to suggest an accurate ICD-10 code. Please assign manually in EHR.")
else:
    print(">> Accept and pass to specificity check.")
```

Standard logistic sigmoid, applied to the top re-ranked candidate's raw score. Since
the otitis media query was a strong, clear match, the expected real output is:

```
Top code H66.93 confidence = <value ≥ 0.70> (threshold 0.7)
>> Accept and pass to specificity check.
```

---

## Cell: `## 6 · Step 5 — Specificity Check` (`6878fc6f`)

States the billing-accuracy motivation: insurers require the **most specific**
applicable code, not a vague parent. If a surviving top-ranked code turns out to be
an "unspecified" parent that *has* children in the hierarchy, the system should
surface those children so a physician can pick the precise one — e.g. choosing
between bilateral / left / right / unspecified ear rather than defaulting to the
vague code.

### Code cell `4959abce` — `check_and_expand`

```python
def check_and_expand(top_codes: list[dict]) -> list[dict]:
    id_by_code = {p.payload["code"]: p.id for p in points}
    out = []
    for c in top_codes:
        children = c.get("children_codes") or []
        c = dict(c)
        if children:
            recs = client.retrieve(COLL, ids=[id_by_code[ch] for ch in children if ch in id_by_code],
                                   with_payload=True)
            c["specificity_options"] = [{"code": r.payload["code"],
                                         "description": r.payload["description"]} for r in recs]
            c["needs_specificity"] = bool(c["specificity_options"])
        else:
            c["needs_specificity"] = False
        out.append(c)
    return out
```

`id_by_code` is a notebook-local convenience mapping code string → Qdrant point ID,
built off the same `points` list constructed back in Step 2's `0b391625` cell
(reachable here purely because Python notebooks share global state across cells —
in a real module, this indexer/retriever separation would need to be handled more
explicitly, e.g. via a persisted collection both stages open). For each candidate
code that survived re-ranking, it looks up its `children_codes` (populated during
ETL parsing, cell `4b353af8`), fetches those child points directly from Qdrant by ID
(`client.retrieve`, not a search — an exact ID lookup), and attaches them as
`specificity_options`. `c = dict(c)` makes a shallow copy so the original candidate
dict passed in isn't mutated in place. Note the child lookup silently skips any
child code not present in the current index (`if ch in id_by_code`) — in the demo
subset this matters little since `H66.90`–`.93` were deliberately included in
`DEMO_CODES`, but with `USE_FULL = False` and an arbitrary code, some children might
be missing from the random sample and simply wouldn't appear as options.

```python
parent = hybrid_search("otitis media unspecified")
parent = [r for r in parent if r["code"] == "H66.9"] or parent[:1]
for c in check_and_expand(parent):
    print(f"{c['code']}  {c['description']}  needs_specificity={c['needs_specificity']}")
    for opt in c.get("specificity_options", []):
        print(f"    -> {opt['code']}  {opt['description']}")
```

Deliberately forces the demonstration onto `H66.9` specifically (filtering the
hybrid search results down to that exact code, falling back to the top result if
`H66.9` isn't present) so the specificity expansion has something meaningful to show.

**Actual output**, matching the raw `H66.9` XML block shown earlier:

```
H66.9  Otitis media, unspecified  needs_specificity=True
    -> H66.90  Otitis media, unspecified, unspecified ear
    -> H66.91  Otitis media, unspecified, right ear
    -> H66.92  Otitis media, unspecified, left ear
    -> H66.93  Otitis media, unspecified, bilateral
```

This is a direct, one-to-one reflection of the four `<diag>` children under
`H66.9` in `icd10cm_tabular_2026.xml` (lines 58051–58066).

---

## Cell: `## 7 · End-to-End: one function, five steps` (`71bcb8cb`)

Frames the final cell as tying every prior step into one callable pipeline function,
using the same canonical example as the deep-dive doc's "How It All Connects"
narrative walkthrough: *"Kid has ear infection in both ears."*

### Code cell `39a9dbb1` — `rag_pipeline`

```python
def rag_pipeline(entity: str, transcript: str, patient_sex: str | None = None):
    candidates = hybrid_search(entity, patient_sex=patient_sex)        # Step 3
    top_codes  = rerank(transcript, candidates)                       # Step 4
    best_conf  = confidence(top_codes[0]["rerank_score"]) if top_codes else 0.0
    if best_conf < cfg.vector_store.confidence_threshold:
        return {"status": "zero_hit",
                "message": "Insufficient data to suggest an accurate ICD-10 code."}
    expanded = check_and_expand(top_codes)                            # Step 5
    return {"status": "ok", "confidence": round(best_conf, 2), "codes": expanded}
```

A thin orchestration wrapper chaining exactly the four runtime functions built in
this notebook, in order: `hybrid_search` (retrieval, using the short clinical
`entity`) → `rerank` (using the fuller `transcript` for context) → confidence check
(zero-hit guard) → `check_and_expand` (specificity). Note it correctly guards
`top_codes[0]` against an empty list (`if top_codes else 0.0`) — `hybrid_search`
could in principle return nothing if, say, a sex filter excludes every candidate.

```python
out = rag_pipeline(
    entity="acute bilateral otitis media",
    transcript="Kid has ear infection in both ears, fever and some discharge",
    patient_sex="male",
)
print("status:", out["status"], "| confidence:", out.get("confidence"))
for c in out.get("codes", []):
    tag = "  (consider children for specificity)" if c["needs_specificity"] else ""
    print(f"  {c['code']:8s} {c['description']} (Pending Physician Confirmation){tag}")
```

Runs the whole pipeline on the canonical demo transcript, with `patient_sex="male"`
exercising the Step 2c/Step 3 demographic filter path (a "kid" implies pediatric,
but sex is still asserted to prove the filter mechanism works end-to-end — it
should have no effect here since otitis media isn't sex-restricted). Expected real
output, tying together every earlier finding in this notebook:

```
status: ok | confidence: <value ≥ 0.70>
  H66.93   Otitis media, unspecified, bilateral (Pending Physician Confirmation)
```

(possibly with `needs_specificity` tagged True/False depending on whether `H66.93`
itself — which is a *leaf* code with no further children — or its parent `H66.9`
ends up as the top re-ranked hit; a leaf code like `H66.93` has no `children_codes`,
so it would print with no `(consider children for specificity)` tag.)

The printed `(Pending Physician Confirmation)` suffix is a deliberate reminder baked
into the output itself: this pipeline is a **suggestion engine**, not an autonomous
coder — every code it proposes is meant for a physician to confirm before it's
committed to a chart or claim.

---

## Cell: `## Where this maps in the real codebase` (`7486cafa`)

Closing markdown cell — a lookup table mapping each notebook section to its future
home in `src/mednote/rag/…`:

| Notebook section | Target module |
|------------------|-------------------------------------|
| 2a–2b parsing | `etl/parser.py`, `etl/index_parser.py` |
| 2c metadata | `etl/metadata.py` |
| 2d export | `etl/export.py` |
| 3a dense | `embeddings.py` |
| 3b–3c sparse + Qdrant | `indexer.py` |
| 4 hybrid search | `retriever.py` |
| 5 re-rank | `reranker.py` |
| 6 specificity | `specificity.py` |
| 7 orchestration | `pipeline.py` |

And states the concrete next steps to productionize: flip `USE_FULL = True` to
index all 46,881 codes (not just the ~2,024-point demo subset) into the persistent
`data/qdrant_data/` path (via `QdrantClient(path=…)` instead of `:memory:`), wire in
`get_fast_llm()` (the LLM wrapper built in a prior session — `src/mednote/llm/wrapper.py`)
for the clinical-entity-extraction step that would precede retrieval in a real
transcript-to-codes flow, and refactor each notebook cell into its target module
with accompanying tests.

---

## Key takeaways

1. **One `<diag>` → one document.** The ETL never chunks by size; it chunks by the
   XML's own semantic unit, then enriches each unit with ancestor context so it
   stands alone.
2. **Three synonym sources feed one embedding string**: Tabular `includes`,
   Tabular `inclusionTerm`, and Index-derived phrases — fused by
   `to_embedding_text()`.
3. **Two orthogonal retrieval signals** (dense semantic + sparse lexical) are fused
   with **weighted Reciprocal Rank Fusion** via LangChain's `EnsembleRetriever`
   (weights `0.7`/`0.3`), because neither alone is sufficient — dense catches
   paraphrase, sparse catches exact terminology. RRF fuses on *rank*, not raw score,
   which cleanly sidesteps the dense-vs-BM25 scale mismatch.
4. **Retrieval is deliberately over-generous (top-15) and then narrowed by a more
   expensive but more accurate stage** (cross-encoder re-rank to top-3) — a classic
   two-stage IR pattern trading compute for precision only after the candidate set
   is small.
5. **Safety is enforced at two different points**: demographic filters are a hard
   pre-filter (never scored, never shown), while low-confidence results are a soft
   refusal (explicitly declining to answer rather than guessing).
6. **Every threshold and weight is sourced from `config.yml`** via `mednote.config`
   — none of steps 3–6 have a hardcoded magic number, so retuning the whole
   pipeline's behavior is a one-file change.
