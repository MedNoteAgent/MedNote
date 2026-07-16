# Clinical Documentation Guidelines

Curated from public clinical references (see `SOURCES.md`) for the MedNote Scribe
RAG knowledge base. Each section is self-contained and chunked by heading at
indexing time (`doc_type: "guideline"`).

## SOAP Note Structure

A SOAP note records an outpatient encounter in four sections:

- **Subjective** — the patient's own report: chief complaint, symptom history
  (onset, duration, character, aggravating/relieving factors), relevant past
  medical history, and medications as stated by the patient.
- **Objective** — measurable findings: vital signs (blood pressure, heart rate,
  temperature, respiratory rate, SpO2), physical examination findings, and any
  point-of-care test results actually obtained during the visit.
- **Assessment** — the clinician's working impression. In assisted
  documentation this section must contain *suggested differentials for
  physician review*, never an asserted final diagnosis.
- **Plan** — next steps grounded in the encounter: investigations ordered,
  treatments discussed, referrals, safety-netting advice, and follow-up
  interval.

Only information present in the encounter (transcript) belongs in the note.
Missing information should be recorded as "not documented," never inferred.

## Documentation Quality Standards

- Record symptoms with timing and laterality when stated ("left ear pain, 2
  days"), because ICD-10-CM codes encode both.
- Distinguish patient-reported statements (Subjective) from clinician-observed
  findings (Objective); do not promote a reported symptom to a finding.
- Quantify vitals exactly as measured; do not round or normalize.
- Every medication mention must preserve the exact drug name and dose stated in
  the encounter. A dose that was not stated must not be added at documentation
  time — dosing is a prescribing decision, not a documentation task.
- Notes are drafts until the treating physician reviews and signs. A note must
  never be marked final or saved to the record without explicit physician
  confirmation.

## Red-Flag Symptom Combinations Requiring Urgent Escalation

The following presentations require immediate escalation to in-person emergency
evaluation. When detected, escalate FIRST; routine documentation is secondary.

- **Chest pain with radiation to the arm, jaw, shoulder, or back**, especially
  with diaphoresis, nausea, or dyspnea — treat as acute coronary syndrome until
  excluded. Do not document as routine musculoskeletal pain.
- **Sudden "worst-ever" (thunderclap) headache**, particularly with neck
  stiffness, vomiting, or altered consciousness — possible subarachnoid
  hemorrhage or meningitis; same-day emergency imaging is indicated.
- **Acute shortness of breath** with chest pain, hemoptysis, unilateral leg
  swelling, or recent immobilization — evaluate for pulmonary embolism.
- **Sudden focal neurological deficit** — unilateral weakness or numbness,
  facial droop, or speech disturbance — treat as stroke; time-critical.
- **Fever with petechial rash, or fever with severe headache and neck
  stiffness** — possible meningococcal disease.

Escalation documentation should state the triggering symptom combination, the
recommendation given, and that routine note-taking was deferred pending
physician review.

## ICD-10-CM Coding Specificity Conventions

- Code to the **highest level of specificity** the documentation supports. A
  three-character category (e.g. H66, Otitis media) is only valid when no
  further subdivision exists; otherwise use the most specific child.
- **Laterality:** many code families distinguish right / left / bilateral /
  unspecified ear, eye, or limb (e.g. H66.91 right, H66.92 left, H66.93
  bilateral). When the encounter states a side, the sided code must be used;
  "unspecified" laterality codes signal documentation gaps to payers.
- **"Unspecified" codes** are legitimate when the encounter genuinely does not
  establish the detail, but they should prompt a specificity check: if a parent
  or unspecified code is selected and more specific children exist, surface the
  children for physician selection.
- **Headache coding example:** R51.9 (Headache, unspecified) is a symptom code;
  G44.2 (Tension-type headache) is the diagnosis family, subdivided into
  G44.20 unspecified, G44.21 episodic, and G44.22 chronic tension-type
  headache. Recurrent or episodic tension headache documentation supports the
  G44.2- family rather than the bare symptom code.
- **Sequencing:** follow "code first" and "use additional code" notes from the
  Tabular List; an Excludes1 note means the excluded condition is never coded
  together with the listed code.
- Every suggested code must cite the coding reference it came from and remain
  pending physician confirmation — codes drive billing and must not be
  fabricated or guessed.

## Non-Diagnostic Language for Assisted Documentation

Documentation assistants provide decision support, not diagnoses. The final
diagnostic determination always belongs to the treating physician.

- Frame assessments as possibilities: "may be consistent with," "consider,"
  "possible," "differential includes."
- Avoid assertive phrasing: "the patient has," "diagnosis is," "confirmed,"
  "this is clearly."
- Mark every suggested differential and every suggested ICD-10 code as
  "for physician review" / "pending physician confirmation."
- When asked to provide a definitive diagnosis, decline and offer suggested
  differentials as decision support only.
- When the encounter lacks sufficient information for a suggestion, say so
  explicitly ("insufficient data to suggest a code — assign manually") rather
  than inferring.

## Visit Continuity and Prior-Note Review

- Before drafting a follow-up note, review the patient's most recent prior
  visit summary: chief complaint, assessment suggestions, and plan.
- Reference relevant prior findings explicitly ("BP 130/85 at last visit,
  138/88 today") so trends are visible to the reviewing physician.
- Retrieved history is context, not carried-forward truth: re-document current
  symptoms from the current encounter rather than copying prior text ("note
  cloning"), which propagates stale findings and is an audit risk.
