# NomadRight scheme intelligence (intent + scenario + rules + RAG)

A layer that answers welfare-scheme questions from a **structured, sourced
knowledge base** before the existing Decision Layer is consulted. The speech
front-end (ASR → NMT), the response back-end (NMT → TTS → speaker), the HDMI UI,
Qwen3-VL and Ollama are unchanged and shared.

```
 mic / keyboard ─► ASR (TensorRT Conformer) ─► NMT → English          (existing)
                                                  │
                                                  ▼
 WorkflowController.process()   ── Step 0 ──►  SchemeIntelligence.handle()
   │                                             ScenarioEngine   facts: age, work, origin/current state ...
   │                                             SessionState     profile kept across turns (TTL 10 min)
   │                                             IntentEngine     rules first, then e5 kNN classifier
   │                                             QueryRouter      DIRECT / ELIGIBILITY / DISCOVERY / COMPLEX / CLARIFY / DEFER
   │                                             SchemeRepository SQLite (scheme_intel.db), metadata filters
   │                                             RulesEngine      deterministic, 6 eligibility states
   │                                             RAGEngine        einsum over pre-computed e5 vectors; FAQ-question index
   │                                             QwenAdapter      COMPLEX only, grounded + sentinel-gated, ends with where to confirm
   │                                             ResponseComposer <= 40 words (<= 20 s of speech)
   │   answer ◄──────────────────────────────────┘
   │   None (DEFER / error) ─► legacy steps 1-8 (IntentRecognizer ... ResponseGenerator), unchanged
   ▼
 StructuredResponsePackage ─► NMT → worker's language ─► TTS (flite) ─► speaker / UI   (existing)

 Camera ─► frame downscaled to <= 1024 px ─► Qwen3-VL (a scheme question asks it to read the card)
        ─► facts from its description ─► SchemeIntelligence.handle_vision() ─► same back-end
        (reading not usable, or not a scheme question: Qwen answers the person's own question, as before)
```

## Switching it off / rollback

| What | How |
|---|---|
| Turn the new layer off (legacy pipeline exactly as before) | `export NOMADRIGHT_SCHEME_INTEL=0` before starting, or set `SCHEME_INTEL_ENABLED = False` in `scheme_intel/si_constants.py` |
| Layer missing its knowledge base | it logs a warning and the legacy pipeline answers everything |
| See what changed in existing files | `git diff refs/checkpoints/pre-intent-integration -- pocket-infer-sw/python/pocketinfer/master.py pocket-infer-sw/python/pocketinfer/applications/nomad_right/app.py pocket-infer-sw/python/pocketinfer/applications/nomad_right/workflow.py` (only these three existing files were edited; everything else is new). Do not diff whole directories against the checkpoint: files that are untracked today (document_crop.py, webview_launcher.py, models/asr.py ...) show up as "deleted" there although they are on disk unchanged. |
| Full code rollback | restore each of the three files with `git show refs/checkpoints/pre-intent-integration:<path> > <path>`, then remove `python/pocketinfer/applications/nomad_right/scheme_intel/` and `scheme_intel_test.py` (the knowledge base in `nomadright/scheme_intel/` can stay - nothing reads it once the code is gone) |

Nothing existing was deleted or overwritten: `nomadright_kb.db`, `chroma_db/`,
the scheme JSON files and every model file are untouched (the new DB lives here,
in `nomadright/scheme_intel/`).

## Knowledge base

Build (offline, ~2-3 min on the Jetson, e5 on CPU):

    python -m pocketinfer.applications.nomad_right.scheme_intel.build_kb   [--no-embeddings]

| Input (read-only) | Used for |
|---|---|
| `sources/gov_scheme_qa/` (copy of the intent project's canonical data, see its PROVENANCE.md) | 34 schemes from the official compilation "scheme 2.pdf": rules, benefits, documents, procedures, FAQs, authorities, exclusions, source chunks |
| `../pds.json, pmjay.json, eshram_osh.json, bocw.json, mgnregs.json` | the 5 existing rich records (myscheme.gov.in, nha.gov.in, eshram.gov.in, labour.gov.in, nrega.nic.in) - primary for those schemes |
| `../<code>.json` compact seeds | official URLs only |
| `sources/curated_overlay.json` | spoken names, 20-second summaries (each citing its source field), routing tags, legacy codes, extra rules derived from the rich records |
| `sources/intent_examples.json` + the intent dataset | labelled questions for the classifier |

Outputs: `scheme_intel.db` (SQLite, ~0.5 MB), `rag_embeddings.npy` (539 chunks ×
384, float16), `intent_prototypes.npz` (797 labelled questions), `faq_questions.npz`
(the FAQ questions embedded on their own), `manifest.json` (counts + source
hashes). Every output is written to a temp file and renamed.
Unknown facts are stored as NULL - nothing is invented.

35 schemes (34 offered in discovery): ONORC/PDS, Ayushman Bharat PM-JAY, e-Shram,
BOCW, MGNREGA (ended 30 Jun 2026) and its successor VB-G RAM G, PM-SYM,
NPS-Traders, PMJJBY, PMSBY, APY, PM-KISAN, PM Surya Ghar, PM SVANidhi,
PM Vishwakarma, PMUY, MUDRA, PMFBY, SSY, PMAY-G, PMJDY, PM-KMY, DDU-GKY, GKRY,
DAY, PMKVY, NSAP and its five components, Punjab MMSBY, HIS for weavers, SRMS.
Schemes people name that have **no verified data** (PMMVY, PMAY-Urban, UDID ...)
get an honest "not in my verified information" reply.

### SQLite schema (`scheme_intel.db`)

| Table | Columns |
|---|---|
| `schemes` | scheme_id PK, scheme_name, short_name, spoken_name, legacy_code, category, domains (JSON), description, voice_summary, benefits (JSON), target_beneficiaries (JSON), eligibility (JSON), age_min, age_max, income_limit, income_limit_period, occupation (JSON), gender, migration_support (0/1/NULL), state, district, documents (JSON), application_process (JSON), application_mode, portability, exceptions (JSON), official_source, official_url, helpline, last_verified, status, effective_from, effective_until, superseded_by, parent_scheme_id, is_listed, provenance (JSON) |
| `rules` | rule_id PK, scheme_id, rule_type (QUALIFICATION/EXCLUSION), logic (JSON tree), unless_logic, description, source, valid_from, valid_until |
| `benefits` | id, scheme_id, rank, text, value_inr, frequency, source |
| `documents` | id, scheme_id, rank, name, description, mandatory, source |
| `procedure_steps` / `procedure_meta` | steps per mode (online/offline); mode, processing_time, fees |
| `exceptions` | id, scheme_id, kind (exclusion/exception/limitation), text, source |
| `faqs`, `contacts` | question/answer; helpline / portal / grievance / state_portal |
| `aliases` | alias, scheme_id, strength (1.0 explicit name, 0.5-0.6 generic phrase) |
| `chunks` | row (= row in rag_embeddings.npy), chunk_id, scheme_id, section, text, source |
| `relationships`, `meta` | scheme links; build info, verify-fields, domain primaries, unverified names |

## Intents

SCHEME_DISCOVERY, ELIGIBILITY_CHECK, BENEFIT_INFORMATION, APPLICATION_PROCEDURE,
REQUIRED_DOCUMENTS, RATION_ACCESS, HEALTH_SUPPORT, WORKER_SUPPORT,
PENSION_SUPPORT, MATERNITY_SUPPORT, EDUCATION_SUPPORT, HOUSING_SUPPORT,
DISABILITY_SUPPORT, FINANCIAL_SUPPORT, MIGRANT_SUPPORT, APPLICATION_STATUS,
GRIEVANCE, DOCUMENT_HELP, GENERAL_SCHEME_INFORMATION, OTHER.

Rules first (question-type patterns × topic lexicon × scheme aliases); only if
they are inconclusive a k-nearest-neighbour vote over e5 embeddings of labelled
questions (scheme names masked) decides. Qwen is never used for intent.

## Scenario fields

age, gender, occupation, employment_status, monthly_income, annual_income,
family_size, marital_status, pregnancy_status, disability_status, current_state,
current_district, current_city, origin_state, origin_district, migration_status,
ration_card, nfsa_beneficiary, aadhaar, e_shram_registration,
existing_scheme_membership - plus the facts the rules need (trade, bank account,
income-tax payer, PF/ESI cover, BPL, ration-card type/state, Aadhaar seeding,
construction days, land, housing, daughters' ages ...). Only stated facts are
recorded; "I am from Bihar and working in Chennai" → origin_state=Bihar,
current_city=Chennai, current_state=Tamil Nadu, migration_status=INTER_STATE_MIGRANT.

## Eligibility states

| State | Meaning |
|---|---|
| ELIGIBLE | every condition known true, every exclusion known false |
| LIKELY_ELIGIBLE | core conditions true; only verification items unconfirmed (Aadhaar seeding, bank account, "not in PF/ESI" ...) |
| POTENTIALLY_ELIGIBLE | nothing fails, a core fact is unknown, and the scheme is for this person's work or something already holds |
| INSUFFICIENT_INFORMATION | the facts that define who the scheme is for are unknown |
| INELIGIBLE | a condition fails, an exclusion applies, or the scheme has ended |
| UNKNOWN | the knowledge base has no rules for the scheme |

An unknown fact never satisfies a condition. Answers never say "definitely".

## Retrieval

Metadata first (scheme ids / sections from SQLite and the rules), then one
`einsum` over the pre-computed chunk vectors - not a BLAS matmul, which fought
PyTorch's still-spinning threads right after each e5 encode (15.8 ms against
0.17 ms, measured). A specific question about a named scheme ("when does the
Sukanya account mature?") and every complex question are first matched
question-to-question against the FAQ questions (`FAQ_MATCH_MIN_SCORE` in
si_constants); a matching FAQ's own answer is read out unless the FAQ names
places the person did not. Qwen is reached only for complex questions without
such a match, and its answer ends with the scheme's portal or helpline.

## Response budget

flite (the voices bhashini_models uses) measured on this device: 0.392 s per
English word after translation to Hindi, 0.437 s for Tamil → 20 s ≈ 45 words.
Answers are capped at 40 words and cut only at sentence/clause boundaries.

## Diagnostics and tests

    python -m pocketinfer.applications.nomad_right.scheme_intel.diagnostics [--selftest] [--no-embedder] [--speech] [--json]
    python python/pocketinfer/applications/nomad_right/scheme_intel_test.py          # offline unit tests
    NOMADRIGHT_SI_HEAVY=1 python .../scheme_intel_test.py                           # + WorkflowController integration

The app records P50/P95/P99 for asr, nmt, decision, si_*, tts, vision and
end_to_end into a bounded window and writes `/tmp/nomad_right_logs/latency_snapshot.json`
every few turns; `diagnostics` prints it together with memory, swap, process RSS
and the loaded Ollama runner.
