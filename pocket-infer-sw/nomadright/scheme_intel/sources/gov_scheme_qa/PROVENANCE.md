# Provenance of this folder (read-only source snapshot)

Copied verbatim (cp -a, no edits) on 2026-09-11 from:

    /home/stark-x/intent/scheme-solutions/gov-scheme-qa/data/                      -> data/
    /home/stark-x/intent/scheme-solutions/gov-scheme-qa/dataset/intent_dataset.json -> dataset/intent_dataset.json

Source repository: /home/stark-x/intent/scheme-solutions (git commit 8982cece86b23d35ff65a3488a896bb1a35c62b9).
The canonical JSON was extracted by that project from the official scheme
compilation "scheme 2.pdf" (/home/stark-x/intent/scheme-solutions/scheme 2.pdf,
261 pages, PDF CreationDate 2026-09-10). Page numbers inside these files
(source_pages / citation) come from that project's extraction and do not
always match the current PDF's pagination - treat them as the dataset's own
references, not verified page numbers.

Only the data was copied. The project's FastAPI app, static UI, trained
joblib models (pickled with scikit-learn 1.9.0, incompatible with this
venv's 1.7.2) and its SQLite DB are intentionally NOT used.

Rebuild the NomadRight scheme-intelligence knowledge base from this folder
(plus the existing ../../*.json scheme files, read-only) with:

    python -m pocketinfer.applications.nomad_right.scheme_intel.build_kb
