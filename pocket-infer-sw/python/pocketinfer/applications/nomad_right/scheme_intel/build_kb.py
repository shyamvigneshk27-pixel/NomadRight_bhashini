"""
Offline builder for the scheme-intelligence knowledge base.

    python -m pocketinfer.applications.nomad_right.scheme_intel.build_kb [--no-embeddings]

Inputs (all read-only):
  * sources/gov_scheme_qa/data/*       canonical data extracted from the official
                                       scheme compilation (copied into this repo)
  * ../pds.json, pmjay.json, eshram_osh.json, bocw.json, mgnregs.json
                                       the existing NomadRight scheme files
  * ../<code>.json compact seeds       official URLs only
  * sources/curated_overlay.json       spoken names, summaries, routing metadata
  * sources/intent_examples.json + gov_scheme_qa/dataset/intent_dataset.json

Outputs (new files next to the inputs - the existing nomadright_kb.db and
chroma_db are never opened):
  * scheme_intel.db          SQLite knowledge base
  * rag_embeddings.npy       e5 passage embeddings, one row per chunk (float16)
  * intent_prototypes.npz    e5 embeddings of labelled example questions
  * manifest.json            counts, source hashes, build time

Every output is written to a temporary file and renamed into place, so a
failed build never leaves a half-written knowledge base behind.
Nothing is invented: unknown values stay NULL.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel.scenario_engine import normalize

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE schemes (
    scheme_id            TEXT PRIMARY KEY,
    scheme_name          TEXT NOT NULL,
    short_name           TEXT NOT NULL,
    spoken_name          TEXT NOT NULL,
    legacy_code          TEXT,
    category             TEXT,
    ministry             TEXT,
    department           TEXT,
    implementing_agency  TEXT,
    domains              TEXT NOT NULL DEFAULT '[]',
    description          TEXT,
    voice_summary        TEXT,
    benefits             TEXT NOT NULL DEFAULT '[]',
    target_beneficiaries TEXT,
    eligibility          TEXT NOT NULL DEFAULT '[]',
    age_min              INTEGER,
    age_max              INTEGER,
    income_limit         INTEGER,
    income_limit_period  TEXT,
    occupation           TEXT NOT NULL DEFAULT '[]',
    gender               TEXT,
    migration_support    INTEGER,
    state                TEXT,
    district             TEXT,
    documents            TEXT NOT NULL DEFAULT '[]',
    application_process  TEXT NOT NULL DEFAULT '[]',
    application_mode     TEXT,
    portability          TEXT,
    exceptions           TEXT NOT NULL DEFAULT '[]',
    official_source      TEXT,
    official_url         TEXT,
    helpline             TEXT,
    last_verified        TEXT,
    status               TEXT,
    effective_from       TEXT,
    effective_until      TEXT,
    superseded_by        TEXT,
    parent_scheme_id     TEXT,
    is_listed            INTEGER NOT NULL DEFAULT 1,
    provenance           TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE benefits (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, rank INTEGER NOT NULL,
    text TEXT NOT NULL, value_inr REAL, frequency TEXT, source TEXT);
CREATE TABLE documents (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, rank INTEGER NOT NULL,
    name TEXT NOT NULL, description TEXT, mandatory INTEGER NOT NULL, source TEXT);
CREATE TABLE procedure_steps (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, mode TEXT NOT NULL,
    step_no INTEGER NOT NULL, text TEXT NOT NULL, source TEXT);
CREATE TABLE procedure_meta (scheme_id TEXT PRIMARY KEY, application_mode TEXT, processing_time TEXT,
    fees TEXT, source TEXT);
CREATE TABLE exceptions (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, kind TEXT NOT NULL,
    text TEXT NOT NULL, source TEXT);
CREATE TABLE faqs (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, question TEXT NOT NULL,
    answer TEXT NOT NULL, source TEXT);
CREATE TABLE contacts (id INTEGER PRIMARY KEY, scheme_id TEXT NOT NULL, kind TEXT NOT NULL,
    value TEXT NOT NULL, description TEXT, source TEXT);
CREATE TABLE rules (rule_id TEXT PRIMARY KEY, scheme_id TEXT NOT NULL, rule_type TEXT NOT NULL,
    logic TEXT NOT NULL, unless_logic TEXT, description TEXT, source TEXT, valid_from TEXT, valid_until TEXT);
CREATE TABLE aliases (alias TEXT NOT NULL, scheme_id TEXT NOT NULL, strength REAL NOT NULL,
    PRIMARY KEY (alias, scheme_id));
CREATE TABLE chunks (row INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, scheme_id TEXT NOT NULL,
    section TEXT, text TEXT NOT NULL, source TEXT);
CREATE TABLE relationships (id INTEGER PRIMARY KEY, source_scheme_id TEXT, target_scheme_id TEXT,
    relationship_type TEXT, description TEXT);
CREATE INDEX idx_benefits_scheme ON benefits(scheme_id);
CREATE INDEX idx_documents_scheme ON documents(scheme_id);
CREATE INDEX idx_steps_scheme ON procedure_steps(scheme_id);
CREATE INDEX idx_rules_scheme ON rules(scheme_id);
CREATE INDEX idx_chunks_scheme ON chunks(scheme_id);
"""

# Compact seed records in the existing data folder - only their official URL is used.
SEED_FILES = {
    "SCH_APY": "apy.json", "SCH_NSAP": "nsap.json", "SCH_PMKISAN": "pm_kisan.json",
    "SCH_PMSURYA": "pm_surya_ghar.json", "SCH_SVANIDHI": "pm_svanidhi.json", "SCH_PMSYM": "pm_sym.json",
    "SCH_VISHWA": "pm_vishwakarma.json", "SCH_PMAYG": "pmay_g.json", "SCH_PMFBY": "pmfby.json",
    "SCH_PMJDY": "pmjdy.json", "SCH_PMJJBY": "pmjjby.json", "SCH_PMMY": "pmmy.json",
    "SCH_PMSBY": "pmsby.json", "SCH_PMUY": "pmuy.json", "SCH_SSY": "sukanya_samriddhi.json",
}

# Canonical rule fields -> Profile fields (see scenario_engine / rules_engine.derive).
FIELD_MAP = {
    "demographic.citizenship": "citizenship", "demographic.age": "age", "demographic.gender": "gender",
    "demographic.marital_status": "marital_status", "demographic.breadwinner_age": "breadwinner_age",
    "economic.monthly_income_inr": "monthly_income", "economic.is_income_tax_payer": "income_tax_payer",
    "economic.annual_turnover_inr": "annual_turnover", "economic.monthly_pension_inr": "monthly_pension",
    "economic.is_bpl": "is_bpl", "economic.is_from_poor_household": "poor_household",
    "institutional.covered_by_epfo": "epfo_esic_member", "institutional.covered_by_esic": "epfo_esic_member",
    "institutional.covered_by_nps_govt": "nps_govt_member", "institutional.enrolled_in_pmsym": "pmsym_member",
    "institutional.has_bank_account": "has_bank_account", "institutional.has_savings_bank_account": "has_bank_account",
    "institutional.has_aadhaar": "aadhaar", "institutional.has_ration_card": "ration_card",
    "institutional.is_aadhaar_seeded_in_ration_card": "aadhaar_seeded_ration",
    "institutional.is_constitutional_post_holder": "constitutional_post",
    "institutional.has_electricity_connection": "has_electricity_connection",
    "institutional.has_cov_or_id_card": "cov_or_id_card", "institutional.is_in_survey_list": "in_survey_list",
    "institutional.has_lor_from_ulb": "lor_from_ulb",
    "institutional.is_govt_employee_or_family": "govt_employee_or_family",
    "institutional.has_family_member_availed_vishwa": "family_availed_vishwa",
    "institutional.availed_pmegp": "availed_pmegp",
    "institutional.has_outstanding_svanidhi_loan": "outstanding_svanidhi_loan",
    "institutional.has_outstanding_mudra_loan": "outstanding_mudra_loan",
    "institutional.has_existing_lpg_connection_in_household": "lpg_connection",
    "institutional.in_secc_2011_data": "in_secc_2011", "institutional.has_smart_ration_card": "smart_ration_card",
    "institutional.is_jform_farmer": "jform_farmer", "institutional.is_small_trader_registered": "small_trader_registered",
    "institutional.is_accredited_journalist": "accredited_journalist",
    "institutional.is_bocw_registered_worker": "bocw_member", "institutional.enrolled_in_ignoaps": "ignoaps_member",
    "occupational.construction_days_in_last_12_months": "construction_days_12m",
    "occupational.willing_for_unskilled_manual_work": "willing_manual_work",
    "occupational.vending_as_of_march_2020": "vending_since_march_2020",
    "occupational.is_central_state_govt_employee": "govt_employee",
    "occupational.is_registered_professional": "registered_professional",
    "occupational.artisan_trade": "trade",
    "geographic.area_type": "area_type", "geographic.state": "residence_state",
    "housing.roof_type": "roof_type", "housing.secc_housing_deprivation": "housing_deprivation",
    "housing.in_awasplus_list": "in_awasplus_list",
    "agriculture.owns_cultivable_land": "owns_land", "agriculture.land_is_in_family_name": "land_in_family_name",
    "agriculture.land_area_ha": "land_area_ha", "agriculture.is_institutional_landholder": "institutional_landholder",
    "agriculture.is_small_marginal_farmer": "small_marginal_farmer", "agriculture.farmer_type": "farmer_type",
    "agriculture.is_tribal_forest_dweller": "tribal_forest_dweller",
    "agriculture.cultivates_community_land": "community_land",
    "agriculture.has_insurable_interest_in_crop": "insurable_interest", "agriculture.crop_is_notified": "crop_notified",
    "disability.percentage": "disability_percentage", "family.existing_ssy_accounts_count": "ssy_accounts_count",
}
# Last-segment names used in the exclusions file's rule_condition strings.
LAST_SEGMENT_MAP = {k.split(".")[-1]: v for k, v in FIELD_MAP.items()}
LAST_SEGMENT_MAP.update({"is_income_tax_payer": "income_tax_payer", "age": "age",
                         "monthly_income": "monthly_income", "monthly_income_inr": "monthly_income"})

# Canonical aliases that name a kind of person rather than a scheme.
ALIAS_BLOCKLIST = {"construction workers", "construction worker", "workers", "farmers", "his", "day"}


def _log(msg: str) -> None:
    print(f"[build_kb] {msg}", flush=True)


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _clean(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (dict, list)):
        text = "; ".join(f"{k}: {v}" for k, v in text.items()) if isinstance(text, dict) else "; ".join(map(str, text))
    text = re.sub(r"\s+", " ", str(text)).strip()
    return "" if text.upper() == "UNKNOWN" else text


def _pages(pages: Optional[List[int]]) -> str:
    return ("p." + ",".join(str(p) for p in pages[:4])) if pages else ""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _split_words(text: str, max_words: int = 90) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], []
    for s in sentences:
        sw = s.split()
        if cur and len(cur) + len(sw) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.extend(sw)
        while len(cur) > max_words:
            out.append(" ".join(cur[:max_words]))
            cur = cur[max_words:]
    if cur:
        out.append(" ".join(cur))
    return out


def _parse_value(raw: str) -> Any:
    raw = raw.strip().strip("'\"")
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


class Builder:
    def __init__(self, with_embeddings: bool = True):
        self.with_embeddings = with_embeddings
        self.overlay = _load(C.CURATED_OVERLAY_PATH)
        self.today = datetime.now().date().isoformat()
        self.sources: Dict[str, str] = {}
        self.warnings: List[str] = []

    # ── sources ───────────────────────────────────────────────────────────
    def _src(self, path: str) -> Any:
        self.sources[os.path.relpath(path, C.KB_DIR)] = _sha(path)
        return _load(path)

    def load_sources(self) -> None:
        cd = C.CANONICAL_DIR
        self.can = {name: self._src(os.path.join(cd, name, f"all_{name}.json"))
                    for name in ("schemes", "benefits", "documents", "procedures", "faqs", "authorities",
                                 "exclusions", "rules", "relationships", "temporal")}
        self.can["chunks"] = self._src(os.path.join(cd, "source_chunks", "all_chunks.json"))
        self.can_schemes = {s["scheme_id"]: s for s in self.can["schemes"]}
        self.legacy: Dict[str, Dict[str, Any]] = {}
        self.seed: Dict[str, Dict[str, Any]] = {}
        for sid, ov in self.overlay["schemes"].items():
            if ov.get("legacy_json"):
                self.legacy[sid] = self._src(os.path.join(C.LEGACY_SCHEME_DIR, ov["legacy_json"]))
            if sid in SEED_FILES and os.path.exists(os.path.join(C.LEGACY_SCHEME_DIR, SEED_FILES[sid])):
                self.seed[sid] = self._src(os.path.join(C.LEGACY_SCHEME_DIR, SEED_FILES[sid]))
        self.merge_into = {sid: ov["merge_into"] for sid, ov in self.overlay["schemes"].items() if ov.get("merge_into")}
        _log(f"sources: {len(self.can_schemes)} canonical schemes, {len(self.legacy)} legacy files, "
             f"{len(self.seed)} seed files")

    def _sid(self, sid: str) -> Optional[str]:
        sid = self.merge_into.get(sid, sid)
        return sid if sid in self.overlay["schemes"] and not self.overlay["schemes"][sid].get("merge_into") else None

    # ── child tables ──────────────────────────────────────────────────────
    def build_children(self) -> None:
        ov = self.overlay["schemes"]
        self.benefits, self.documents, self.steps, self.proc_meta = [], [], [], {}
        self.exceptions, self.faqs, self.contacts = [], [], []

        def add_benefit(sid, text, value=None, freq=None, source=""):
            text = _clean(text)
            if not text:
                return
            if any(bad in text for bad in ov[sid].get("exclude_benefit_contains", [])):
                return
            k = _key(text)
            if any(b["scheme_id"] == sid and _key(b["text"]) == k for b in self.benefits):
                return
            rank = sum(1 for b in self.benefits if b["scheme_id"] == sid)
            self.benefits.append({"scheme_id": sid, "rank": rank, "text": text,
                                  "value_inr": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                                  "frequency": freq, "source": source})

        def add_doc(sid, name, desc, mandatory, source):
            name = _clean(name)
            if not name:
                return
            k = _key(name)
            if any(d["scheme_id"] == sid and _key(d["name"]) == k for d in self.documents):
                return
            rank = sum(1 for d in self.documents if d["scheme_id"] == sid)
            self.documents.append({"scheme_id": sid, "rank": rank, "name": name, "description": _clean(desc),
                                   "mandatory": 1 if mandatory else 0, "source": source})

        def add_contact(sid, kind, value, desc, source):
            value = _clean(value)
            if not value or value.upper() == "UNKNOWN":
                return
            if any(c["scheme_id"] == sid and c["kind"] == kind and c["value"] == value for c in self.contacts):
                return
            self.contacts.append({"scheme_id": sid, "kind": kind, "value": value, "description": _clean(desc),
                                  "source": source})

        # Legacy (existing NomadRight JSON) first for the schemes built on it.
        for sid, lj in self.legacy.items():
            for item in lj.get("benefits", []):
                add_benefit(sid, item if isinstance(item, str) else item.get("benefit"), source=_clean(
                    None if isinstance(item, str) else item.get("source")))
            for item in lj.get("financial_benefits", []):
                if isinstance(item, str):
                    add_benefit(sid, item, source=lj.get("scheme_id", ""))
                else:
                    amount = item.get("amount") or item.get("details")
                    add_benefit(sid, f"{_clean(item.get('benefit_type'))}: {_clean(amount)}", source=_clean(item.get("source")))
            for item in lj.get("required_documents", []):
                if isinstance(item, str):
                    add_doc(sid, item, "", True, lj.get("scheme_id", ""))
                else:
                    add_doc(sid, item.get("document"), item.get("description"), item.get("mandatory", True),
                            _clean(item.get("source")))
            app = lj.get("application_process", {})
            if isinstance(app, dict):
                for mode in ("offline", "online"):
                    steps = app.get(mode) or []
                    if isinstance(steps, str):
                        steps = [steps]
                    for i, st in enumerate(steps):
                        text = _clean(st if isinstance(st, str) else st.get("description"))
                        if text:
                            self.steps.append({"scheme_id": sid, "mode": mode, "step_no": i + 1, "text": text,
                                               "source": _clean(None if isinstance(st, str) else st.get("source"))})
            modes = [m for m in ("online", "offline") if (app.get(m) if isinstance(app, dict) else None)]
            self.proc_meta[sid] = {"scheme_id": sid,
                                   "application_mode": "ONLINE_AND_OFFLINE" if len(modes) == 2 else (modes[0].upper() if modes else None),
                                   "processing_time": None, "fees": None, "source": lj.get("scheme_id", "")}
            for kind, key, field in (("exception", "exceptions", "exception"), ("limitation", "limitations", "limitation")):
                for item in lj.get(key, []):
                    text = _clean(item if isinstance(item, str) else item.get(field))
                    if text:
                        self.exceptions.append({"scheme_id": sid, "kind": kind, "text": text,
                                                "source": _clean(None if isinstance(item, str) else item.get("source"))})
            for item in lj.get("faq", []):
                if isinstance(item, dict) and item.get("question") and item.get("answer"):
                    self.faqs.append({"scheme_id": sid, "question": _clean(item["question"]),
                                      "answer": _clean(item["answer"]), "source": _clean(item.get("source"))})
            for item in lj.get("official_contacts", []):
                t = (item.get("type") or "").lower()
                kind = "helpline" if "helpline" in t else ("grievance" if "grievance" in t else "portal")
                add_contact(sid, kind, item.get("number") or item.get("url"), item.get("description"), _clean(item.get("source")))
            for item in lj.get("state_portals", []):
                add_contact(sid, "state_portal", item.get("portal"), item.get("state"), lj.get("scheme_id", ""))

        # Canonical data (official compilation).
        for b in self.can["benefits"]:
            sid = self._sid(b["scheme_id"])
            if sid:
                val = b.get("value_inr")
                add_benefit(sid, b.get("description"), val if not isinstance(val, list) else None,
                            b.get("frequency"), "canonical " + _pages(b.get("source_pages")))
        for d in self.can["documents"]:
            sid = self._sid(d["scheme_id"])
            if not sid:
                continue
            src = "canonical " + _pages(d.get("source_pages"))
            for m in d.get("mandatory", []):
                add_doc(sid, m.get("name") if isinstance(m, dict) else m, m.get("description") if isinstance(m, dict) else "", True, src)
            for m in d.get("optional", []):
                add_doc(sid, m.get("name") if isinstance(m, dict) else m, m.get("description") if isinstance(m, dict) else "", False, src)
        for p in self.can["procedures"]:
            sid = self._sid(p["scheme_id"])
            if not sid:
                continue
            src = "canonical " + _pages(p.get("source_pages"))
            have_steps = any(s["scheme_id"] == sid for s in self.steps)
            if not have_steps:
                for mode, key in (("offline", "offline_steps"), ("online", "online_steps")):
                    for i, st in enumerate(p.get(key) or []):
                        if _clean(st):
                            self.steps.append({"scheme_id": sid, "mode": mode, "step_no": i + 1, "text": _clean(st), "source": src})
            meta = self.proc_meta.get(sid, {"scheme_id": sid})
            meta.update({"application_mode": p.get("mode") or meta.get("application_mode"),
                         "processing_time": _clean(p.get("processing_time")) or meta.get("processing_time"),
                         "fees": _clean(p.get("fees")) or meta.get("fees"), "source": src})
            self.proc_meta[sid] = meta
        for x in self.can["exclusions"]:
            sid = self._sid(x["scheme_id"])
            if sid:
                self.exceptions.append({"scheme_id": sid, "kind": "exclusion", "text": _clean(x["description"]),
                                        "source": "canonical " + _pages(x.get("source_pages"))})
        for f in self.can["faqs"]:
            sid = self._sid(f["scheme_id"])
            if sid:
                self.faqs.append({"scheme_id": sid, "question": _clean(f["question"]), "answer": _clean(f["answer"]),
                                  "source": "canonical " + _pages(f.get("source_pages"))})
        for a in self.can["authorities"]:
            sid = self._sid(a["scheme_id"])
            if not sid:
                continue
            src = "canonical " + _pages(a.get("source_pages"))
            add_contact(sid, "helpline", a.get("helpline"), "Helpline", src)
            add_contact(sid, "portal", a.get("portal_url"), "Official portal", src)
            add_contact(sid, "grievance", a.get("grievance_redressal"), "Grievance redressal", src)
        for s in self.can["schemes"]:
            sid = self._sid(s["scheme_id"])
            if not sid:
                continue
            if s.get("helpline"):
                add_contact(sid, "helpline", s["helpline"], "Helpline", "canonical " + _pages(s.get("source_pages")))
            portal = s.get("portal") or ""
            if isinstance(portal, str) and re.search(r"\.(?:gov|nic|org)\.in|\.in\b", portal):
                add_contact(sid, "portal", portal, "Official portal", "canonical " + _pages(s.get("source_pages")))
        _log(f"children: {len(self.benefits)} benefits, {len(self.documents)} documents, {len(self.steps)} steps, "
             f"{len(self.exceptions)} exceptions, {len(self.faqs)} faqs, {len(self.contacts)} contacts")

    # ── rules ─────────────────────────────────────────────────────────────
    def _map_leaf(self, cond: Dict[str, Any], patches: Dict[str, Any]) -> Dict[str, Any]:
        f, op, v = cond.get("field", ""), cond.get("operator", "=="), cond.get("value")
        base = {"desc": _clean(cond.get("description")), "source": _clean(cond.get("citation"))}
        if f == "occupational.occupation_type":
            if op == "==" and v == "UNORGANISED_WORKER":
                return dict(base, field="is_unorganised_worker", op="==", value=True)
            return dict(base, field="occupation", op=op, value=v)
        leaf = dict(base, field=FIELD_MAP.get(f, f.split(".")[-1]), op=op, value=v)
        soft = patches.get("soft_if")
        if soft and leaf["field"] == soft["field"]:
            leaf["soft_if_occupation"] = soft["when_occupation_in"]
        return leaf

    def _map_node(self, cond: Dict[str, Any], patches: Dict[str, Any]) -> Dict[str, Any]:
        if "conditions" in cond:
            node = {"op": cond.get("operator", "AND").upper(), "desc": _clean(cond.get("description")),
                    "items": [self._map_node(c, patches) for c in cond["conditions"]]}
            if patches.get("group_kind"):
                node["kind"] = patches["group_kind"]
            return node
        return self._map_leaf(cond, patches)

    def build_rules(self) -> None:
        self.rules = []
        patches_all = self.overlay.get("rule_patches", {})
        for r in self.can["rules"]:
            sid = self._sid(r["scheme_id"])
            if not sid or r.get("rule_type") == "SPECIAL_CONDITION":
                continue
            patches = patches_all.get(r["rule_id"], {})
            conds = list(r.get("conditions", []))
            if "replace_fields" in patches:   # SSY: the girl child's age/gender, not the applicant's
                fields = {"demographic.gender", "demographic.age"}
                cit = next((c.get("citation", "") for c in conds if c.get("field") in fields), "")
                conds = [c for c in conds if c.get("field") not in fields]
                conds.insert(0, {"field": "girl_child_under_10", "operator": "==", "value": True,
                                 "description": "Has a daughter below 10 years", "citation": cit})
            items = []
            for c in conds:
                if c.get("field") == "girl_child_under_10":
                    items.append({"field": "girl_child_under_10", "op": "==", "value": True,
                                  "desc": c["description"], "source": c.get("citation", "")})
                else:
                    items.append(self._map_node(c, patches))
            self.rules.append({
                "rule_id": r["rule_id"], "scheme_id": sid, "rule_type": r["rule_type"],
                "logic": {"op": r.get("operator", "AND").upper(), "items": items}, "unless_logic": None,
                "description": _clean(r.get("description")),
                "source": "canonical " + _pages(r.get("source_pages")),
                "valid_from": r.get("valid_from"), "valid_until": r.get("valid_until"),
            })
        for r in self.overlay.get("extra_rules", []):
            def conv(n):
                if "items" in n:
                    return {"op": n.get("op", "AND").upper(), "desc": n.get("desc", ""), "items": [conv(c) for c in n["items"]]}
                return {k: n[k] for k in ("field", "op", "value", "desc", "source") if k in n}
            op = "OR" if r["rule_type"] == "EXCLUSION" else "AND"
            self.rules.append({
                "rule_id": r["rule_id"], "scheme_id": r["scheme_id"], "rule_type": r["rule_type"],
                "logic": {"op": op, "items": [conv(c) for c in r["conditions"]]},
                "unless_logic": conv(r["unless"]) if r.get("unless") else None,
                "description": r.get("description", ""), "source": r.get("source", "curated_overlay extra_rules"),
                "valid_from": None, "valid_until": None,
            })
        # Exclusions file: add only (scheme, field) pairs no rule already covers.
        covered = set()

        def walk(sid, node):
            if "items" in node:
                for c in node["items"]:
                    walk(sid, c)
            else:
                covered.add((sid, node["field"]))
        for r in self.rules:
            walk(r["scheme_id"], r["logic"])
        for x in self.can["exclusions"]:
            sid = self._sid(x["scheme_id"])
            m = re.match(r"^\s*([\w.]+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$", x.get("rule_condition") or "")
            if not sid or not m:
                continue
            field = FIELD_MAP.get(m.group(1)) or LAST_SEGMENT_MAP.get(m.group(1).split(".")[-1])
            if not field or (sid, field) in covered:
                continue
            covered.add((sid, field))
            self.rules.append({
                "rule_id": f"EXCL_FILE_{x['exclusion_id']}", "scheme_id": sid, "rule_type": "EXCLUSION",
                "logic": {"op": "OR", "items": [{"field": field, "op": m.group(2), "value": _parse_value(m.group(3)),
                                                  "desc": _clean(x["description"]),
                                                  "source": "canonical " + _pages(x.get("source_pages"))}]},
                "unless_logic": None, "description": _clean(x["description"]),
                "source": "canonical exclusions " + _pages(x.get("source_pages")), "valid_from": None, "valid_until": None,
            })
        _log(f"rules: {len(self.rules)} ({sum(1 for r in self.rules if r['rule_type'] == 'EXCLUSION')} exclusions)")

    # ── scheme records ────────────────────────────────────────────────────
    def _rule_bounds(self, sid: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
        age_min = age_max = income = gender = None
        for r in self.rules:
            if r["scheme_id"] != sid or r["rule_type"] != "QUALIFICATION":
                continue
            if r.get("valid_until") and r["valid_until"] < self.today:
                continue
            for n in r["logic"]["items"]:
                if "items" in n:
                    continue
                if n["field"] == "age" and n["op"] in (">=", ">"):
                    v = int(n["value"]) + (1 if n["op"] == ">" else 0)
                    age_min = v if age_min is None else max(age_min, v)
                elif n["field"] == "age" and n["op"] in ("<=", "<"):
                    v = int(n["value"]) - (1 if n["op"] == "<" else 0)
                    age_max = v if age_max is None else min(age_max, v)
                elif n["field"] == "monthly_income" and n["op"] in ("<=", "<"):
                    income = int(n["value"])
                elif n["field"] == "gender" and n["op"] == "==":
                    gender = n["value"]
        return age_min, age_max, income, gender

    def build_schemes(self) -> None:
        self.schemes = []
        for sid, ov in self.overlay["schemes"].items():
            if ov.get("merge_into"):
                continue
            can = self.can_schemes.get(sid) if "canonical" in ov["base"] else None
            lj = self.legacy.get(sid)
            seed = self.seed.get(sid)
            if not can and not lj:
                self.warnings.append(f"{sid}: no source record - skipped")
                continue
            age_min, age_max, income, gender = self._rule_bounds(sid)
            if sid == "SCH_SSY":
                gender = "FEMALE"  # the account holder is the girl child (RULE_SSY_ELIG)
            elig = []
            if lj:
                elig += [_clean(e if isinstance(e, str) else e.get("criterion")) for e in lj.get("eligibility", [])]
            for r in self.rules:
                if r["scheme_id"] == sid and r["rule_type"] == "QUALIFICATION" and not (r.get("valid_until") and r["valid_until"] < self.today):
                    for n in r["logic"]["items"]:
                        if n.get("desc") and _key(n["desc"]) not in {_key(e) for e in elig}:
                            elig.append(n["desc"])
            elig = [e for e in elig if e]
            docs = [d for d in self.documents if d["scheme_id"] == sid]
            steps = [s for s in self.steps if s["scheme_id"] == sid]
            excs = [e["text"] for e in self.exceptions if e["scheme_id"] == sid]
            bens = [b["text"] for b in self.benefits if b["scheme_id"] == sid]
            port = None
            if lj and lj.get("portability"):
                p = lj["portability"]
                port = " ".join(_clean(v) for k, v in p.items() if k != "source" and isinstance(v, str)) if isinstance(p, dict) else _clean(p)
            elif any(b.get("benefit_type") == "PDS_PORTABILITY" for b in self.can["benefits"] if self._sid(b["scheme_id"]) == sid):
                port = next(_clean(b["description"]) for b in self.can["benefits"]
                            if self._sid(b["scheme_id"]) == sid and b.get("benefit_type") == "PDS_PORTABILITY")
            helplines = [c["value"] for c in self.contacts if c["scheme_id"] == sid and c["kind"] == "helpline"]
            portals = [c["value"] for c in self.contacts if c["scheme_id"] == sid and c["kind"] == "portal"
                       and re.search(r"\.(?:in|com|org)\b", c["value"])]
            url = portals[0] if portals else None
            if not url and lj:
                url = next((s.get("url") for s in lj.get("official_sources", []) if s.get("url")), None)
            if not url and seed:
                url = next((s.get("url") for s in seed.get("official_sources", []) if s.get("url")), None)
            official_source = []
            if lj:
                official_source += [s.get("website") for s in lj.get("official_sources", []) if s.get("website")]
            if can:
                official_source.append(f"Official scheme compilation (scheme 2.pdf), dataset {_pages(can.get('source_pages'))}")
            proc = self.proc_meta.get(sid, {})
            status = can.get("status") if can else "ACTIVE"
            self.schemes.append({
                "scheme_id": sid,
                "scheme_name": (lj or {}).get("scheme_name") or can["official_name"],
                "short_name": ov["short_name"], "spoken_name": ov["spoken_name"],
                "legacy_code": ov.get("legacy_code"),
                "category": (can or {}).get("category") or ov.get("category"),
                # Only the canonical records carry these fields; legacy-only schemes stay NULL.
                "ministry": _clean((can or {}).get("ministry")) or None,
                "department": _clean((can or {}).get("department")) or None,
                "implementing_agency": _clean((can or {}).get("implementing_agency")) or None,
                "domains": ov.get("domains", []),
                "description": _clean((lj or {}).get("description")) or _clean((can or {}).get("description")),
                "voice_summary": ov.get("voice_summary"),
                "benefits": bens[:6],
                "target_beneficiaries": [_clean(c.get("category")) for c in lj.get("beneficiary_categories", [])] if lj else None,
                "eligibility": elig,
                "age_min": age_min, "age_max": age_max,
                "income_limit": income, "income_limit_period": "MONTHLY" if income else None,
                "occupation": ov.get("target_occupations", []),
                "gender": gender,
                "migration_support": None if ov.get("migration_support") is None else int(bool(ov["migration_support"])),
                "state": ov.get("state"), "district": None,
                "documents": [d["name"] + ("" if d["mandatory"] else " (optional)") for d in docs],
                "application_process": [f"{s['mode']}: {s['text']}" for s in steps],
                "application_mode": proc.get("application_mode"),
                "portability": port,
                "exceptions": excs,
                "official_source": "; ".join(dict.fromkeys(official_source)) or None,
                "official_url": url,
                "helpline": " / ".join(dict.fromkeys(helplines)) or None,
                "last_verified": (lj or {}).get("last_updated"),
                "status": status,
                "effective_from": (can or {}).get("effective_from"),
                "effective_until": (can or {}).get("effective_until"),
                "superseded_by": (can or {}).get("superseded_by"),
                "parent_scheme_id": (can or {}).get("parent_scheme_id"),
                "is_listed": 0 if ov.get("listed") is False else 1,
                "provenance": {"base": ov["base"], "legacy_json": ov.get("legacy_json"),
                               "canonical_pages": (can or {}).get("source_pages"),
                               "seed_file": SEED_FILES.get(sid) if seed else None,
                               "summary_basis": ov.get("summary_basis"),
                               "migration_support_source": ov.get("migration_support_source")},
            })
        _log(f"schemes: {len(self.schemes)} ({sum(s['is_listed'] for s in self.schemes)} listed for discovery)")

    # ── aliases ───────────────────────────────────────────────────────────
    def build_aliases(self) -> None:
        self.aliases: Dict[Tuple[str, str], float] = {}
        known = {s["scheme_id"] for s in self.schemes}
        for sid, ov in self.overlay["schemes"].items():
            target = self._sid(sid)
            if not target or target not in known:
                continue
            drop = {normalize(a) for a in ov.get("drop_aliases", [])} | ALIAS_BLOCKLIST
            weak = {normalize(a) for a in ov.get("weak_aliases", [])}
            names = list(ov.get("aliases", []))
            can = self.can_schemes.get(sid)
            if can:
                names += can.get("aliases", []) + [can.get("official_name", ""), can.get("abbreviation", "")]
            lj = self.legacy.get(sid)
            if lj:
                names.append(re.sub(r"\s*\(.*?\)|\s[-+—].*$", "", lj.get("scheme_name", "")))
            for raw in names + list(ov.get("weak_aliases", [])):
                a = normalize(raw or "")
                if len(a) < 3 or a in drop:
                    continue
                strength = 0.6 if a in weak else 1.0
                self.aliases[(a, target)] = max(self.aliases.get((a, target), 0.0), strength)
        # An alias claimed by two different schemes is ambiguous - keep neither as explicit.
        by_alias: Dict[str, List[str]] = {}
        for (a, sid) in self.aliases:
            by_alias.setdefault(a, []).append(sid)
        for a, sids in by_alias.items():
            if len(sids) > 1:
                self.warnings.append(f"alias '{a}' shared by {sids} - demoted to weak")
                for sid in sids:
                    self.aliases[(a, sid)] = min(self.aliases[(a, sid)], 0.5)
        _log(f"aliases: {len(self.aliases)}")

    # ── RAG chunks ────────────────────────────────────────────────────────
    def build_chunks(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        seen = set()
        spoken = {s["scheme_id"]: s["spoken_name"] for s in self.schemes}
        names = {s["scheme_id"]: s["scheme_name"] for s in self.schemes}

        def add(sid, section, text, source):
            text = _clean(text)
            if not sid or sid not in spoken or len(text.split()) < 4:
                return
            for part in _split_words(text):
                k = _key(part)
                if k in seen:
                    continue
                seen.add(k)
                if not re.search(re.escape(_key(spoken[sid]).split(" ")[-1]), _key(part)):
                    part = f"{names[sid]}: {part}"
                self.chunks.append({"chunk_id": f"{sid}::{section}::{len(self.chunks)}", "scheme_id": sid,
                                    "section": section, "text": part, "source": source})

        for c in self.can["chunks"]:
            add(self._sid(c["scheme_id"]), c.get("section", "GENERAL").lower(), c.get("text"),
                "canonical " + _pages(c.get("page_numbers")))
        for sid, lj in self.legacy.items():
            src = lj.get("scheme_id", "")
            add(sid, "overview", lj.get("description"), src)
            for e in lj.get("eligibility", []):
                add(sid, "eligibility", e if isinstance(e, str) else e.get("criterion"), src)
            for e in lj.get("beneficiary_categories", []):
                if isinstance(e, dict):
                    add(sid, "beneficiaries", f"{e.get('category')}: {e.get('description')}", src)
            port = lj.get("portability")
            if isinstance(port, dict):
                for k, v in port.items():
                    if k != "source" and isinstance(v, str):
                        add(sid, "portability", v, src)
            for item in lj.get("worker_rights_under_osh_code", []):
                add(sid, "rights", f"{item.get('right', '')} {item.get('action_if_violated', '')}", src)
            for item in lj.get("state_specific_rules", []) if isinstance(lj.get("state_specific_rules"), list) else []:
                add(sid, "state_rules", item if isinstance(item, str) else item.get("note"), src)
            for item in lj.get("important_definitions", []):
                if isinstance(item, dict):
                    add(sid, "definitions", f"{item.get('term')}: {item.get('definition')}", src)
            portals = lj.get("state_portals", [])
            if portals:
                add(sid, "state_portals", "State board portals: " + "; ".join(f"{p['state']}: {p['portal']}" for p in portals), src)
        for b in self.benefits:
            add(b["scheme_id"], "benefits", b["text"], b["source"])
        for e in self.exceptions:
            add(e["scheme_id"], e["kind"], e["text"], e["source"])
        for f in self.faqs:
            add(f["scheme_id"], "faq", f"Q: {f['question']} A: {f['answer']}", f["source"])
        for sid in spoken:
            docs = [d["name"] for d in self.documents if d["scheme_id"] == sid and d["mandatory"]]
            if docs:
                add(sid, "documents", "Documents required: " + ", ".join(docs) + ".", "documents table")
            steps = [s["text"] for s in self.steps if s["scheme_id"] == sid and s["mode"] == "offline"][:4]
            if steps:
                add(sid, "procedure", "How to apply: " + " ".join(steps), "procedure table")
        _log(f"chunks: {len(self.chunks)}")

    # ── intent prototypes ─────────────────────────────────────────────────
    def build_prototypes(self) -> None:
        dataset = self._src(C.INTENT_DATASET_PATH)
        examples = self._src(C.INTENT_EXAMPLES_PATH)["examples"]
        label_map = {"ELIGIBILITY": "Q:ELIGIBILITY", "CHECK_ELIGIBILITY": "Q:ELIGIBILITY", "EXCLUSIONS": "Q:ELIGIBILITY",
                     "FAQ": "Q:OVERVIEW", "OVERVIEW": "Q:OVERVIEW", "PROCEDURE": "Q:PROCEDURE",
                     "DOCUMENTS": "Q:DOCUMENTS", "AUTHORITY": "Q:CONTACT", "BENEFITS": "Q:BENEFITS",
                     "LIST_SCHEMES": "I:SCHEME_DISCOVERY", "OUT_OF_SCOPE": "I:OTHER"}
        alias_re = re.compile(r"(?<![a-z0-9])(" + "|".join(
            re.escape(a) for a in sorted({a for (a, _) in self.aliases} | {normalize(s["scheme_name"]) for s in self.schemes},
                                         key=len, reverse=True) if len(a) >= 3) + r")(?![a-z0-9])")
        per_label: Dict[str, List[str]] = {}
        for row in dataset:
            label = label_map.get(row.get("intent"))
            if not label:
                continue
            masked = alias_re.sub("this scheme", normalize(row["query"]))
            masked = re.sub(r"(this scheme\s*)+", "this scheme ", masked).strip()
            bucket = per_label.setdefault(label, [])
            if masked not in bucket and len(bucket) < 160:
                bucket.append(masked)
        for label, items in examples.items():
            bucket = per_label.setdefault(label, [])
            for q in items:
                masked = alias_re.sub("this scheme", normalize(q))
                if masked not in bucket:
                    bucket.append(masked)
        self.proto_texts, self.proto_labels = [], []
        for label, items in sorted(per_label.items()):
            self.proto_texts += items
            self.proto_labels += [label] * len(items)
        _log(f"intent prototypes: {len(self.proto_texts)} across {len(per_label)} labels")

    # ── writing ───────────────────────────────────────────────────────────
    def write_db(self) -> None:
        tmp = C.DB_PATH + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = sqlite3.connect(tmp)
        con.executescript(SCHEMA)
        meta = {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "build_date": self.today,
            "embedding_model": C.EMBEDDING_MODEL_NAME,
            "domain_primary": self.overlay.get("domain_primary", {}),
            "unverified_schemes": {normalize(k): v for k, v in self.overlay.get("unverified_schemes", {}).items()},
            "verify_fields": self.overlay.get("verify_fields", []),
            "chunk_count": len(self.chunks),
        }
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        cols = ["scheme_id", "scheme_name", "short_name", "spoken_name", "legacy_code", "category", "ministry",
                "department", "implementing_agency", "domains",
                "description", "voice_summary", "benefits", "target_beneficiaries", "eligibility", "age_min", "age_max",
                "income_limit", "income_limit_period", "occupation", "gender", "migration_support", "state", "district",
                "documents", "application_process", "application_mode", "portability", "exceptions", "official_source",
                "official_url", "helpline", "last_verified", "status", "effective_from", "effective_until",
                "superseded_by", "parent_scheme_id", "is_listed", "provenance"]
        for s in self.schemes:
            row = [json.dumps(s[c], ensure_ascii=False) if isinstance(s[c], (list, dict)) else s[c] for c in cols]
            con.execute(f"INSERT INTO schemes ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", row)
        con.executemany("INSERT INTO benefits (scheme_id, rank, text, value_inr, frequency, source) VALUES (?,?,?,?,?,?)",
                        [(b["scheme_id"], b["rank"], b["text"], b["value_inr"], b["frequency"], b["source"]) for b in self.benefits])
        con.executemany("INSERT INTO documents (scheme_id, rank, name, description, mandatory, source) VALUES (?,?,?,?,?,?)",
                        [(d["scheme_id"], d["rank"], d["name"], d["description"], d["mandatory"], d["source"]) for d in self.documents])
        con.executemany("INSERT INTO procedure_steps (scheme_id, mode, step_no, text, source) VALUES (?,?,?,?,?)",
                        [(s["scheme_id"], s["mode"], s["step_no"], s["text"], s["source"]) for s in self.steps])
        con.executemany("INSERT INTO procedure_meta VALUES (?,?,?,?,?)",
                        [(m["scheme_id"], m.get("application_mode"), m.get("processing_time"), m.get("fees"), m.get("source"))
                         for m in self.proc_meta.values()])
        con.executemany("INSERT INTO exceptions (scheme_id, kind, text, source) VALUES (?,?,?,?)",
                        [(e["scheme_id"], e["kind"], e["text"], e["source"]) for e in self.exceptions])
        con.executemany("INSERT INTO faqs (scheme_id, question, answer, source) VALUES (?,?,?,?)",
                        [(f["scheme_id"], f["question"], f["answer"], f["source"]) for f in self.faqs])
        con.executemany("INSERT INTO contacts (scheme_id, kind, value, description, source) VALUES (?,?,?,?,?)",
                        [(c["scheme_id"], c["kind"], c["value"], c["description"], c["source"]) for c in self.contacts])
        con.executemany("INSERT INTO rules VALUES (?,?,?,?,?,?,?,?,?)",
                        [(r["rule_id"], r["scheme_id"], r["rule_type"], json.dumps(r["logic"]),
                          json.dumps(r["unless_logic"]) if r["unless_logic"] else None, r["description"], r["source"],
                          r["valid_from"], r["valid_until"]) for r in self.rules])
        con.executemany("INSERT INTO aliases VALUES (?,?,?)", [(a, sid, s) for (a, sid), s in self.aliases.items()])
        con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                        [(i, c["chunk_id"], c["scheme_id"], c["section"], c["text"], c["source"]) for i, c in enumerate(self.chunks)])
        rel = [(self._sid(r.get("source_scheme_id", "")) or r.get("source_scheme_id"),
                self._sid(r.get("target_scheme_id", "")) or r.get("target_scheme_id"),
                r.get("relationship_type"), _clean(r.get("description"))) for r in self.can["relationships"]]
        con.executemany("INSERT INTO relationships (source_scheme_id, target_scheme_id, relationship_type, description) VALUES (?,?,?,?)", rel)
        con.commit()
        con.execute("VACUUM")
        con.close()
        os.replace(tmp, C.DB_PATH)
        _log(f"wrote {C.DB_PATH} ({os.path.getsize(C.DB_PATH) // 1024} KB)")

    def embed(self) -> Dict[str, Any]:
        import numpy as np
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        t0 = time.monotonic()
        model = SentenceTransformer(C.EMBEDDING_MODEL_NAME, device="cpu")
        passages = [C.PASSAGE_PREFIX + c["text"] for c in self.chunks]
        emb = model.encode(passages, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float16)
        tmp = C.RAG_EMBEDDINGS_PATH + ".tmp.npy"
        np.save(tmp, emb)
        os.replace(tmp, C.RAG_EMBEDDINGS_PATH)
        queries = [C.QUERY_PREFIX + q for q in self.proto_texts]
        pemb = model.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float16)
        tmp = C.INTENT_PROTOTYPES_PATH + ".tmp.npz"
        np.savez(tmp, embeddings=pemb, labels=np.array(self.proto_labels), texts=np.array(self.proto_texts))
        os.replace(tmp, C.INTENT_PROTOTYPES_PATH)
        # FAQ questions on their own (ids read back from the database just written),
        # so a spoken question is matched question-to-question at runtime.
        con = sqlite3.connect(f"file:{C.DB_PATH}?mode=ro", uri=True)
        faq_rows = con.execute("SELECT id, scheme_id, question FROM faqs ORDER BY id").fetchall()
        con.close()
        femb = model.encode([C.QUERY_PREFIX + q for _i, _s, q in faq_rows], batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        tmp = C.FAQ_QUESTIONS_PATH + ".tmp.npz"
        np.savez(tmp, embeddings=femb, ids=np.array([i for i, _s, _q in faq_rows]),
                 scheme_ids=np.array([s for _i, s, _q in faq_rows]), questions=np.array([q for _i, _s, q in faq_rows]))
        os.replace(tmp, C.FAQ_QUESTIONS_PATH)
        took = time.monotonic() - t0
        _log(f"embeddings: {emb.shape} chunks + {pemb.shape} prototypes + {femb.shape} FAQ questions in {took:.1f}s")
        return {"chunks": list(emb.shape), "prototypes": list(pemb.shape), "faq_questions": list(femb.shape),
                "seconds": round(took, 1)}

    def run(self) -> None:
        t0 = time.monotonic()
        self.load_sources()
        self.build_children()
        self.build_rules()
        self.build_schemes()
        self.build_aliases()
        self.build_chunks()
        self.build_prototypes()
        self.write_db()
        emb_info = self.embed() if self.with_embeddings else {"skipped": True}
        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": C.EMBEDDING_MODEL_NAME, "embeddings": emb_info,
            "counts": {"schemes": len(self.schemes), "rules": len(self.rules), "benefits": len(self.benefits),
                       "documents": len(self.documents), "procedure_steps": len(self.steps),
                       "exceptions": len(self.exceptions), "faqs": len(self.faqs), "contacts": len(self.contacts),
                       "aliases": len(self.aliases), "chunks": len(self.chunks), "prototypes": len(self.proto_texts)},
            "sources": self.sources, "warnings": self.warnings, "build_seconds": round(time.monotonic() - t0, 1),
        }
        tmp = C.MANIFEST_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
        os.replace(tmp, C.MANIFEST_PATH)
        for w in self.warnings:
            _log("warning: " + w)
        _log(f"done in {manifest['build_seconds']}s")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the NomadRight scheme-intelligence knowledge base")
    ap.add_argument("--no-embeddings", action="store_true", help="skip e5 embeddings (RAG/classifier disabled)")
    args = ap.parse_args(argv)
    Builder(with_embeddings=not args.no_embeddings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
