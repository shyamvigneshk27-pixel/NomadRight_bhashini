"""
SchemeRepository - read-only access to the scheme-intelligence SQLite
knowledge base (built offline by build_kb.py; never written at runtime).

The whole KB is a few hundred rows, so per-scheme detail is cached in memory
at start-up (a few ms); candidate filtering for discovery still runs as a SQL
query against the file.
"""

import json
import os
import re
import sqlite3
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C

_JSON_COLUMNS = ("domains", "target_beneficiaries", "eligibility", "occupation", "documents",
                 "application_process", "exceptions", "benefits", "provenance")


class KnowledgeBaseMissing(RuntimeError):
    """The scheme-intelligence DB has not been built (run build_kb)."""


class SchemeRepository:
    def __init__(self, db_path: str = C.DB_PATH):
        if not os.path.exists(db_path):
            raise KnowledgeBaseMissing(
                f"{db_path} not found - build it with: "
                "python -m pocketinfer.applications.nomad_right.scheme_intel.build_kb")
        self.db_path = db_path
        # immutable=1: a build artifact, never modified while the app runs (build_kb
        # writes a new file and renames it into place), so no locking/WAL needed.
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._load()

    # ── loading ───────────────────────────────────────────────────────────
    def _rows(self, sql: str, args: Tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _load(self) -> None:
        self.meta: Dict[str, Any] = {}
        for r in self._rows("SELECT key, value FROM meta"):
            try:
                self.meta[r["key"]] = json.loads(r["value"])
            except (TypeError, ValueError):
                self.meta[r["key"]] = r["value"]

        self._schemes: Dict[str, Dict[str, Any]] = {}
        for r in self._rows("SELECT * FROM schemes"):
            d = dict(r)
            for col in _JSON_COLUMNS:
                if col in d and isinstance(d[col], str):
                    try:
                        d[col] = json.loads(d[col])
                    except ValueError:
                        pass
            self._schemes[d["scheme_id"]] = d

        self._benefits = defaultdict(list)
        for r in self._rows("SELECT * FROM benefits ORDER BY scheme_id, rank"):
            self._benefits[r["scheme_id"]].append(dict(r))
        self._documents = defaultdict(list)
        for r in self._rows("SELECT * FROM documents ORDER BY scheme_id, rank"):
            self._documents[r["scheme_id"]].append(dict(r))
        self._steps = defaultdict(lambda: {"online": [], "offline": []})
        for r in self._rows("SELECT * FROM procedure_steps ORDER BY scheme_id, mode, step_no"):
            self._steps[r["scheme_id"]].setdefault(r["mode"], []).append(dict(r))
        self._proc_meta = {r["scheme_id"]: dict(r) for r in self._rows("SELECT * FROM procedure_meta")}
        self._exceptions = defaultdict(list)
        for r in self._rows("SELECT * FROM exceptions ORDER BY id"):
            self._exceptions[r["scheme_id"]].append(dict(r))
        self._faqs = defaultdict(list)
        for r in self._rows("SELECT * FROM faqs ORDER BY id"):
            self._faqs[r["scheme_id"]].append(dict(r))
        self._contacts = defaultdict(list)
        for r in self._rows("SELECT * FROM contacts ORDER BY id"):
            self._contacts[r["scheme_id"]].append(dict(r))
        self._rules = defaultdict(list)
        for r in self._rows("SELECT * FROM rules ORDER BY scheme_id, rule_type DESC, rule_id"):
            d = dict(r)
            d["logic"] = json.loads(d["logic"])
            d["unless_logic"] = json.loads(d["unless_logic"]) if d.get("unless_logic") else None
            self._rules[r["scheme_id"]].append(d)
        self.chunks: List[Dict[str, Any]] = [dict(r) for r in self._rows("SELECT * FROM chunks ORDER BY row")]

        alias_rows = self._rows("SELECT alias, scheme_id, strength FROM aliases")
        self._alias_map: Dict[str, Tuple[str, float]] = {}
        for r in alias_rows:
            prev = self._alias_map.get(r["alias"])
            if prev is None or r["strength"] > prev[1]:
                self._alias_map[r["alias"]] = (r["scheme_id"], float(r["strength"]))
        self._alias_re = self._build_alias_re(self._alias_map.keys())

        self._unverified: Dict[str, str] = self.meta.get("unverified_schemes", {}) or {}
        self._unverified_re = self._build_alias_re(self._unverified.keys()) if self._unverified else None
        self._domain_primary: Dict[str, List[str]] = self.meta.get("domain_primary", {}) or {}
        self._legacy_to_sid = {d["legacy_code"]: sid for sid, d in self._schemes.items() if d.get("legacy_code")}
        # MGNREGS is shared by MGNREGA and its successor; the legacy record is MGNREGA's.
        if "MGNREGS" in self._legacy_to_sid and "SCH_MGNREGA" in self._schemes:
            self._legacy_to_sid["MGNREGS"] = "SCH_MGNREGA"

    @staticmethod
    def _build_alias_re(aliases) -> Optional["re.Pattern"]:
        items = sorted((a for a in aliases if a), key=len, reverse=True)
        if not items:
            return None
        return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in items) + r")(?![a-z0-9])")

    # ── scheme records ────────────────────────────────────────────────────
    def has(self, sid: Optional[str]) -> bool:
        return bool(sid) and sid in self._schemes

    def scheme(self, sid: str) -> Dict[str, Any]:
        return self._schemes[sid]

    def all_ids(self, listed_only: bool = True) -> List[str]:
        return [sid for sid, d in self._schemes.items() if d.get("is_listed") or not listed_only]

    def spoken(self, sid: str) -> str:
        return self._schemes[sid].get("spoken_name") or self._schemes[sid]["scheme_name"]

    def short(self, sid: str) -> str:
        return self._schemes[sid].get("short_name") or sid

    def legacy_code(self, sid: Optional[str]) -> Optional[str]:
        return self._schemes.get(sid, {}).get("legacy_code") if sid else None

    def sid_for_legacy(self, code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        if code in self._schemes:
            return code
        return self._legacy_to_sid.get(code.upper())

    def benefits(self, sid: str) -> List[Dict[str, Any]]:
        return self._benefits.get(sid, [])

    def documents(self, sid: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        docs = self._documents.get(sid, [])
        return [d for d in docs if d["mandatory"]], [d for d in docs if not d["mandatory"]]

    def procedure(self, sid: str) -> Dict[str, Any]:
        steps = self._steps.get(sid, {"online": [], "offline": []})
        meta = self._proc_meta.get(sid, {})
        return {
            "online": [s["text"] for s in steps.get("online", [])],
            "offline": [s["text"] for s in steps.get("offline", [])],
            "mode": meta.get("application_mode") or self._schemes.get(sid, {}).get("application_mode"),
            "fees": meta.get("fees"),
            "processing_time": meta.get("processing_time"),
        }

    def exceptions(self, sid: str) -> List[Dict[str, Any]]:
        return self._exceptions.get(sid, [])

    def faqs(self, sid: str) -> List[Dict[str, Any]]:
        return self._faqs.get(sid, [])

    def contacts(self, sid: str) -> List[Dict[str, Any]]:
        return self._contacts.get(sid, [])

    def rules(self, sid: str) -> List[Dict[str, Any]]:
        return self._rules.get(sid, [])

    def domain_primary(self, domain: str) -> List[str]:
        return [s for s in self._domain_primary.get(domain, []) if s in self._schemes]

    # ── text matching ─────────────────────────────────────────────────────
    def find_schemes(self, text_norm: str) -> List[Tuple[str, float, str, int]]:
        """(scheme_id, strength, alias, position) for every alias found, in order."""
        if not self._alias_re or not text_norm:
            return []
        found, seen = [], set()
        for m in self._alias_re.finditer(text_norm):
            sid, strength = self._alias_map[m.group(1)]
            if sid in seen:
                continue
            seen.add(sid)
            found.append((sid, strength, m.group(1), m.start()))
        return found

    def find_unverified(self, text_norm: str) -> Optional[str]:
        if not self._unverified_re:
            return None
        m = self._unverified_re.search(text_norm)
        return self._unverified[m.group(1)] if m else None

    # ── SQL filter for discovery ──────────────────────────────────────────
    def candidates(self, domains: Optional[List[str]] = None, state: Optional[str] = None,
                   listed_only: bool = True) -> List[str]:
        """
        Pre-filter by metadata before any semantic ranking: listed, still in
        force (or replaced - kept so the successor can be mentioned), and
        either national or specific to the person's state.
        """
        sql = ["SELECT scheme_id FROM schemes WHERE 1=1"]
        args: List[Any] = []
        if listed_only:
            sql.append("AND is_listed = 1")
        if state:
            sql.append("AND (state IS NULL OR state = 'ALL' OR state = ?)")
            args.append(state)
        else:
            sql.append("AND (state IS NULL OR state = 'ALL')")
        if domains:
            sql.append("AND (" + " OR ".join("domains LIKE ?" for _ in domains) + ")")
            args.extend(f'%"{d}"%' for d in domains)
        return [r["scheme_id"] for r in self._rows(" ".join(sql), tuple(args))]

    def stats(self) -> Dict[str, int]:
        return {
            "schemes": len(self._schemes),
            "listed": sum(1 for d in self._schemes.values() if d.get("is_listed")),
            "rules": sum(len(v) for v in self._rules.values()),
            "documents": sum(len(v) for v in self._documents.values()),
            "benefits": sum(len(v) for v in self._benefits.values()),
            "faqs": sum(len(v) for v in self._faqs.values()),
            "chunks": len(self.chunks),
            "aliases": len(self._alias_map),
        }
