"""
FormSession - the deterministic question/answer state machine for one citizen's
form, with no I/O of its own: the app feeds it ASR text (or a document image for
the fields read from a passbook/Aadhaar), and it says what to speak next and what
the screen should show. Field order, required-ness and validation come from the
form definition only.

States: CONFIRM_FORM -> ASKING -> (CONFIRMING the value for sensitive fields) ->
... -> REVIEW -> DONE (payload ready) | CANCELLED. A document field first asks
for the document (DOC_WAIT); if OCR cannot read the value the same field is
asked by voice (spoken_question) instead.

Personal data lives only in self.values; nothing here logs a value. clear()
overwrites and drops them.
"""
import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pocketinfer.applications.nomad_right.formfill import validators as V
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3          # per field before an optional one is skipped / a required one is marked for the officer


@dataclass
class Step:
    """What the app should do next."""
    speak: Optional[str]                 # text for TTS in the session language (None: nothing to say)
    listen: bool = True                  # then wait for the citizen's answer
    want_document: Optional[str] = None  # ask for a camera capture of this document instead of speech
    done: bool = False                   # payload ready (DONE) or session ended (CANCELLED)
    ui: Dict = field(default_factory=dict)


@dataclass
class FieldRef:
    key: str                             # canonical key (for groups: "<group>.<index>.<key>")
    spec: dict
    group: Optional[str] = None
    index: Optional[int] = None


class FormSession:
    def __init__(self, catalog: FormCatalog, form_id: str, lang: str, review: bool = True, session_id: Optional[str] = None):
        self.catalog, self.lang = catalog, lang
        self.form = catalog.form(form_id)
        self.form_id, self.scheme_id = form_id, self.form["scheme_id"]
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.review_enabled = review
        self.values: Dict[str, object] = {}
        self.displays: Dict[str, str] = {}
        self.skipped: List[str] = []
        self.unanswered_required: List[str] = []
        self.state = "CONFIRM_FORM"
        self.plan: List[FieldRef] = []
        self.pos = 0
        self.attempts = 0
        self.pending: Optional[Tuple[FieldRef, V.Parsed]] = None   # value awaiting yes/no
        self.doc_failed = False           # document field fell back to voice
        self.created = time.monotonic()
        self.last_activity = self.created
        self._rebuild_plan()

    # ── plan ────────────────────────────────────────────────────────────────
    def _rebuild_plan(self) -> None:
        plan: List[FieldRef] = []
        for spec in self.form["fields"]:
            if spec["type"] == "group":
                n = self.values.get(spec.get("count_from"), 0)
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 0
                for i in range(1, min(n, spec.get("max_items", 6)) + 1):
                    for sub in spec["fields"]:
                        plan.append(FieldRef(f"{spec['key']}.{i}.{sub['key']}", sub, spec["key"], i))
            else:
                plan.append(FieldRef(spec["key"], spec))
        self.plan = plan

    def _applicable(self, ref: FieldRef) -> bool:
        cond = ref.spec.get("condition")
        if not cond:
            return True
        dep = self.values.get(cond["field"])
        if "equals" in cond:
            return dep == cond["equals"]
        if cond.get("minor"):
            if not isinstance(dep, str) or len(dep) < 4:
                return False
            try:
                return time.localtime().tm_year - int(dep[:4]) < 18
            except ValueError:
                return False
        return True

    def _current(self) -> Optional[FieldRef]:
        while self.pos < len(self.plan):
            ref = self.plan[self.pos]
            if self._applicable(ref):
                return ref
            self.pos += 1
        return None

    def progress(self) -> Tuple[int, int]:
        applicable = [r for r in self.plan if self._applicable(r)]
        done = sum(1 for r in applicable if r.key in self.values or r.key in self.skipped)
        return done, len(applicable)

    # ── prompts ─────────────────────────────────────────────────────────────
    def _p(self, key: str, **kw) -> str:
        return self.catalog.prompt(key, self.lang, **kw)

    def _question(self, ref: FieldRef, spoken_fallback: bool = False) -> str:
        spec = ref.spec
        q = spec["spoken_question"] if (spoken_fallback and "spoken_question" in spec) else spec["question"]
        text = q.get(self.lang) or q["en"]
        if ref.index is not None:
            text = text.replace("{n}", str(ref.index))
        if not spec.get("required", True) and "skip" not in text.lower() and "छोड़ो" not in text and "தவிர்" not in text:
            text = f"{text} {self._p('optional_hint')}"
        return text

    def _label(self, ref: FieldRef) -> str:
        lab = ref.spec.get("label") or {}
        base = lab.get(self.lang) or lab.get("en") or ref.spec["key"]
        return f"{base} {ref.index}" if ref.index is not None else base

    def _ui(self, **extra) -> Dict:
        done, total = self.progress()
        cur = self._current() if self.state in ("ASKING", "DOC_WAIT", "CONFIRMING") else None
        ui = {"form_id": self.form_id, "scheme_id": self.scheme_id, "title": self.form["title"].get(self.lang) or self.form["title"]["en"],
              "state": self.state, "field_index": done + (1 if cur else 0), "field_count": total,
              "field_key": cur.key if cur else None, "field_label": self._label(cur) if cur else None,
              "question": self._question(cur, self.doc_failed) if cur else None,
              "lang": self.lang, "session_id": self.session_id}
        ui.update(extra)
        return ui

    # ── flow ────────────────────────────────────────────────────────────────
    def start(self, form_confirmed: bool = False) -> Step:
        self.touch()
        if not form_confirmed:
            self.state = "CONFIRM_FORM"
            return Step(self._p("confirm_form", title=self.catalog.spoken_name(self.form_id, self.lang)), ui=self._ui())
        return self._begin()

    def _begin(self) -> Step:
        self.state = "ASKING"
        self.pos, self.attempts = 0, 0
        return self._ask(prefix=self._p("start"))

    def _ask(self, prefix: Optional[str] = None, spoken_fallback: bool = False) -> Step:
        ref = self._current()
        if ref is None:
            return self._to_review()
        self.doc_failed = spoken_fallback
        if ref.spec.get("source_document") and not spoken_fallback:
            self.state = "DOC_WAIT"
            text = self._question(ref)
            return Step((prefix + " " if prefix else "") + text, listen=False, want_document=ref.spec["source_document"], ui=self._ui())
        self.state = "ASKING"
        done, total = self.progress()
        text = self._question(ref, spoken_fallback)
        return Step((prefix + " " if prefix else "") + text, ui=self._ui())

    def handle_document(self, ocr_text: str) -> Step:
        """OCR text of the document the citizen showed for the current field."""
        self.touch()
        ref = self._current()
        if ref is None or self.state != "DOC_WAIT":
            return self._ask()
        from pocketinfer.applications.nomad_right.formfill import ocr as O
        ftype = ref.spec["type"]
        val = None
        if ftype == "aadhaar":
            val = O.find_aadhaar(ocr_text)
        elif ftype == "account_number":
            val = O.find_account_number(ocr_text, exclude=[str(v) for v in self.values.values() if isinstance(v, str)])
        elif ftype == "ifsc":
            val = O.find_ifsc(ocr_text)
        elif ref.spec["key"] == "pan":
            val = O.find_pan(ocr_text)
        elif ref.spec["key"] == "mobile":
            val = O.find_mobile(ocr_text)
        if val is None:
            self.attempts += 1
            if not ref.spec.get("required", True) or ftype == "ifsc":
                # nobody can dictate an IFSC code; an optional value the camera could not
                # read is left for the officer rather than asked for by voice
                self.skipped.append(ref.key)
                self.attempts = 0
                self.pos += 1
                return self._ask(prefix=self._p("doc_read_fail_skip"))
            return self._ask(prefix=self._p("doc_read_fail"), spoken_fallback=True)
        parsed = V.parse_by_type(ftype if ftype in ("aadhaar", "account_number", "ifsc", "phone") else "text", val, "en", ref.spec, self.catalog.words) \
            if ftype in ("aadhaar", "account_number", "ifsc") else V.Parsed(val, val)
        self.pending = (ref, parsed)
        self.state = "CONFIRMING"
        return Step(self._p("doc_read_ok", value=self._spell(parsed.display)), ui=self._ui(pending_value=parsed.display))

    def handle_answer(self, text: str) -> Step:
        self.touch()
        text = (text or "").strip()
        if self.state == "CANCELLED" or self.state == "DONE":
            return Step(None, listen=False, done=True, ui=self._ui())
        cmd = V.detect_command(text, self.lang, self.catalog.words)
        if cmd == "cancel":
            return self.cancel()
        if self.state == "CONFIRM_FORM":
            yn = V.parse_yesno(text, self.lang, self.catalog.words)
            if yn is True:
                return self._begin()
            if yn is False:
                self.state = "CANCELLED"
                return Step(self._p("cancelled"), listen=False, done=True, ui=self._ui(reason="wrong_form"))
            return Step(self._p("confirm_form", title=self.catalog.spoken_name(self.form_id, self.lang)), ui=self._ui())
        if self.state == "REVIEW":
            return self._handle_review(text)
        if self.state == "CHANGE_WHICH":
            return self._handle_change_which(text)
        if self.state == "CONFIRMING":
            yn = V.parse_yesno(text, self.lang, self.catalog.words)
            ref, parsed = self.pending
            if yn is True:
                self._store(ref, parsed)
                self.pending, self.attempts = None, 0
                self.pos += 1
                return self._ask()
            if yn is False or cmd == "change":
                self.pending = None
                self.attempts += 1
                return self._ask(spoken_fallback=bool(ref.spec.get("source_document")))
            return Step(self._p("confirm_value", value=self._spell(parsed.display)), ui=self._ui(pending_value=parsed.display))
        # ASKING (or DOC_WAIT with a spoken answer)
        ref = self._current()
        if ref is None:
            return self._to_review()
        if cmd == "repeat":
            return self._ask(spoken_fallback=self.doc_failed)
        if cmd == "change":
            return self._go_back()
        if cmd == "skip" or (not text and not ref.spec.get("required", True)):
            return self._skip(ref)
        if not text:
            self.attempts += 1
            return self._retry(ref, "retry")
        try:
            parsed = V.parse_by_type(ref.spec["type"], text, self.lang, ref.spec, self.catalog.words)
        except V.Invalid as exc:
            self.attempts += 1
            return self._retry(ref, exc.prompt_key)
        if ref.spec.get("confirm") or ref.spec["type"] in ("aadhaar", "account_number", "phone", "date", "amount"):
            self.pending = (ref, parsed)
            self.state = "CONFIRMING"
            return Step(self._p("confirm_value", value=self._spell(parsed.display)), ui=self._ui(pending_value=parsed.display))
        self._store(ref, parsed)
        self.attempts = 0
        self.pos += 1
        return self._ask()

    def _retry(self, ref: FieldRef, key: str) -> Step:
        if self.attempts >= MAX_ATTEMPTS:
            if not ref.spec.get("required", True):
                return self._skip(ref)
            # a required answer the kiosk cannot get: leave it for the officer and move on
            self.unanswered_required.append(ref.key)
            self.skipped.append(ref.key)
            self.attempts = 0
            self.pos += 1
            return self._ask()
        extra = {}
        if key == "invalid_choice":
            opts = ref.spec.get("options", [])
            extra["options"] = ", ".join((o.get(self.lang) or o.get("en") or [o["value"]])[0] for o in opts)
        return Step(self._p(key, **extra) + " " + self._question(ref, self.doc_failed), ui=self._ui())

    def _skip(self, ref: FieldRef) -> Step:
        if ref.spec.get("required", True) and self.attempts < MAX_ATTEMPTS:
            self.attempts += 1
            return Step(self._p("retry") + " " + self._question(ref, self.doc_failed), ui=self._ui())
        self.skipped.append(ref.key)
        if ref.spec.get("required", True):
            self.unanswered_required.append(ref.key)
        self.attempts = 0
        self.pos += 1
        return self._ask()

    def _go_back(self) -> Step:
        # the previous applicable, answered field
        i = self.pos - 1
        while i >= 0:
            ref = self.plan[i]
            if self._applicable(ref) and (ref.key in self.values or ref.key in self.skipped):
                self.values.pop(ref.key, None); self.displays.pop(ref.key, None)
                if ref.key in self.skipped:
                    self.skipped.remove(ref.key)
                if ref.key in self.unanswered_required:
                    self.unanswered_required.remove(ref.key)
                self.pos, self.attempts = i, 0
                return self._ask()
            i -= 1
        return self._ask()

    def _store(self, ref: FieldRef, parsed: V.Parsed) -> None:
        self.values[ref.key] = parsed.value
        self.displays[ref.key] = parsed.display
        if ref.spec["type"] == "integer" and any(g.get("count_from") == ref.key for g in self.form["fields"] if g["type"] == "group"):
            self._rebuild_plan()

    def _spell(self, display: str) -> str:
        return display

    # ── review ──────────────────────────────────────────────────────────────
    def _to_review(self) -> Step:
        if not self.review_enabled:
            return self._finish()
        self.state = "REVIEW"
        lines = [self._p("review_intro")]
        n = 0
        for ref in self.plan:
            if ref.key in self.values:
                n += 1
                lines.append(f"{n}. {self._label(ref)}: {self.displays.get(ref.key, self.values[ref.key])}.")
        lines.append(self._p("review_confirm"))
        return Step(" ".join(lines), ui=self._ui(review=self.review_items()))

    def review_items(self) -> List[Dict]:
        items, n = [], 0
        for ref in self.plan:
            if ref.key in self.values:
                n += 1
                items.append({"n": n, "key": ref.key, "label": self._label(ref), "value": self.displays.get(ref.key, str(self.values[ref.key]))})
        return items

    def _handle_review(self, text: str) -> Step:
        yn = V.parse_yesno(text, self.lang, self.catalog.words)
        cmd = V.detect_command(text, self.lang, self.catalog.words)
        if yn is True and cmd != "change":
            return self._finish()
        if yn is False or cmd == "change":
            self.state = "CHANGE_WHICH"
            return Step(self._p("which_field"), ui=self._ui(review=self.review_items()))
        return Step(self._p("review_confirm"), ui=self._ui(review=self.review_items()))

    def _handle_change_which(self, text: str) -> Step:
        items = self.review_items()
        try:
            n = V.parse_integer(text, self.lang, 1, len(items)).value
        except V.Invalid:
            # maybe a label was spoken
            t = V.norm(text)
            match = [it for it in items if V.norm(it["label"]) and V.norm(it["label"]) in t]
            if not match:
                return Step(self._p("which_field"), ui=self._ui(review=items))
            n = match[0]["n"]
        key = items[n - 1]["key"]
        for i, ref in enumerate(self.plan):
            if ref.key == key:
                self.values.pop(key, None); self.displays.pop(key, None)
                self.pos, self.attempts = i, 0
                self.state = "ASKING"
                return self._ask()
        return self._to_review()

    def _finish(self) -> Step:
        self.state = "DONE"
        return Step(None, listen=False, done=True, ui=self._ui())

    def cancel(self) -> Step:
        self.state = "CANCELLED"
        self.clear()
        return Step(self._p("cancelled"), listen=False, done=True, ui=self._ui(reason="cancelled"))

    # ── payload / lifecycle ────────────────────────────────────────────────
    def payload(self) -> Dict:
        """The completed form as one structured record (still plaintext - the caller
        seals it immediately). Group answers become lists of dicts."""
        fields: Dict[str, object] = {}
        groups: Dict[str, Dict[int, Dict[str, object]]] = {}
        for key, val in self.values.items():
            parts = key.split(".")
            if len(parts) == 3 and parts[1].isdigit():
                groups.setdefault(parts[0], {}).setdefault(int(parts[1]), {})[parts[2]] = val
            else:
                fields[key] = val
        for g, items in groups.items():
            fields[g] = [items[i] for i in sorted(items)]
        return {"version": 1, "session_id": self.session_id, "form_id": self.form_id, "scheme_id": self.scheme_id, "language": self.lang,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fields": fields,
                "skipped": list(self.skipped), "unanswered_required": list(self.unanswered_required)}

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    def clear(self) -> None:
        """Forget every answer (overwrite, then drop)."""
        for k in list(self.values):
            self.values[k] = None
        self.values.clear(); self.displays.clear(); self.pending = None
