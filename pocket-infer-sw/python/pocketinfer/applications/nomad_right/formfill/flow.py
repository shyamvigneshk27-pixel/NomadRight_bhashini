"""
FormFlow - runs one assisted form-filling session on the app's turn thread:

  photo of the form -> header OCR -> identification (accept / confirm / choose /
  unsupported / not readable) -> spoken confirmation -> FormSession questions,
  each spoken through TTS in the selected language, answered by hold-to-talk +
  ASR (or a document shown to the camera for the sensitive fields) -> review ->
  sealed payload handed to the outbox.

All device I/O comes through a small adapter (FlowIO) so the flow can be driven
by fakes in tests. Nothing here logs an answer. Home, a language change or
idleness cancel the session and wipe its answers.
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pocketinfer.applications.nomad_right import constants
from pocketinfer.applications.nomad_right.formfill import ocr as O
from pocketinfer.applications.nomad_right.formfill import validators as V
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.identifier import FormIdentifier, Identification
from pocketinfer.applications.nomad_right.formfill.session import FormSession, Step

logger = logging.getLogger(__name__)

HOME = "__home__"          # listen() result when the person pressed Home
TIMEOUT = "__timeout__"    # nobody pressed the button in time


@dataclass
class FlowIO:
    speak: Callable[[str], bool]                       # TTS + playback in the session language; False if interrupted
    listen: Callable[[float], str]                     # wait for hold-to-talk, record, ASR -> text | HOME | TIMEOUT
    capture: Callable[[float], Optional[np.ndarray]]   # wait for the button, grab a frame (None: HOME / timeout)
    ui: Callable[[Dict], None]                         # push the form panel state to the screen
    status: Callable[[str], None]                      # status bar line


@dataclass
class FlowResult:
    outcome: str                                       # sent | pending | cancelled | unsupported | not_readable | failed
    form_id: Optional[str] = None
    payload: Optional[Dict] = None
    seconds: float = 0.0
    timings: Dict[str, float] = None


class FormFlow:
    def __init__(self, catalog: FormCatalog, io: FlowIO, lang: str, outbox=None):
        self.catalog, self.io, self.lang, self.outbox = catalog, io, lang, outbox
        self.ident = FormIdentifier(catalog)
        self.session: Optional[FormSession] = None
        self.timings: Dict[str, float] = {}
        self.snapshots = 0                 # how many partial pictures of the session were handed to the outbox
        self._last_fp: Optional[str] = None

    # ── helpers ────────────────────────────────────────────────────────────
    def _p(self, key: str, **kw) -> str:
        return self.catalog.prompt(key, self.lang, **kw)

    def _ui(self, **state) -> None:
        base = {"active": True, "lang": self.lang}
        if self.session is not None:
            base.update(self.session._ui())
        base.update(state)
        try:
            self.io.ui(base)
        except Exception:
            logger.debug("form ui push failed", exc_info=True)

    def _t(self, key: str, t0: float) -> None:
        self.timings[key] = round(time.perf_counter() - t0, 3)

    # ── save as you go ─────────────────────────────────────────────────────
    def _queue_snapshot(self, payload: Dict) -> None:
        """Seal a picture of the session and hand it to the outbox without waiting for
        the network (the kiosk's background sender takes it from there)."""
        q = getattr(self.outbox, "queue", None)
        if self.outbox is None or q is None or not constants.FORM_SEND_AS_YOU_GO:
            return
        q(payload)
        self.snapshots += 1

    def _maybe_snapshot(self) -> None:
        """After every step: if an answer was stored, skipped, replaced or left for the
        officer since the last look, send the session as it stands now."""
        se = self.session
        if se is None or se.state in ("DONE", "CANCELLED") or not (se.values or se.skipped):
            return
        fp = json.dumps([se.values, se.skipped, se.unanswered_required], sort_keys=True, default=str, ensure_ascii=False)
        if fp != self._last_fp:
            self._last_fp = fp
            se.snapshot("in_progress")

    # ── identification ─────────────────────────────────────────────────────
    def identify(self, image: np.ndarray, prior_scheme_id: Optional[str] = None) -> Identification:
        t0 = time.perf_counter()
        passes = O.read_header(image, langs=("en",))
        res = self.ident.identify(passes, prior_scheme_id)
        if res.status != "accept" and self.lang in ("hi", "ta"):
            passes.update(O.read_header(image, langs=(self.lang,)))
            res = self.ident.identify(passes, prior_scheme_id)
        if res.status != "accept" and (res.status in ("unsupported", "not_readable") or (res.candidates and res.candidates[0].score < 0.6)):
            passes["en_full"] = O.read_document(image, "en", frac=0.7, max_width=1400)
            res = self.ident.identify(passes, prior_scheme_id)
        self._t("identify", t0)
        logger.info(f"[FORM] identify: {res.status} best={res.best} words={res.words_read} "
                    f"scores={[(c.form_id, c.score) for c in res.candidates[:3]]} {self.timings['identify']}s")
        return res

    def _ask_spoken(self, prompt: str) -> str:
        """Speak a pre-session question (confirm / choose / which scheme) with the
        screen showing it as the current question, then wait for the answer. A
        cancel word or the on-screen Cancel ends the session like Home."""
        self._ui(state="CONFIRM_FORM", question=prompt, listen=True)
        self.io.status("[FORM] Say yes or no")
        for _ in range(4):                      # 'repeat' (or an unrelated command) asks the same question again
            if not self.io.speak(prompt):
                return HOME
            ans = self.io.listen(constants.FORM_ANSWER_TIMEOUT_S)
            if ans in (HOME, TIMEOUT):
                return ans
            cmd = V.detect_command(ans, self.lang, self.catalog.words)
            if cmd == "cancel":
                return HOME
            if cmd in ("repeat", "change", "skip"):
                continue
            return ans
        return TIMEOUT

    def _match_form_by_speech(self, text: str, candidates: List[str]) -> Optional[str]:
        """The person names the form ('suraksha bima', 'சுரக்ஷா'): match spoken names,
        title words and scheme short names of the candidate forms (all forms when no
        candidates). No translation needed - names are compared in the session
        language and in English."""
        t = V.norm(text)
        if not t:
            return None
        pool = candidates or list(self.catalog.forms)
        best, best_hits = None, 0
        for fid in pool:
            form = self.catalog.form(fid)
            names = [form["spoken_name"].get(self.lang, ""), form["spoken_name"]["en"], form["title"].get(self.lang, ""), form["title"]["en"]] + \
                    list(form["title_patterns"].get("en", [])) + list(form["title_patterns"].get(self.lang, []))
            words = {w for n in names for w in V.norm(n).split() if len(w) >= 4}
            hits = sum(1 for w in words if w in t)
            if hits > best_hits:
                best, best_hits = fid, hits
        return best if best_hits >= 1 else None

    def resolve_form(self, res: Identification, prior_scheme_id: Optional[str]) -> Tuple[Optional[str], str]:
        """Turn an identification into a confirmed form id, asking the person as
        needed. Returns (form_id or None, reason)."""
        if res.status == "accept" or res.status == "confirm":
            fid = res.best
            for _ in range(2):
                ans = self._ask_spoken(self._p("confirm_form", title=self.catalog.spoken_name(fid, self.lang)))
                if ans in (HOME, TIMEOUT):
                    return None, "cancelled"
                yn = V.parse_yesno(ans, self.lang, self.catalog.words)
                if yn is True:
                    return fid, "confirmed"
                if yn is False:
                    other = self._match_form_by_speech(ans, [c.form_id for c in res.candidates[:3] if c.form_id != fid])
                    if other:
                        fid = other
                        continue
                    return self._ask_which(prior_scheme_id)
                named = self._match_form_by_speech(ans, [c.form_id for c in res.candidates[:3]])
                if named:
                    return named, "named"
            return None, "cancelled"
        if res.status == "choose":
            a, b = res.top_ids[0], res.top_ids[1]
            ans = self._ask_spoken(self._p("choose_form", a=self.catalog.spoken_name(a, self.lang), b=self.catalog.spoken_name(b, self.lang)))
            if ans in (HOME, TIMEOUT):
                return None, "cancelled"
            fid = self._match_form_by_speech(ans, [a, b])
            return (fid, "chosen") if fid else self._ask_which(prior_scheme_id)
        if res.status == "not_readable":
            return self._ask_which(prior_scheme_id)
        # unsupported: enough text was read and no form fits
        self.io.speak(self._p("unsupported", forms=self.catalog.supported_names(self.lang)))
        return None, "unsupported"

    def _ask_which(self, prior_scheme_id: Optional[str]) -> Tuple[Optional[str], str]:
        """The photo did not settle it: if the conversation already named a scheme
        with one form, offer that; otherwise ask which scheme the form is for."""
        if prior_scheme_id and self.catalog.by_scheme.get(prior_scheme_id):
            fid = self.catalog.by_scheme[prior_scheme_id][0]
            ans = self._ask_spoken(self._p("confirm_form", title=self.catalog.spoken_name(fid, self.lang)))
            if ans in (HOME, TIMEOUT):
                return None, "cancelled"
            if V.parse_yesno(ans, self.lang, self.catalog.words) is True:
                return fid, "prior"
            named = self._match_form_by_speech(ans, [])
            if named:
                return named, "named"
        ans = self._ask_spoken(self._p("not_readable"))
        if ans in (HOME, TIMEOUT):
            return None, "cancelled"
        fid = self._match_form_by_speech(ans, [])
        if fid:
            return fid, "named"
        self.io.speak(self._p("unsupported", forms=self.catalog.supported_names(self.lang)))
        return None, "unsupported"

    # ── the whole session ──────────────────────────────────────────────────
    def run(self, image: np.ndarray, prior_scheme_id: Optional[str] = None) -> FlowResult:
        t_all = time.perf_counter()
        self.io.status("[FORM] Reading the form")
        self._ui(state="IDENTIFYING")
        res = self.identify(image, prior_scheme_id)
        fid, reason = self.resolve_form(res, prior_scheme_id)
        if fid is None:
            self._ui(active=False, state="IDLE")
            return FlowResult("cancelled" if reason == "cancelled" else reason, timings=self.timings, seconds=time.perf_counter() - t_all)
        self.session = FormSession(self.catalog, fid, self.lang, review=constants.FORM_REVIEW_ENABLED)
        self.session.on_snapshot = self._queue_snapshot
        self._last_fp = None
        step = self.session.start(form_confirmed=True)
        outcome = "cancelled"
        while True:
            self._maybe_snapshot()
            if step.speak:
                self._ui()
                if not self.io.speak(step.speak):
                    outcome = "cancelled"; break
            if step.done:
                outcome = "done" if self.session.state == "DONE" else "cancelled"
                break
            if step.want_document:
                self.io.status("[FORM] Show the document, then press the button")
                frame = self.io.capture(constants.FORM_ANSWER_TIMEOUT_S)
                if frame is None:
                    outcome = "cancelled"; break
                if isinstance(frame, str):
                    # an on-screen command (skip / repeat / change / cancel) instead of a document
                    step = self.session.handle_answer(frame)
                    continue
                t0 = time.perf_counter()
                text = O.read_document(frame, constants.FORM_DOC_OCR_LANG).text
                self._t("doc_ocr", t0)
                step = self.session.handle_document(text)
                continue
            if step.listen:
                self.io.status("[FORM] Tap the button and answer")
                ans = self.io.listen(constants.FORM_ANSWER_TIMEOUT_S)
                if ans == HOME:
                    outcome = "cancelled"; break
                if ans == TIMEOUT:
                    outcome = "timeout"; break
                step = self.session.handle_answer(ans)
                continue
            step = self.session.handle_answer("")
        if outcome != "done":
            # Home, a walk-away or a spoken cancel: whatever was collected is already with the
            # officer (or sealed on disk) marked as stopped; the plain answers are wiped here
            handed_over = self.session.snapshot("cancelled" if outcome == "cancelled" else "timeout") if self.session.state != "CANCELLED" else False
            self.session.clear()
            self._ui(active=False, state="IDLE")
            if outcome in ("cancelled", "timeout") and self.session.state != "CANCELLED":
                self.io.speak(self._p("cancelled_saved" if handed_over and self.outbox is not None else "cancelled"))
            return FlowResult("cancelled", fid, timings=self.timings, seconds=time.perf_counter() - t_all)
        unconfirmed = self.session.unconfirmed
        payload = self.session.payload("complete_unconfirmed" if unconfirmed else "complete")
        self.session.clear()
        result = FlowResult("sent", fid, payload=None, timings=self.timings)
        if self.outbox is not None:
            self.io.speak(self._p("saved_unconfirmed" if unconfirmed else "sending"))
            self._ui(state="SENDING")
            t0 = time.perf_counter()
            status = self.outbox.submit(payload)          # seals immediately, tries to send, keeps the sealed copy on failure
            self._t("send", t0)
            del payload
            if status == "sent":
                if not unconfirmed:
                    self.io.speak(self._p("sent"))
                self._ui(state="SENT")
            else:
                self.io.speak(self._p("send_pending")); self._ui(state="PENDING"); result.outcome = "pending"
        else:
            result.payload = payload
        result.seconds = round(time.perf_counter() - t_all, 2)
        self._ui(active=False, state="IDLE")
        return result
