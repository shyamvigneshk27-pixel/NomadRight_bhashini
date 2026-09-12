"""
Deterministic form identification from OCR text - no language model.

Signals per form (from forms.json): title patterns in the languages printed on the
form, weighted keywords, and discriminators against look-alike forms (PMSBY and
PMJJBY share "Pradhan Mantri ... Bima Yojana"). Each candidate gets a confidence
in [0, 1]; the decision rule never guesses:

  ACCEPT   best >= ACCEPT_MIN and the runner-up is at least MARGIN_MIN behind
  CONFIRM  best >= CONFIRM_MIN                (ask "is this the X form?")
  CHOOSE   two candidates close together      (ask "X or Y?")
  UNSUPPORTED  enough text was read, no form fits
  NOT_READABLE hardly any text was read
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.ocr import OcrResult

ACCEPT_MIN, CONFIRM_MIN, MARGIN_MIN, MIN_WORDS = 0.72, 0.45, 0.15, 6
PRIOR_BONUS = 0.08          # the scheme the conversation already settled on
KW_FULL, KW_STRONG = 6.0, 8.0


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    t = t.replace("’", "'").replace("‘", "'")
    # keep Indic combining marks (matras, virama): \w does not match them, and
    # stripping them turns "पेंशन" into "प शन" for both pattern and text - lossy
    t = re.sub(r"[^\w\s'\-\u0900-\u0DFF]", " ", t)
    return " ".join(t.split())


@dataclass
class Candidate:
    form_id: str
    score: float
    title_score: float
    keyword_score: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class Identification:
    status: str                              # accept | confirm | choose | unsupported | not_readable
    candidates: List[Candidate]
    words_read: int
    best: Optional[str] = None
    margin: float = 0.0

    @property
    def top_ids(self) -> List[str]:
        return [c.form_id for c in self.candidates[:2]]


SHORT_PATTERN = 14          # shorter patterns only count when they appear verbatim
FUZZY_MIN = {24: 0.80, 0: 0.86}   # pattern length threshold -> minimum fuzzy ratio that counts


def _best_match(pattern: str, lines: Sequence[str]) -> float:
    """Best similarity of a title pattern against any OCR line or pair of adjacent
    lines (titles wrap), both as a whole and as the best-aligned window. Short
    patterns must appear verbatim; longer ones need a high fuzzy ratio - a
    generic 'form 1' fuzzily matching 'form no' is what run 1 got wrong."""
    p = normalize(pattern)
    if not p:
        return 0.0
    best = 0.0
    cands = list(lines) + [f"{a} {b}" for a, b in zip(lines, lines[1:])]
    pw = len(p)
    for line in cands:
        l = normalize(line)
        if not l:
            continue
        if p in l:
            return 1.0
        r = difflib.SequenceMatcher(None, p, l).ratio()
        if len(l) > pw:  # sliding window of the pattern's length, cheap and good enough for titles
            step = max(1, pw // 3)
            for i in range(0, len(l) - pw + 1, step):
                r = max(r, difflib.SequenceMatcher(None, p, l[i:i + pw]).ratio())
        best = max(best, r)
    if len(p) < SHORT_PATTERN and best < 1.0:
        return 0.0                      # short patterns count only verbatim
    return best


class FormIdentifier:
    def __init__(self, catalog: FormCatalog):
        self.catalog = catalog

    def score(self, ocr: Dict[str, OcrResult], prior_scheme_id: Optional[str] = None) -> List[Candidate]:
        lines_by_lang = {lang: [l.text for l in res.lines if l.conf >= 30 or len(l.text) > 12] for lang, res in ocr.items()}
        all_text = normalize("\n".join(res.text for res in ocr.values()))
        out: List[Candidate] = []
        for fid, form in self.catalog.forms.items():
            ev: List[str] = []
            # titles: the language of the OCR pass decides which patterns apply; English
            # patterns are tried against every pass (forms print English titles too)
            t_best, best_pat = 0.0, ""
            for lang, lines in lines_by_lang.items():
                pats = list(form["title_patterns"].get("en", []))
                if lang != "en":
                    pats += form["title_patterns"].get(lang, [])
                for pat in pats:
                    s = _best_match(pat, lines)
                    if s > t_best:
                        t_best, best_pat = s, pat
            # a fuzzy match below the floor for its length is a weak title: it can ask
            # for confirmation but never accept on its own
            floor = FUZZY_MIN[24] if len(normalize(best_pat if t_best else "")) >= 24 else FUZZY_MIN[0]
            strong_title = t_best >= floor
            if 0.6 <= t_best < floor:
                ev.append(f"title?{t_best:.2f}:{best_pat}")
            elif t_best >= floor:
                ev.append(f"title~{t_best:.2f}:{best_pat}")
            if t_best < 0.6:
                t_best = 0.0
            elif not strong_title:
                t_best = t_best * 0.75
            # keywords
            kw = form.get("keywords", {})
            total = sum(kw.values()) or 1.0
            hit = 0.0
            for word, weight in kw.items():
                if normalize(word) in all_text:
                    hit += weight
                    ev.append(f"kw:{word}")
            # two strong keywords (weight ~6) are full keyword credit - the per-script
            # keyword lists must not dilute each other
            k_score = min(1.0, hit / KW_FULL)
            score = 0.65 * t_best + 0.35 * k_score
            if t_best == 0.0 and hit >= KW_STRONG:
                score = max(score, 0.5)          # strong keywords, no title read: worth asking
            # a title alone (no keyword at all) can at most ask for confirmation, never accept;
            # keywords alone (no title) likewise
            if hit == 0.0 or t_best == 0.0 or not strong_title:
                score = min(score, ACCEPT_MIN - 0.05)
            if prior_scheme_id and form["scheme_id"] == prior_scheme_id:
                score += PRIOR_BONUS
                ev.append("prior")
            out.append(Candidate(fid, round(min(score, 1.0), 3), round(t_best, 3), round(k_score, 3), ev))
        # discriminators: look-alike pairs
        by_id = {c.form_id: c for c in out}
        for fid, form in self.catalog.forms.items():
            for other, words in (form.get("discriminators", {}).get("against") or {}).items():
                mine = by_id[fid]
                if other in by_id and any(normalize(w) in all_text for w in words):
                    by_id[other].score = round(max(0.0, by_id[other].score - 0.25), 3)
                    by_id[other].evidence.append(f"discriminated by {fid}")
        out.sort(key=lambda c: -c.score)
        return out

    def identify(self, ocr: Dict[str, OcrResult], prior_scheme_id: Optional[str] = None) -> Identification:
        words = sum(res.words for res in ocr.values())
        cands = self.score(ocr, prior_scheme_id)
        if not cands:
            return Identification("not_readable", [], words)
        best, second = cands[0], (cands[1] if len(cands) > 1 else None)
        margin = best.score - (second.score if second else 0.0)
        if words < MIN_WORDS and best.score < ACCEPT_MIN:
            return Identification("not_readable", cands, words, None, margin)
        if best.score >= ACCEPT_MIN and margin >= MARGIN_MIN:
            return Identification("accept", cands, words, best.form_id, margin)
        if best.score >= CONFIRM_MIN:
            if second and second.score >= CONFIRM_MIN and margin < MARGIN_MIN:
                return Identification("choose", cands, words, best.form_id, margin)
            return Identification("confirm", cands, words, best.form_id, margin)
        return Identification("unsupported" if words >= MIN_WORDS else "not_readable", cands, words, None, margin)
