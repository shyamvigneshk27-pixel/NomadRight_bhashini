"""
Lightweight OCR for the form-filling flow: tesseract 4.1 (LSTM, fast models for
eng/hin/tam, ~4 MB each) run as a subprocess - no ML runtime stays resident, the
process is gone after every call (~110 MB while it runs, measured 2026-09-12).

Only what the flow needs:
  read_header()   - the top band of a document (titles, headings) for form identification,
                    one language at a time: 1-3 s per pass on the Orin CPU
  read_document() - a whole document (passbook, Aadhaar card) for a specific value
  find_*()        - deterministic extraction of the sensitive values from OCR text
"""
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TESSERACT = "/usr/bin/tesseract"
LANG_CODES = {"en": "eng", "hi": "hin", "ta": "tam"}
_ENV = {**os.environ, "OMP_THREAD_LIMIT": os.environ.get("NR_OCR_THREADS", "4")}
_TMP = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


@dataclass
class OcrLine:
    text: str
    conf: float          # mean word confidence 0-100
    top: int
    left: int
    height: int


@dataclass
class OcrResult:
    lang: str
    lines: List[OcrLine] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def words(self) -> int:
        return sum(len(l.text.split()) for l in self.lines)


def available() -> bool:
    return os.path.exists(TESSERACT)


def preprocess(image_bgr: np.ndarray, crop_document: bool = True, max_width: int = 1400) -> np.ndarray:
    """Document crop (the app's existing edge-detect + perspective warp), grey, resized
    so the page is about max_width px wide - the working resolution tesseract's
    fast models like (300-dpi-scale text on a 1400 px page)."""
    img = image_bgr
    if crop_document:
        try:
            from pocketinfer.applications.nomad_right import document_crop
            cropped = document_crop.auto_crop_document(img)
            # The crop looks for the largest 4-point contour; on a form that fills the
            # frame that is usually an inner table, not the page - and the title is
            # gone (measured: 1280x1657 -> 977x560, title lost). Keep the crop only
            # when it still covers most of the frame.
            fh, fw = img.shape[:2]; ch, cw = cropped.shape[:2]
            if cw * ch >= 0.6 * fw * fh and cw >= 0.7 * fw:
                img = cropped
            else:
                logger.debug(f"document crop rejected ({cw}x{ch} of {fw}x{fh}); using the full frame")
        except Exception as exc:
            logger.debug(f"document crop skipped: {exc}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if w > max_width:
        s = max_width / w
        gray = cv2.resize(gray, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)
    elif w < 900:
        s = 900 / w
        gray = cv2.resize(gray, (900, int(h * s)), interpolation=cv2.INTER_CUBIC)
    return gray


def header_band(gray: np.ndarray, frac: float = 0.42) -> np.ndarray:
    return gray[: max(60, int(gray.shape[0] * frac))]


def run_tesseract(gray: np.ndarray, lang: str, psm: int = 6, timeout: float = 25.0) -> OcrResult:
    """One tesseract pass; lines are rebuilt from the TSV output with their mean confidence."""
    code = LANG_CODES.get(lang, lang)
    fd, path = tempfile.mkstemp(prefix="nr_ocr_", suffix=".png", dir=_TMP)
    os.close(fd)
    t0 = time.perf_counter()
    try:
        cv2.imwrite(path, gray)
        proc = subprocess.run([TESSERACT, path, "-", "-l", code, "--psm", str(psm), "tsv"],
                              capture_output=True, text=True, timeout=timeout, env=_ENV)
        out = proc.stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    res = OcrResult(lang=lang, seconds=time.perf_counter() - t0)
    current: Optional[Tuple[Tuple[int, int, int], List[str], List[float], int, int, int]] = None
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        key = (int(cols[2]), int(cols[3]), int(cols[4]))
        word = cols[11].strip()
        try:
            conf = float(cols[10])
        except ValueError:
            conf = -1.0
        if not word or conf < 0:
            continue
        if current is None or current[0] != key:
            if current is not None:
                res.lines.append(_line(current))
            current = (key, [word], [conf], int(cols[7]), int(cols[6]), int(cols[9]))
        else:
            current[1].append(word); current[2].append(conf)
    if current is not None:
        res.lines.append(_line(current))
    return res


def _line(cur) -> OcrLine:
    _key, words, confs, top, left, height = cur
    return OcrLine(text=" ".join(words), conf=sum(confs) / len(confs), top=top, left=left, height=height)


def read_header(image_bgr: np.ndarray, langs: Sequence[str] = ("en",), crop_document: bool = True,
                frac: float = 0.42, band_width: int = 1800) -> Dict[str, OcrResult]:
    """The header band, upscaled to band_width so small webcam captures still give
    tesseract ~30 px tall title glyphs (measured: 1280-px frames lost small titles)."""
    gray = preprocess(image_bgr, crop_document=crop_document)
    band = header_band(gray, frac)
    if band.shape[1] < band_width:
        s = band_width / band.shape[1]
        band = cv2.resize(band, (band_width, int(band.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    return {lang: run_tesseract(band, lang, psm=6) for lang in langs}


def read_document(image_bgr: np.ndarray, lang: str = "en", crop_document: bool = True, psm: int = 4,
                  frac: float = 1.0, max_width: int = 1600) -> OcrResult:
    """A document (or its top `frac`) in one pass - passbooks, cards, and the
    identification escalation (frac 0.7: titles sit in the top part of every form)."""
    gray = preprocess(image_bgr, crop_document=crop_document, max_width=max_width)
    if frac < 1.0:
        gray = gray[: int(gray.shape[0] * frac)]
    return run_tesseract(gray, lang, psm=psm, timeout=40.0)


# ── deterministic value extraction ──────────────────────────────────────────
_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
               [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
               [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
               [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_check_digit(base: str) -> str:
    """The check digit that makes base+digit pass verhoeff_ok (used to build test numbers)."""
    c = 0
    for i, ch in enumerate(reversed(base)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def verhoeff_ok(number: str) -> bool:
    """Aadhaar numbers carry a Verhoeff check digit - a wrong OCR digit fails it."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


_OCR_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "z": "2"})


def _digit_runs(text: str, fix: bool = True) -> List[str]:
    t = text.translate(_OCR_DIGIT_FIX) if fix else text
    return [re.sub(r"[\s\-]", "", m) for m in re.findall(r"(?<![0-9])(?:\d[\s\-]?){8,20}(?![0-9])", t)]


def find_aadhaar(text: str) -> Optional[str]:
    for run in _digit_runs(text):
        if len(run) == 12 and run[0] in "23456789" and verhoeff_ok(run):
            return run
    return None


def find_mobile(text: str) -> Optional[str]:
    for m in re.findall(r"(?<![0-9])[6-9]\d{9}(?![0-9])", re.sub(r"[\s\-]", "", text)):
        return m
    return None


def find_ifsc(text: str) -> Optional[str]:
    """IFSC = 4 letters, a zero, 6 alphanumerics. OCR turns the zero into O/Q/D
    (measured: 'SBINQ001234') and may split the code with spaces ('SBIN 0001234'),
    so up to three neighbouring tokens are joined. A joined window made only of
    plain words is not a code ('BANK OF INDIA' would otherwise pass)."""
    tokens = text.upper().split()
    best: Optional[str] = None
    for i in range(len(tokens)):
        for w in (1, 2, 3):
            if i + w > len(tokens):
                break
            window = tokens[i:i + w]
            raw = "".join(window)
            labelled = raw.startswith("IFSC")
            raw = re.sub(r"^IFSC(?:CODE)?", "", raw)
            raw = re.sub(r"[^A-Z0-9]", "", raw)
            m = re.fullmatch(r"([A-Z]{4})[0OQD]([A-Z0-9]{6})", raw)
            if not m:
                continue
            if w > 1 and not labelled and all(t.isalpha() for t in window):
                continue
            code = f"{m.group(1)}0{m.group(2)}"
            if labelled:
                return code
            best = best or code
    return best


def find_account_number(text: str, exclude: Sequence[str] = ()) -> Optional[str]:
    """Bank account numbers are 9-18 digits; the longest run that is not an Aadhaar,
    a mobile number or an IFSC fragment wins (passbooks print the account number
    once, prominently)."""
    best = None
    for run in _digit_runs(text):
        if run in exclude or (len(run) == 12 and verhoeff_ok(run)) or (len(run) == 10 and run[0] in "6789"):
            continue
        if 9 <= len(run) <= 18 and (best is None or len(run) > len(best)):
            best = run
    return best


def find_pan(text: str) -> Optional[str]:
    m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text.upper().replace(" ", ""))
    return m.group(1) if m else None


def find_pincode(text: str) -> Optional[str]:
    m = re.search(r"(?<![0-9])[1-9]\d{5}(?![0-9])", re.sub(r"[\s\-]", "", text))
    return m.group(0) if m else None
