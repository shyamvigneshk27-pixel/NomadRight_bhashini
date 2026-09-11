"""
Vision helpers: keep frames small before they reach Qwen3-VL, and turn
Qwen's description of a photographed document into structured facts.

Measured 2026-09-11 (Qwen3-VL-2B Q4_K_M, num_ctx 2048): a 1920x1080 camera
frame is 2230 prompt tokens and Ollama rejects it outright ("request (2230
tokens) exceeds the available context size (2048 tokens)"); every size from
640x360 to 1280x720 costs ~1230 tokens (Ollama's minimum image budget) and
reads a synthetic ration card correctly in ~5.7 s. Frames are therefore
downscaled to VISION_MAX_SIDE on the longest side - never upscaled.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel import gazetteer
from pocketinfer.applications.nomad_right.scheme_intel.models import Profile

logger = logging.getLogger(__name__)

_DOC_TYPES = [
    ("ration_card", r"\bration card\b|\bnfsa\b|\bfair price shop\b"),
    ("aadhaar", r"\baadhaa?r\b|\buidai\b"),
    ("labour_card", r"\blabou?r card\b|\bconstruction workers?'? (?:welfare )?board\b|\bbocw\b"),
    ("eshram_card", r"\be-?shram\b|\buniversal account number\b"),
    ("ayushman_card", r"\bayushman\b|\bpm-?jay\b|\bgolden card\b"),
    ("job_card", r"\bjob card\b|\bmgnrega\b|\bnrega\b"),
    ("bank_passbook", r"\bpassbook\b|\bbank account\b"),
    ("pan_card", r"\bpan card\b|\bincome tax department\b"),
    ("voter_id", r"\bvoter\b|\belection commission\b|\bepic\b"),
]
_DOC_TYPES_RE = [(k, re.compile(p)) for k, p in _DOC_TYPES]


def downscale_jpeg(jpg: bytes, max_side: int = C.VISION_MAX_SIDE,
                   quality: int = C.VISION_JPEG_QUALITY) -> Tuple[bytes, Dict[str, int]]:
    """Returns (jpeg bytes, info). Anything that cannot be decoded is passed through untouched."""
    info = {"in_bytes": len(jpg or b"")}
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpg, info
        h, w = img.shape[:2]
        info.update({"in_w": w, "in_h": h})
        scale = max_side / float(max(h, w))
        if scale >= 1.0:
            return jpg, info
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return jpg, info
        out = buf.tobytes()
        info.update({"out_w": img.shape[1], "out_h": img.shape[0], "out_bytes": len(out)})
        return out, info
    except Exception as exc:
        logger.warning(f"[SI] frame downscale skipped: {exc}")
        return jpg, info


def document_facts(description: str) -> Tuple[Optional[str], Profile]:
    """(document type, facts) read from Qwen's description of a photographed document."""
    p = Profile()
    t = (description or "").lower()
    doc_type = None
    for kind, rx in _DOC_TYPES_RE:
        if rx.search(t):
            doc_type = kind
            break
    state = None
    for pl in gazetteer.find_places(t):
        state = pl.state
        break
    if doc_type == "ration_card":
        p.set("ration_card", True, "photographed ration card")
        if state:
            p.set("ration_card_state", state, "state on the card")
        m = re.search(r"\b(aay|antyodaya|phh|priority household|bpl)\b", t)
        if m:
            code = {"aay": "AAY", "antyodaya": "AAY", "phh": "PHH", "priority household": "PHH", "bpl": "BPL"}[m.group(1)]
            p.set("ration_card_type", code, "card type on the card")
            if code in ("AAY", "PHH"):
                p.set("nfsa_beneficiary", True, "card type on the card")
    elif doc_type == "aadhaar":
        p.set("aadhaar", True, "photographed Aadhaar")
    elif doc_type == "labour_card":
        p.set("existing_scheme_membership", ["SCH_BOCW"], "photographed labour card")
    elif doc_type == "eshram_card":
        p.set("e_shram_registration", True, "photographed e-Shram card")
        p.set("existing_scheme_membership", ["SCH_ESHRAM"], "photographed e-Shram card")
    elif doc_type == "ayushman_card":
        p.set("existing_scheme_membership", ["SCH_PMJAY"], "photographed Ayushman card")
    elif doc_type == "bank_passbook":
        p.set("has_bank_account", True, "photographed passbook")
    return doc_type, p
