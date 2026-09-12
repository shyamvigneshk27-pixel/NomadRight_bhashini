"""
Deterministic parsing of spoken answers into typed field values - no model.

Every parser gets the ASR text in the selected language and returns a Parsed
(value to store, text to read back) or raises Invalid(prompt_key) with the prompt
the kiosk should speak instead ("invalid_number", "invalid_date", ...). Command
words (repeat / change / skip / cancel) are detected first by the session.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pocketinfer.applications.nomad_right.formfill.ocr import verhoeff_ok


class Invalid(Exception):
    def __init__(self, prompt_key: str = "retry"):
        super().__init__(prompt_key)
        self.prompt_key = prompt_key


@dataclass
class Parsed:
    value: object
    display: str


def norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower().strip()
    t = re.sub(r"[।\.,;:!?\"'()\[\]{}]+", " ", t)
    return " ".join(t.split())


# ── numbers ──────────────────────────────────────────────────────────────────
_UNITS = {
    "en": {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
           "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "double": -2, "triple": -3},
    "hi": {"शून्य": 0, "जीरो": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
           "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
           "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
           "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
           "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
           "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
           "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
           "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
           "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
           "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
           "डबल": -2, "ट्रिपल": -3},
    "ta": {"பூஜ்யம்": 0, "ஜீரோ": 0, "சைபர்": 0, "ஒன்று": 1, "ஒண்ணு": 1, "இரண்டு": 2, "ரெண்டு": 2, "மூன்று": 3, "மூணு": 3, "நான்கு": 4, "நாலு": 4, "ஐந்து": 5, "அஞ்சு": 5,
           "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9, "ஒம்பது": 9, "பத்து": 10, "பதினொன்று": 11, "பதினொண்ணு": 11, "பன்னிரண்டு": 12, "பன்னிரெண்டு": 12, "பதின்மூன்று": 13,
           "பதினான்கு": 14, "பதினாலு": 14, "பதினைந்து": 15, "பதினஞ்சு": 15, "பதினாறு": 16, "பதினேழு": 17, "பதினெட்டு": 18, "பத்தொன்பது": 19, "இருபது": 20, "முப்பது": 30,
           "நாற்பது": 40, "ஐம்பது": 50, "அறுபது": 60, "எழுபது": 70, "எண்பது": 80, "தொண்ணூறு": 90, "டபுள்": -2, "ட்ரிபிள்": -3},
}
_TA_TENS_PREFIX = {"இருபத்து": 20, "இருபத்தி": 20, "முப்பத்து": 30, "முப்பத்தி": 30, "நாற்பத்து": 40, "நாற்பத்தி": 40, "ஐம்பத்து": 50, "ஐம்பத்தி": 50,
                   "அறுபத்து": 60, "அறுபத்தி": 60, "எழுபத்து": 70, "எழுபத்தி": 70, "எண்பத்து": 80, "எண்பத்தி": 80, "தொண்ணூற்று": 90, "தொண்ணூத்தி": 90}
_SCALES = {"en": {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000},
           "hi": {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000},
           "ta": {"நூறு": 100, "நூற்று": 100, "ஆயிரம்": 1000, "ஆயிரத்து": 1000, "லட்சம்": 100000}}
_TA_HUNDREDS = {"இருநூறு": 200, "இருநூற்று": 200, "முந்நூறு": 300, "முந்நூற்று": 300, "நானூறு": 400, "நானூற்று": 400, "ஐநூறு": 500, "ஐநூற்று": 500,
                "அறுநூறு": 600, "அறுநூற்று": 600, "எழுநூறு": 700, "எழுநூற்று": 700, "எண்ணூறு": 800, "எண்ணூற்று": 800, "தொள்ளாயிரம்": 900, "தொள்ளாயிரத்து": 900}
_TA_THOUSANDS = {"இரண்டாயிரத்து": 2000, "ரெண்டாயிரத்து": 2000, "மூவாயிரத்து": 3000, "மூணாயிரத்து": 3000, "நான்காயிரத்து": 4000, "நாலாயிரத்து": 4000, "ஐயாயிரத்து": 5000,
                 "அஞ்சாயிரத்து": 5000, "ஆறாயிரத்து": 6000, "ஏழாயிரத்து": 7000, "எட்டாயிரத்து": 8000, "ஒன்பதாயிரத்து": 9000, "பத்தாயிரத்து": 10000, "இருபதாயிரத்து": 20000,
                 "ஐம்பதாயிரத்து": 50000, "இரண்டாயிரம்": 2000, "ரெண்டாயிரம்": 2000, "மூவாயிரம்": 3000, "மூணாயிரம்": 3000, "நான்காயிரம்": 4000, "நாலாயிரம்": 4000, "ஐயாயிரம்": 5000, "அஞ்சாயிரம்": 5000,
                 "ஆறாயிரம்": 6000, "ஏழாயிரம்": 7000, "எட்டாயிரம்": 8000, "ஒன்பதாயிரம்": 9000, "பத்தாயிரம்": 10000, "இருபதாயிரம்": 20000, "முப்பதாயிரம்": 30000,
                 "ஐம்பதாயிரம்": 50000, "ஒரு லட்சம்": 100000}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")


def _tokens_to_number(tokens: List[str], lang: str) -> Optional[int]:
    """'one lakh twenty five thousand' / 'पाँच सौ' / 'இரண்டாயிரத்து ஐந்நூறு' -> int (None if no number words)."""
    units, scales = _UNITS.get(lang, {}), _SCALES.get(lang, {})
    total, current, seen = 0, 0, False
    for tok in tokens:
        tok = tok.strip()
        if tok.isdigit():
            current += int(tok); seen = True; continue
        if lang == "ta":
            if tok in _TA_THOUSANDS:
                total += _TA_THOUSANDS[tok]; seen = True; continue
            if tok in _TA_HUNDREDS:
                current += _TA_HUNDREDS[tok]; seen = True; continue
            pref = next((v for k, v in _TA_TENS_PREFIX.items() if tok.startswith(k)), None)
            if pref is not None:
                rest = tok[len(next(k for k in _TA_TENS_PREFIX if tok.startswith(k))):]
                current += pref + (units.get(rest, 0) if rest else 0); seen = True; continue
        if tok in units and units[tok] >= 0:
            current += units[tok]; seen = True
        elif tok in scales:
            s = scales[tok]
            if s >= 1000:
                total += (current if current else 1) * s; current = 0
            else:
                current = (current if current else 1) * s
            seen = True
        elif tok in ("and", "और", "मे", "மற்றும்"):
            continue
        else:
            return None if not seen else total + current
    return (total + current) if seen else None


def digits_from_speech(text: str, lang: str) -> str:
    """Digit sequences as people read IDs: '9 8 7 6', 'नौ आठ सात', 'double two' -> '9876', '987', '22'."""
    t = norm(text).translate(_DEVANAGARI_DIGITS).translate(_TAMIL_DIGITS)
    out, repeat = [], 1
    for tok in t.split():
        if tok.isdigit():
            out.append(tok * repeat if repeat > 1 else tok); repeat = 1; continue
        u = _UNITS.get(lang, {}).get(tok, None)
        if u is None:
            u = _UNITS["en"].get(tok)
        if u is None:
            continue
        if u < 0:
            repeat = -u
        elif u < 10:
            out.append(str(u) * repeat); repeat = 1
        elif u < 100:
            out.append(str(u)); repeat = 1
    return "".join(out)


def parse_integer(text: str, lang: str, lo: Optional[int] = None, hi: Optional[int] = None) -> Parsed:
    t = norm(text).translate(_DEVANAGARI_DIGITS).translate(_TAMIL_DIGITS)
    m = re.search(r"\d[\d,]*", t)
    value = int(m.group(0).replace(",", "")) if m else _tokens_to_number(t.split(), lang)
    if value is None:
        value = _tokens_to_number([w for w in t.split() if w in _UNITS.get(lang, {}) or w in _SCALES.get(lang, {}) or w in _UNITS["en"] or w in _SCALES["en"]], lang)
    if value is None:
        raise Invalid("invalid_number")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise Invalid("invalid_number")
    return Parsed(value, str(value))


def parse_amount(text: str, lang: str, lo=None, hi=None) -> Parsed:
    return parse_integer(text, lang, lo, hi)


def parse_number(text: str, lang: str, lo=None, hi=None) -> Parsed:
    t = norm(text).translate(_DEVANAGARI_DIGITS)
    m = re.search(r"\d+(?:[.,]\d+)?", t)
    if m:
        v = float(m.group(0).replace(",", "."))
    else:
        iv = _tokens_to_number(t.split(), lang)
        if iv is None:
            raise Invalid("invalid_number")
        v = float(iv)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise Invalid("invalid_number")
    return Parsed(v, f"{v:g}")


def parse_digits(text: str, lang: str, length: Optional[int] = None, lengths: Sequence[int] = ()) -> str:
    d = digits_from_speech(text, lang)
    if length and len(d) != length:
        raise Invalid("invalid_number")
    if lengths and len(d) not in lengths:
        raise Invalid("invalid_number")
    return d


def parse_phone(text: str, lang: str) -> Parsed:
    d = parse_digits(text, lang)
    d = d[-10:] if len(d) in (11, 12) and d.startswith(("0", "91")) else d
    if len(d) != 10 or d[0] not in "6789":
        raise Invalid("invalid_number")
    return Parsed(d, " ".join(d[i:i + 2] for i in range(0, 10, 2)))


def parse_aadhaar(text: str, lang: str) -> Parsed:
    d = parse_digits(text, lang)
    if len(d) != 12 or d[0] in "01" or not verhoeff_ok(d):
        raise Invalid("invalid_number")
    return Parsed(d, " ".join(d[i:i + 4] for i in range(0, 12, 4)))


def parse_account_number(text: str, lang: str) -> Parsed:
    d = parse_digits(text, lang)
    if not 9 <= len(d) <= 18:
        raise Invalid("invalid_number")
    return Parsed(d, " ".join(d[i:i + 3] for i in range(0, len(d), 3)))


def parse_pincode(text: str, lang: str) -> Parsed:
    d = parse_digits(text, lang)
    if len(d) != 6 or d[0] == "0":
        raise Invalid("invalid_number")
    return Parsed(d, " ".join(d))


_MONTHS = {
    "en": ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
    "hi": ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "ta": ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"],
}
_MONTH_ALIASES = {"hi": {"सितम्बर": 9, "अक्तूबर": 10, "नवम्बर": 11, "दिसम्बर": 12, "फ़रवरी": 2}, "ta": {"ஆகஸ்டு": 8, "செப்டெம்பர்": 9}, "en": {"sept": 9}}


def _month(tok: str, lang: str) -> Optional[int]:
    for l in (lang, "en"):
        for i, m in enumerate(_MONTHS[l], 1):
            if tok == m or (len(tok) >= 3 and m.startswith(tok)):
                return i
        if tok in _MONTH_ALIASES.get(l, {}):
            return _MONTH_ALIASES[l][tok]
    return None


def parse_date(text: str, lang: str) -> Parsed:
    """'15 8 1985', '15/08/1985', '15 अगस्त 1985', 'பதினைந்து ஆகஸ்ட் ஆயிரத்து தொள்ளாயிரத்து எண்பத்தி ஐந்து' -> 1985-08-15."""
    t = norm(text).translate(_DEVANAGARI_DIGITS).translate(_TAMIL_DIGITS).replace("/", " ").replace("-", " ").replace(".", " ")
    toks = t.split()
    nums: List[int] = []
    month: Optional[int] = None
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.isdigit():
            nums.append(int(tok)); i += 1; continue
        mo = _month(tok, lang)
        if mo:
            month = mo; i += 1; continue
        # a run of number words -> one number
        j = i
        while j < len(toks) and (toks[j] in _UNITS.get(lang, {}) or toks[j] in _SCALES.get(lang, {}) or toks[j] in _UNITS["en"] or toks[j] in _SCALES["en"]
                                 or (lang == "ta" and (toks[j] in _TA_HUNDREDS or toks[j] in _TA_THOUSANDS or any(toks[j].startswith(k) for k in _TA_TENS_PREFIX)))):
            j += 1
        if j > i:
            v = _tokens_to_number(toks[i:j], lang)
            if v is not None:
                nums.append(v)
            i = j
        else:
            i += 1
    day = mth = year = None
    if month:
        rest = nums
        for n in rest:
            if day is None and 1 <= n <= 31:
                day = n
            elif year is None and (n >= 1900 or 0 <= n <= 99):
                year = n
        mth = month
    elif len(nums) >= 3:
        day, mth, year = nums[0], nums[1], nums[2]
    if day is None or mth is None or year is None:
        raise Invalid("invalid_date")
    if year < 100:
        year += 1900 if year > 30 else 2000
    if not (1 <= day <= 31 and 1 <= mth <= 12 and 1900 <= year <= 2030):
        raise Invalid("invalid_date")
    iso = f"{year:04d}-{mth:02d}-{day:02d}"
    return Parsed(iso, f"{day} {_MONTHS[lang][mth - 1] if lang in _MONTHS else mth} {year}")


def parse_choice(text: str, lang: str, options: List[dict]) -> Parsed:
    t = norm(text)
    best, best_len = None, 0
    for opt in options:
        for syn in list(opt.get(lang, [])) + list(opt.get("en", [])):
            s = norm(syn)
            if s and (s == t or re.search(r"(?<!\S)" + re.escape(s) + r"(?!\S)", t) or (len(s) >= 4 and s in t)):
                if len(s) > best_len:
                    best, best_len = opt, len(s)
    if best is None:
        # a bare number for numbered/amount options ("2000")
        d = re.sub(r"\D", "", t)
        for opt in options:
            if d and opt["value"] == d:
                best = opt
    if best is None:
        raise Invalid("invalid_choice")
    spoken = (best.get(lang) or best.get("en") or [best["value"]])[0]
    return Parsed(best["value"], spoken)


def parse_yesno(text: str, lang: str, words: Dict[str, Dict[str, List[str]]]) -> Optional[bool]:
    t = norm(text)
    yes = [norm(w) for w in words["yes"].get(lang, []) + words["yes"]["en"]]
    no = [norm(w) for w in words["no"].get(lang, []) + words["no"]["en"]]
    # "no" words first: "जी नहीं" contains "जी" (a yes word)
    if any(re.search(r"(?<!\S)" + re.escape(w) + r"(?!\S)", t) for w in no):
        return False
    if any(re.search(r"(?<!\S)" + re.escape(w) + r"(?!\S)", t) for w in yes):
        return True
    return None


def detect_command(text: str, lang: str, words: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    t = norm(text)
    for kind in ("cancel", "repeat", "change", "skip"):
        for w in words[kind].get(lang, []) + words[kind]["en"]:
            w = norm(w)
            if t == w or re.search(r"(?<!\S)" + re.escape(w) + r"(?!\S)", t):
                return kind
    return None


_NAME_FILLERS = {
    "en": [r"^my name is\s+", r"^i am\s+", r"^it is\s+", r"^it's\s+", r"^the name is\s+", r"^name\s+"],
    "hi": [r"^मेरा नाम\s+", r"^मेरा\s+", r"^नाम\s+", r"\s+है$", r"\s+हैं$", r"^मैं\s+", r"^जी\s+"],
    "ta": [r"^என் பெயர்\s+", r"^என்னுடைய பெயர்\s+", r"^எனது பெயர்\s+", r"^பெயர்\s+", r"^நான்\s+", r"\s+தான்$", r"\s+ங்க$"],
}


def parse_text(text: str, lang: str, min_len: int = 2, name: bool = False) -> Parsed:
    t = (text or "").strip()
    t = re.sub(r"[।\.!?]+$", "", t).strip()
    if name:
        for pat in _NAME_FILLERS.get(lang, []) + _NAME_FILLERS["en"]:
            t = re.sub(pat, "", t, flags=re.I).strip()
    if len(t) < min_len:
        raise Invalid("retry")
    return Parsed(t, t)


def parse_email(text: str, lang: str) -> Parsed:
    t = norm(text).replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
    if not re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", t):
        raise Invalid("retry")
    return Parsed(t, t)


def parse_by_type(ftype: str, text: str, lang: str, field: dict, words: Dict) -> Parsed:
    v = field.get("validation") or {}
    if ftype == "integer":
        return parse_integer(text, lang, v.get("min"), v.get("max"))
    if ftype == "amount":
        return parse_amount(text, lang, v.get("min"), v.get("max"))
    if ftype == "number":
        return parse_number(text, lang, v.get("min"), v.get("max"))
    if ftype == "phone":
        return parse_phone(text, lang)
    if ftype == "aadhaar":
        return parse_aadhaar(text, lang)
    if ftype == "account_number":
        return parse_account_number(text, lang)
    if ftype == "pincode":
        return parse_pincode(text, lang)
    if ftype == "date":
        return parse_date(text, lang)
    if ftype == "choice":
        return parse_choice(text, lang, field["options"])
    if ftype == "email":
        return parse_email(text, lang)
    if ftype == "ifsc":
        t = norm(text).upper().replace(" ", "")
        m = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", t)
        if not m:
            raise Invalid("retry")
        return Parsed(m.group(0), " ".join(m.group(0)))
    if ftype == "name":
        return parse_text(text, lang, name=True)
    return parse_text(text, lang)
