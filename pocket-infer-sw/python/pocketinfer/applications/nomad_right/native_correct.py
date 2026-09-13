"""
Native-script correction of recognised speech BEFORE translation.

The Hindi and Tamil recognisers write scheme names and government words the way
they sound: "पीडीएस" comes out as "इनडिएस", "मानदंड" as "महनदंड", "பிடிஎஸ்" is
translated as "BTS". The English-side corrector (scheme_intel/repository.py,
`correct()`) cannot recover those once the translator has turned them into
"IndS" or "BTS". This module does the same job one step earlier, in the person's
own script, with the same philosophy: only a word that is not a known word is
replaced, and only by a vocabulary word that sounds the same.

    correct(text, lang) -> (corrected_text, scheme_id | None)

Sound key: the word's consonant skeleton - vowel signs, independent vowels,
nasal marks and nukta removed, look-alike consonants merged. Two skeletons that
are equal or one edit apart, plus a spelling similarity check, snap the word.
Vocabulary: the scheme names of the trilingual catalogue (scheme_catalog.json)
in that script plus a short list of scheme and welfare words. Function words
are listed explicitly and never touched. Standard library only.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_CATALOG = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "nomadright", "scheme_intel", "scheme_catalog.json"))

# ── extra vocabulary (spoken forms of scheme names and welfare words) ──────────
EXTRA_VOCAB: Dict[str, List[str]] = {
    "hi": ["पीडीएस", "राशन", "कार्ड", "आयुष्मान", "भारत", "ईश्रम", "श्रम", "किसान", "पीएम", "आवास", "उज्ज्वला", "जनधन", "पेंशन", "मनरेगा",
           "बीओसीडब्ल्यू", "मानदंड", "पात्रता", "दस्तावेज़", "दस्तावेज", "आवेदन", "लाभ", "योजना", "पंजीकरण", "शिकायत", "हेल्पलाइन", "सहायता",
           "बीमा", "मज़दूर", "मजदूर", "प्रवासी", "सब्सिडी", "छात्रवृत्ति", "विधवा", "विकलांग", "इलाज", "अस्पताल", "बैंक", "खाता", "आधार",
           "फॉर्म", "फ़ॉर्म", "सरकारी", "सरकार", "पैसा", "रुपये", "मुआवज़ा", "मजदूरी", "वेतन", "गैस", "सिलेंडर", "घर", "मकान", "बुढ़ापा", "वृद्धावस्था"],
    "ta": ["பிடிஎஸ்", "ரேஷன்", "அட்டை", "ஆயுஷ்மான்", "பாரத்", "இஷ்ரம்", "ஈஷ்ரம்", "கிசான்", "பிஎம்", "ஆவாஸ்", "உஜ்வலா", "ஜன்தன்", "ஓய்வூதியம்",
           "பென்ஷன்", "தகுதி", "ஆவணம்", "ஆவணங்கள்", "விண்ணப்பம்", "பயன்", "திட்டம்", "பதிவு", "புகார்", "உதவி", "காப்பீடு", "தொழிலாளர்",
           "புலம்பெயர்", "மானியம்", "உதவித்தொகை", "விதவை", "மாற்றுத்திறனாளி", "சிகிச்சை", "மருத்துவமனை", "வங்கி", "கணக்கு", "ஆதார்", "படிவம்",
           "அரசு", "பணம்", "ரூபாய்", "இழப்பீடு", "கூலி", "சம்பளம்", "எரிவாயு", "சிலிண்டர்", "வீடு", "முதியோர்"],
}

# spoken scheme names -> catalogue id (used to hint the pipeline even if the translator mangles the name)
SCHEME_HINTS: Dict[str, Dict[str, str]] = {
    "hi": {"पीडीएस": "SCH_ONORC", "राशन": "SCH_ONORC", "आयुष्मान": "SCH_PMJAY", "ईश्रम": "SCH_ESHRAM", "श्रम कार्ड": "SCH_ESHRAM", "किसान": "SCH_PMKISAN",
           "मनरेगा": "SCH_MGNREGA", "उज्ज्वला": "SCH_PMUY", "जनधन": "SCH_PMJDY", "आवास": "SCH_PMAYG"},
    "ta": {"பிடிஎஸ்": "SCH_ONORC", "ரேஷன்": "SCH_ONORC", "ஆயுஷ்மான்": "SCH_PMJAY", "இஷ்ரம்": "SCH_ESHRAM", "ஈஷ்ரம்": "SCH_ESHRAM", "கிசான்": "SCH_PMKISAN",
           "உஜ்வலா": "SCH_PMUY", "ஜன்தன்": "SCH_PMJDY", "ஆவாஸ்": "SCH_PMAYG"},
}

# function words and everyday words: never replaced
STOP: Dict[str, Set[str]] = {
    "hi": set("के लिए क्या है हैं कौन से चाहिए मुझे मेरा मेरी मेरे आप हम और का की को में पर यह वह कैसे कब कहाँ कहां क्यों कितना कितने कितनी "
              "नहीं हाँ हां तो भी ही या एक दो तीन बताओ बताइए बताएं बताए चाहता चाहती चाहते हूँ हूं मिलेगा मिलेगी मिल सकता सकती सकते करें करना कर "
              "किस किसे किसको कोई कुछ सब सभी बहुत थोड़ा अभी फिर जो जब तब यहाँ यहां वहाँ वहां था थी थे होगा होगी हो होता होती नाम उम्र साल".split()),
    "ta": set("என்ன எப்படி எங்கே எப்போது ஏன் யார் எது எந்த வேண்டும் தேவை எனக்கு நான் நீங்கள் அது இது ஆம் இல்லை உள்ளது இருக்கு "
              "சொல்லுங்கள் தயவுசெய்து ஒரு மற்றும் அல்லது கிடைக்குமா முடியுமா எனது என் உங்கள் அவர் இந்த அந்த எல்லாம் மிகவும் இப்போது "
              "பின் யாருக்கு எவ்வளவு எத்தனை பெயர் வயது வருடம் ஆனால் தான் கூட "
              # greetings and polite words: known words, never "corrected" into scheme words
              "வணக்கம் நன்றி ஹலோ ஹாய் நமஸ்காரம் மன்னிக்கவும் தயவு சரி இல்லை ஆமாம் ஆமா வேண்டாம் நல்லது".split()),
}
STOP["hi"] |= set("नमस्ते नमस्कार प्रणाम धन्यवाद शुक्रिया हेलो हैलो हाय जी ठीक अच्छा माफ़ कीजिए कृपया बिल्कुल सही गलत".split())

_DEVA_MARKS = re.compile(r"[ऀ-ःऺ-ॏ॑-ॗॢॣ]")
_DEVA_VOWELS = re.compile(r"[ऄ-औॠॡॲ-ॷ]")
_DEVA_MERGE = {"श": "स", "ष": "स", "ण": "न", "ब": "व", "ऋ": "र", "ज्ञ": "ग"}
_TAMIL_MARKS = re.compile(r"[ஂஃா-்ௗ]")
_TAMIL_VOWELS = re.compile(r"[அ-ஔ]")
_TAMIL_MERGE = {"ண": "ந", "ன": "ந", "ள": "ல", "ழ": "ல", "ற": "ர", "ஷ": "ச", "ஸ": "ச", "ஹ": "க", "ஜ": "ச"}


def _strip_nukta(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if ch != "़")


def skeleton(word: str, lang: str) -> str:
    """Consonant skeleton with look-alike consonants merged: 'पीडीएस' and 'इनडिएस' -> 'पडस' / 'नडस'."""
    w = word.strip().lower()
    if lang == "hi":
        w = _strip_nukta(w)
        w = _DEVA_MARKS.sub("", w)
        w = _DEVA_VOWELS.sub("", w)
        for a, b in _DEVA_MERGE.items():
            w = w.replace(a, b)
    elif lang == "ta":
        w = _TAMIL_MARKS.sub("", w)
        w = _TAMIL_VOWELS.sub("", w)
        for a, b in _TAMIL_MERGE.items():
            w = w.replace(a, b)
    else:
        w = re.sub(r"[aeiouy]", "", re.sub(r"[^a-z]", "", w))
    return re.sub(r"(.)\1+", r"\1", w)


def _edit1(a: str, b: str) -> bool:
    """True when a and b are at most one edit apart."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1:
                return False
    return True


class NativeCorrector:
    def __init__(self, catalog_path: str = _CATALOG):
        self.vocab: Dict[str, Set[str]] = {"hi": set(), "ta": set()}
        self.by_key: Dict[str, Dict[str, Set[str]]] = {"hi": {}, "ta": {}}
        try:
            cat = json.load(open(catalog_path, encoding="utf-8"))
        except (OSError, ValueError):
            cat = {}
        for lang in ("hi", "ta"):
            words: Set[str] = set(EXTRA_VOCAB[lang])
            for item in cat.get(lang) or []:
                for field in ("name", "short", "category"):
                    for tok in str(item.get(field, "")).replace("-", " ").split():
                        tok = tok.strip("()।,.:;!?\"'")
                        if len(tok) >= 2:
                            words.add(tok)
            self.vocab[lang] = words
            for w in words:
                self.by_key[lang].setdefault(skeleton(w, lang), set()).add(w)

    def correct_token(self, tok: str, lang: str) -> str:
        bare = tok.strip("।,.:;!?\"'()")
        if not bare or lang not in self.vocab:
            return tok
        if bare in self.vocab[lang] or bare in STOP[lang] or bare.isdigit():
            return tok
        key = skeleton(bare, lang)
        if len(key) < 2:
            return tok
        cands: Set[str] = set(self.by_key[lang].get(key, ()))
        if len(key) >= 3:
            for k, ws in self.by_key[lang].items():
                if len(k) >= 3 and _edit1(key, k):
                    cands |= ws
        if not cands:
            return tok
        best, best_r = None, 0.0
        for w in cands:
            r = difflib.SequenceMatcher(None, bare, w).ratio()
            if r > best_r:
                best, best_r = w, r
        if best is None or best_r < 0.45:
            return tok
        return tok.replace(bare, best, 1)

    def correct(self, text: str, lang: str) -> Tuple[str, Optional[str]]:
        """Corrected text and a scheme hint (catalogue id) when a scheme name is present."""
        lang = (lang or "").lower()[:2]
        if lang not in ("hi", "ta") or not text:
            return text, None
        try:
            from pocketinfer.applications.nomad_right import kiosk_intents
            if kiosk_intents.match(text, "", lang) is not None:      # a greeting or thanks: leave it alone
                return text, None
        except Exception:
            pass
        out = [self.correct_token(tok, lang) for tok in text.split()]
        fixed = " ".join(out)
        hint = None
        for name, sid in SCHEME_HINTS[lang].items():
            if name in fixed:
                hint = sid
                break
        return fixed, hint


_shared: Optional[NativeCorrector] = None


def shared() -> NativeCorrector:
    global _shared
    if _shared is None:
        _shared = NativeCorrector()
    return _shared


def correct(text: str, lang: str) -> Tuple[str, Optional[str]]:
    return shared().correct(text, lang)
