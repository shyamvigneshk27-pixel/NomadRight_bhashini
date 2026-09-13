"""
Kiosk intents: greetings, thanks and questions about the device itself,
answered from predefined text in the selected language - no model, no
translation at run time, nothing sent anywhere.

    match(native_text, english_text, lang, scheme_terms) -> KioskHit | None

The English text (the translation the pipeline already made, or the typed /
recognised English) is matched fuzzily against short phrase lists; the native
text is matched against greeting and thanks words in Devanagari, Tamil and
romanised form, because "नमस्ते" or "வணக்கம்" may or may not come back as
"Hello". A sentence that names a scheme or a welfare topic is never taken
here: it belongs to the scheme pipeline even when it also says "can you help".
Only difflib from the standard library is used.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'")
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


@dataclass
class KioskHit:
    intent: str
    score: float
    native: str          # the answer in the selected language, spoken as written
    en: str              # the same answer in English (log / display)


# ── answers ─────────────────────────────────────────────────────────────────
RESPONSES: Dict[str, Dict[str, str]] = {
    "greeting": {
        "en": "Welcome to NomadRight. How can I help you?",
        "hi": "नोमैडराइट में आपका स्वागत है। मैं आपकी कैसे मदद कर सकता हूँ?",
        "ta": "நோமட்ரைட்டிற்கு வரவேற்கிறோம். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    },
    "thanks": {
        "en": "You are welcome. Ask me anything about government schemes.",
        "hi": "आपका स्वागत है। सरकारी योजनाओं के बारे में कुछ भी पूछिए।",
        "ta": "நன்றி. அரசுத் திட்டங்கள் பற்றி எதையும் கேளுங்கள்.",
    },
    "identity": {
        "en": "I am NomadRight. I help interstate migrant workers find and access government schemes and their rights.",
        "hi": "मैं नोमैडराइट हूँ। मैं दूसरे राज्यों में काम करने वाले प्रवासी मज़दूरों को सरकारी योजनाएँ और उनके अधिकार पाने में मदद करता हूँ।",
        "ta": "நான் நோமட்ரைட். வெளி மாநிலத்தில் வேலை செய்யும் புலம்பெயர் தொழிலாளர்களுக்கு அரசுத் திட்டங்களையும் அவர்களின் உரிமைகளையும் பெற உதவுகிறேன்.",
    },
    "capability": {
        "en": "I can tell you about government schemes, check which ones fit you, and help you fill a form by voice.",
        "hi": "मैं सरकारी योजनाओं के बारे में बता सकता हूँ, आपके लिए सही योजना ढूँढ सकता हूँ, और बोलकर फ़ॉर्म भरने में मदद कर सकता हूँ।",
        "ta": "அரசுத் திட்டங்களைப் பற்றிச் சொல்லலாம், உங்களுக்குப் பொருந்தும் திட்டங்களைச் சரிபார்க்கலாம், பேசியே படிவம் நிரப்ப உதவலாம்.",
    },
    "usage_how": {
        "en": "Tap the microphone, ask your question, and tap again when you finish. I will answer aloud.",
        "hi": "माइक्रोफ़ोन दबाइए, अपना सवाल पूछिए, और पूरा होने पर फिर दबाइए। मैं बोलकर जवाब दूँगा।",
        "ta": "மைக்ரோஃபோனைத் தட்டி, உங்கள் கேள்வியைக் கேட்டு, முடிந்ததும் மீண்டும் தட்டுங்கள். நான் பதிலைச் சொல்வேன்.",
    },
    "usage_language": {
        "en": "Yes. You can speak Tamil, Hindi or English. Choose the language on the first screen.",
        "hi": "हाँ। आप तमिल, हिंदी या अंग्रेज़ी में बोल सकते हैं। पहली स्क्रीन पर भाषा चुनिए।",
        "ta": "ஆம். நீங்கள் தமிழ், இந்தி அல்லது ஆங்கிலத்தில் பேசலாம். முதல் திரையில் மொழியைத் தேர்வு செய்யுங்கள்.",
    },
    "usage_form": {
        "en": "Yes. Go back, choose Scan a Document, show the form to the camera, and I will ask you the questions one by one.",
        "hi": "हाँ। पीछे जाकर 'दस्तावेज़ स्कैन करें' चुनिए, फ़ॉर्म कैमरे को दिखाइए, और मैं एक-एक करके सवाल पूछूँगा।",
        "ta": "ஆம். பின் சென்று 'ஆவணத்தை ஸ்கேன் செய்யவும்' என்பதைத் தேர்வு செய்து, படிவத்தைக் கேமராவிற்குக் காட்டுங்கள்; நான் கேள்விகளை ஒவ்வொன்றாகக் கேட்பேன்.",
    },
    "schemes_general": {
        "en": "Yes. Ask me about a scheme by name, or tell me about your work and family and I will find schemes for you.",
        "hi": "हाँ। किसी योजना का नाम लेकर पूछिए, या अपने काम और परिवार के बारे में बताइए, मैं आपके लिए योजनाएँ ढूँढूँगा।",
        "ta": "ஆம். ஒரு திட்டத்தின் பெயரைச் சொல்லிக் கேளுங்கள், அல்லது உங்கள் வேலை மற்றும் குடும்பத்தைப் பற்றிச் சொல்லுங்கள்; உங்களுக்கான திட்டங்களைக் கண்டறிவேன்.",
    },
}

# ── phrases (English, matched fuzzily on the translated / typed English) ──
PHRASES: Dict[str, List[str]] = {
    "greeting": ["hello", "hi", "hey", "welcome", "good morning", "good afternoon", "good evening", "good day", "namaste", "namaskar",
                 "vanakkam", "greetings", "hello there", "hi there"],
    "thanks": ["thank you", "thanks", "thank you very much", "thanks a lot", "dhanyavad", "nandri", "ok thank you", "thank you so much"],
    "identity": ["who are you", "what are you", "what is nomadright", "what is nomad right", "what is this device", "what is this machine",
                 "what is this", "tell me about yourself", "introduce yourself", "what is your name", "what is nomadright for",
                 "what is this kiosk", "who is nomadright"],
    "capability": ["what can you do", "what do you do", "what you can do", "how can you help me", "how can you help", "what is your purpose",
                   "what help can i get here", "can you help me", "i need help", "what can i ask you", "what can i ask", "what services do you provide",
                   "what services are available", "how can you help migrant workers", "what do you offer", "help me", "what all can you do",
                   "what are you able to do", "what you do"],
    "usage_how": ["how does this work", "how do i use this", "how do i ask a question", "how to use this", "how does it work", "how do i talk to you",
                  "how should i ask", "how to ask a question", "how do i speak", "how to speak"],
    "usage_language": ["can i speak in tamil", "can i speak in hindi", "can i speak in english", "can i speak tamil", "can i speak hindi",
                       "can i speak english", "do you understand tamil", "do you understand hindi", "do you understand english",
                       "which languages do you know", "can i talk in tamil", "can i talk in hindi", "can i talk in english", "do you know tamil",
                       "do you know hindi", "do you know english", "can you speak tamil", "can you speak hindi", "can you speak english"],
    "usage_form": ["can you help me fill a form", "can you help me fill the form", "can you fill a form", "help me fill a form", "i want to fill a form",
                   "can you help me fill the government form", "how do i fill a form", "can you help with a form", "fill a form", "i need to fill a form",
                   "can you fill the form for me", "how to fill a form"],
    "schemes_general": ["can you tell me about government schemes", "can you help me find a scheme", "which government scheme can help me",
                        "which scheme can help me", "can you help me access a scheme", "tell me about government schemes", "what schemes are there",
                        "which schemes are available", "can i ask about government schemes", "can you help me with schemes", "what schemes can i get",
                        "which scheme is for me", "can you tell me if i am eligible", "am i eligible for any scheme", "what schemes do you know",
                        "tell me about schemes", "government schemes", "schemes for migrant workers"],
}

# native greeting / thanks words (Devanagari, Tamil, romanised) - whole-utterance matches only
NATIVE_WORDS: Dict[str, Dict[str, List[str]]] = {
    "greeting": {
        "hi": ["नमस्ते", "नमस्कार", "प्रणाम", "हेलो", "हैलो", "हाय", "सुप्रभात", "शुभ प्रभात", "शुभ संध्या", "नमस्ते जी", "राम राम", "सत श्री अकाल", "आदाब"],
        "ta": ["வணக்கம்", "ஹலோ", "ஹாய்", "காலை வணக்கம்", "மாலை வணக்கம்", "வணக்கம் சார்", "வணக்கம்ங்க", "வணக்கம் ஐயா"],
        "en": ["namaste", "namaskar", "vanakkam", "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "welcome"],
    },
    "thanks": {
        "hi": ["धन्यवाद", "शुक्रिया", "थैंक यू", "थैंक्स", "बहुत धन्यवाद", "धन्यवाद जी"],
        "ta": ["நன்றி", "மிக்க நன்றி", "ரொம்ப நன்றி", "தேங்க்ஸ்", "நன்றிங்க"],
        "en": ["thank you", "thanks", "dhanyavad", "nandri", "shukriya", "thank you very much"],
    },
}

# words that mean the sentence is really about a scheme or welfare topic - never a kiosk intent
DEFAULT_SCHEME_TERMS: Set[str] = {
    "ration", "pds", "onorc", "ayushman", "pmjay", "shram", "eshram", "e-shram", "bocw", "kisan", "pension", "loan", "insurance", "housing",
    "awas", "ujjwala", "gas", "mgnrega", "nrega", "job card", "labour card", "labor card", "jan dhan", "bank", "money", "rupees", "subsidy",
    "scholarship", "widow", "disability", "maternity", "hospital", "treatment", "wages", "wage", "salary", "employer", "accident", "compensation",
    "document", "documents", "aadhaar", "aadhar", "eligib", "apply", "application", "benefit", "helpline", "complain", "complaint", "card",
    # a person describing themselves is asking about schemes, not about the kiosk
    "worker", "working", "farmer", "labour", "labor", "job", "family", "wife", "husband", "daughter", "son", "children", "years old", "village",
    "district", "state", "migrant", "bihar", "tamil nadu", "uttar pradesh", "jharkhand", "odisha", "bengal", "chennai", "mumbai", "delhi",
}

_INTENT_ORDER = ("greeting", "thanks", "usage_language", "usage_form", "usage_how", "identity", "capability", "schemes_general")


def _has_scheme_term(text_norm: str, terms: Iterable[str]) -> bool:
    for term in terms:
        if not term:
            continue
        if " " in term:
            if term in text_norm:
                return True
        elif re.search(r"\b" + re.escape(term), text_norm):
            return True
    return False


def _best_phrase(text_norm: str, phrases: List[str]) -> float:
    best = 0.0
    words = text_norm.split()
    for p in phrases:
        r = difflib.SequenceMatcher(None, text_norm, p).ratio()
        if r > best:
            best = r
        # token overlap for longer utterances with filler ("hello can you tell me what can you do")
        pw = p.split()
        # filler around a known phrase ("hello, can you tell me what can you do") - short utterances only,
        # so a real question that merely contains "which scheme can help me" is not taken
        if len(pw) >= 3 and len(words) <= 8 and all(w in words for w in pw):
            best = max(best, 0.9)
    return best


def match(native_text: str, english_text: str, lang: str, scheme_terms: Optional[Iterable[str]] = None) -> Optional[KioskHit]:
    lang = (lang or "en").lower()[:2]
    if lang not in ("hi", "ta", "en"):
        lang = "en"
    native = _norm(native_text)
    en = _norm(english_text)
    terms = set(scheme_terms or ()) | DEFAULT_SCHEME_TERMS

    # 1. whole-utterance greeting / thanks in the person's own words
    for intent in ("greeting", "thanks"):
        for l in (lang, "en"):
            for w in NATIVE_WORDS[intent].get(l, ()):
                if native and (native == _norm(w) or difflib.SequenceMatcher(None, native, _norm(w)).ratio() >= 0.86):
                    return KioskHit(intent, 1.0, RESPONSES[intent][lang], RESPONSES[intent]["en"])

    if not en:
        return None
    if _has_scheme_term(en, terms) or _has_scheme_term(native, ("योजना", "कार्ड", "राशन", "पेंशन", "திட்டம்", "ரேஷன்", "அட்டை", "ஓய்வூதியம்")):
        # a scheme question that happens to contain "help me" stays with the pipeline
        return None
    if len(en.split()) > 12:
        return None

    best_intent, best_score = None, 0.0
    for intent in _INTENT_ORDER:
        s = _best_phrase(en, PHRASES[intent])
        if s > best_score:
            best_intent, best_score = intent, s
    threshold = 0.9 if best_intent in ("greeting", "thanks") else 0.82
    if best_intent and best_score >= threshold:
        return KioskHit(best_intent, round(best_score, 3), RESPONSES[best_intent][lang], RESPONSES[best_intent]["en"])
    return None
