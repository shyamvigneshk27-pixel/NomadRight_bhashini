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

# punctuation only: Devanagari and Tamil vowel signs and viramas are combining marks, which
# \w does not match - removing them split "क्या" into "क या" and broke every native match
_PUNCT = re.compile(r"[^\w\s'\u0900-\u0963\u0966-\u097F\u0B80-\u0BFF]+", re.UNICODE)


def _norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("\u200c", "").replace("\u200d", "")
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
    "eligibility": {
        "en": "Yes. Tell me your age, your work, where you are from and where you live now, and I will check which schemes you may be eligible for.",
        "hi": "हाँ। अपनी उम्र, अपना काम, आप कहाँ से हैं और अभी कहाँ रहते हैं, यह बताइए, मैं देखूँगा कि आप किन योजनाओं के पात्र हो सकते हैं।",
        "ta": "ஆம். உங்கள் வயது, உங்கள் வேலை, நீங்கள் எந்த ஊரைச் சேர்ந்தவர், இப்போது எங்கே வசிக்கிறீர்கள் என்று சொல்லுங்கள்; நீங்கள் எந்தத் திட்டங்களுக்குத் தகுதியானவர் என்று பார்க்கிறேன்.",
    },
    "help": {
        "en": "Just tell me what you need in your own words. For example: which scheme is for me, how do I get a ration card, or help me fill a form.",
        "hi": "आपको जो चाहिए, अपने शब्दों में बताइए। जैसे: मेरे लिए कौन सी योजना है, राशन कार्ड कैसे मिलेगा, या फ़ॉर्म भरने में मदद कीजिए।",
        "ta": "உங்களுக்கு என்ன வேண்டும் என்று உங்கள் சொந்த வார்த்தைகளில் சொல்லுங்கள். உதாரணமாக: எனக்கு எந்தத் திட்டம், ரேஷன் அட்டை எப்படிப் பெறுவது, அல்லது படிவம் நிரப்ப உதவுங்கள்.",
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
                   "what help can i get here", "can you help me", "what can i ask you", "what can i ask", "what services do you provide",
                   "what services are available", "how can you help migrant workers", "what do you offer", "help me", "what all can you do",
                   "what are you able to do", "what you do", "what services do you provide here"],
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
                        "which scheme is for me", "what schemes do you know", "can you find a scheme for me", "can you tell me which scheme i can get",
                        "tell me about schemes", "government schemes", "schemes for migrant workers", "can you help me access a government scheme"],
    "eligibility": ["can you tell me if i am eligible", "am i eligible for any scheme", "can you check whether i am eligible", "check if i am eligible",
                    "can you check my eligibility", "am i eligible", "which schemes am i eligible for", "can you check if i can get a scheme",
                    "tell me what i am eligible for"],
    "help": ["help", "i need help", "please help", "please help me", "what should i do", "how can i get help",
             "how will you help me", "i do not know what to ask", "what do i ask"],
}

# ── native phrasings: how people actually say it in Hindi and Tamil, including the English
# words the recognisers write in Devanagari / Tamil script. Matched word by word with fuzzy
# word similarity, tolerating filler words ("अरे", "जी", "सर", "அண்ணா") and small recognition
# errors; see _native_score().
NATIVE_PHRASES: Dict[str, Dict[str, List[str]]] = {
    "identity": {
        "hi": ["आप कौन हैं", "आप कौन हो", "तुम कौन हो", "तुम कौन हैं", "आप क्या हैं", "यह क्या है", "ये क्या है", "यह मशीन क्या है", "ये मशीन क्या है",
               "यह कौन सी मशीन है", "नोमैडराइट क्या है", "नोमेडराइट क्या है", "यह डिवाइस क्या है", "आपका नाम क्या है", "अपने बारे में बताइए",
               "अपने बारे में बताओ", "यह किस लिए है", "ये किस काम की है", "यह कियोस्क क्या है"],
        "ta": ["நீங்கள் யார்", "நீ யார்", "நீங்க யாரு", "நீ யாரு", "இது என்ன", "இது என்ன இயந்திரம்", "இந்த இயந்திரம் என்ன", "இந்த மெஷின் என்ன",
               "நோமட்ரைட் என்றால் என்ன", "நோமட்ரைட் என்ன", "உங்கள் பெயர் என்ன", "உன் பெயர் என்ன", "உங்களைப் பற்றி சொல்லுங்கள்", "இது எதற்கு",
               "இது எதுக்கு", "இந்த கருவி என்ன"],
    },
    "capability": {
        "hi": ["आप क्या कर सकते हैं", "आप क्या कर सकते हो", "क्या कर सकते हो", "क्या कर सकते हैं", "तुम क्या कर सकते हो", "आप क्या करते हैं",
               "आप क्या करते हो", "मेरी मदद कैसे करोगे", "मेरी मदद कैसे करेंगे", "आप मेरी क्या मदद कर सकते हैं", "आप मेरी मदद कैसे कर सकते हैं",
               "आप किस तरह मदद करते हैं", "यहाँ क्या मदद मिलती है", "यहां क्या मदद मिलेगी", "आप कौन सी सेवाएं देते हैं", "मैं आपसे क्या पूछ सकता हूँ",
               "मैं क्या पूछ सकता हूं", "आप क्या क्या कर सकते हैं"],
        "ta": ["நீங்கள் என்ன செய்ய முடியும்", "நீங்க என்ன செய்வீங்க", "நீ என்ன செய்வாய்", "என்ன செய்ய முடியும்", "நீங்கள் என்ன செய்கிறீர்கள்",
               "எனக்கு எப்படி உதவ முடியும்", "எனக்கு எப்படி உதவுவீர்கள்", "நீங்கள் எப்படி உதவுவீர்கள்", "நீங்க எப்படி உதவுவீங்க", "என்ன உதவி கிடைக்கும்",
               "இங்கே என்ன உதவி கிடைக்கும்", "நான் உங்களிடம் என்ன கேட்கலாம்", "என்ன கேட்கலாம்", "என்ன சேவைகள் தருகிறீர்கள்", "உங்களால் என்ன செய்ய முடியும்"],
    },
    "usage_how": {
        "hi": ["यह कैसे काम करता है", "ये कैसे काम करता है", "इसे कैसे इस्तेमाल करें", "इसका इस्तेमाल कैसे करूं", "मैं सवाल कैसे पूछूं", "सवाल कैसे पूछें",
               "कैसे बोलना है", "मैं कैसे बोलूं", "इसको कैसे चलाएं", "यह कैसे चलता है"],
        "ta": ["இது எப்படி வேலை செய்கிறது", "இது எப்படி வேலை செய்யும்", "இதை எப்படி பயன்படுத்துவது", "எப்படி பயன்படுத்துவது", "நான் எப்படி கேள்வி கேட்பது",
               "கேள்வி எப்படி கேட்பது", "எப்படி பேசுவது", "நான் எப்படி பேசணும்", "இதை எப்படி இயக்குவது"],
    },
    "usage_language": {
        "hi": ["क्या मैं तमिल में बोल सकता हूँ", "क्या मैं हिंदी में बोल सकता हूँ", "क्या मैं अंग्रेजी में बोल सकता हूँ", "तमिल में बोल सकते हैं",
               "हिंदी में बोल सकते हैं", "अंग्रेजी में बोल सकते हैं", "आप कौन सी भाषा समझते हैं", "कौन कौन सी भाषा", "क्या आप तमिल समझते हैं",
               "क्या आप हिंदी समझते हैं", "क्या आप इंग्लिश समझते हैं", "इंग्लिश में बोल सकता हूं"],
        "ta": ["நான் தமிழில் பேசலாமா", "நான் இந்தியில் பேசலாமா", "நான் ஆங்கிலத்தில் பேசலாமா", "தமிழில் பேசலாமா", "இந்தியில் பேசலாமா",
               "ஆங்கிலத்தில் பேசலாமா", "உங்களுக்கு தமிழ் தெரியுமா", "உங்களுக்கு இந்தி தெரியுமா", "உங்களுக்கு ஆங்கிலம் தெரியுமா", "என்ன மொழிகள் தெரியும்",
               "எந்த மொழியில் பேசலாம்", "இங்கிலீஷ்ல பேசலாமா"],
    },
    "usage_form": {
        "hi": ["क्या आप फॉर्म भरने में मदद कर सकते हैं", "फॉर्म भरने में मदद कीजिए", "फ़ॉर्म भरने में मदद करो", "मुझे फॉर्म भरना है", "फॉर्म कैसे भरें",
               "क्या आप फॉर्म भर सकते हैं", "मेरा फॉर्म भर दो", "सरकारी फॉर्म भरना है", "फॉर्म भरवा दीजिए"],
        "ta": ["படிவம் நிரப்ப உதவ முடியுமா", "படிவம் நிரப்ப உதவுங்கள்", "எனக்கு படிவம் நிரப்ப வேண்டும்", "படிவம் எப்படி நிரப்புவது", "ஃபார்ம் நிரப்ப உதவுங்கள்",
               "ஃபார்ம் ஃபில் பண்ணணும்", "அரசு படிவம் நிரப்ப வேண்டும்", "என் படிவத்தை நிரப்பித் தர முடியுமா"],
    },
    "schemes_general": {
        "hi": ["सरकारी योजनाओं के बारे में बताइए", "सरकारी योजनाओं के बारे में बताओ", "कौन सी योजनाएं हैं", "कौन कौन सी योजना है", "मेरे लिए कौन सी योजना है",
               "मुझे कौन सी योजना मिल सकती है", "मेरे लिए योजना ढूंढ दीजिए", "योजना ढूंढने में मदद करो", "सरकारी योजना कैसे मिलेगी", "कौन सी सरकारी योजना मिलेगी"],
        "ta": ["அரசு திட்டங்கள் பற்றி சொல்லுங்கள்", "அரசுத் திட்டங்கள் பற்றி சொல்லுங்க", "என்னென்ன திட்டங்கள் உள்ளன", "எனக்கு எந்த திட்டம் கிடைக்கும்",
               "எனக்கான திட்டம் எது", "எனக்கு ஒரு திட்டம் கண்டுபிடித்துத் தாருங்கள்", "அரசு திட்டம் எப்படி பெறுவது", "எந்த அரசு திட்டம் கிடைக்கும்"],
    },
    "eligibility": {
        "hi": ["क्या मैं पात्र हूँ", "क्या मैं पात्र हूं", "मेरी पात्रता जांच दीजिए", "मैं किस योजना के लिए पात्र हूँ", "देखिए मैं पात्र हूं या नहीं",
               "मुझे मिलेगा या नहीं", "क्या मुझे योजना मिलेगी", "मेरी योग्यता चेक करो"],
        "ta": ["நான் தகுதியானவரா", "எனக்கு தகுதி இருக்கிறதா", "என் தகுதியை சரிபார்க்க முடியுமா", "நான் எந்த திட்டத்திற்கு தகுதியானவர்",
               "எனக்கு கிடைக்குமா இல்லையா பாருங்கள்", "எனக்கு தகுதி உண்டா"],
    },
    "help": {
        "hi": ["मदद", "मदद चाहिए", "मुझे मदद चाहिए", "मेरी मदद करो", "मेरी मदद कीजिए", "कृपया मदद करें", "मुझे कैसे मदद मिलेगी", "मदद कैसे मिलेगी",
               "मुझे क्या करना चाहिए", "मैं क्या करूं", "हेल्प चाहिए", "हेल्प"],
        "ta": ["உதவி", "உதவி வேண்டும்", "எனக்கு உதவி வேண்டும்", "எனக்கு உதவுங்கள்", "தயவுசெய்து உதவுங்கள்", "எனக்கு எப்படி உதவி கிடைக்கும்",
               "நான் என்ன செய்ய வேண்டும்", "நான் என்ன செய்வது", "ஹெல்ப் வேண்டும்", "ஹெல்ப்"],
    },
}

# words that carry no meaning for intent matching (fillers, politeness, address terms)
FILLERS: Dict[str, Set[str]] = {
    "hi": set("अरे जी हाँ हां हा ओ ओह भाई भैया दीदी सर मैडम साहब बहन जी जरा ज़रा ना तो भी कृपया प्लीज अच्छा ठीक है हम्म मुख्य मैं आप".split()),
    "ta": set("அண்ணா அக்கா சார் மேடம் ஐயா அம்மா தம்பி தங்கச்சி ஆமா ஆம் சரி ஓகே கொஞ்சம் தயவுசெய்து ப்ளீஸ் ஹ்ம்ம் இங்க இங்கே".split()),
    "en": set("oh hey please sir madam ma'am brother sister ok okay so um uh well yes ya yeah hmm and main just".split()),
}

# native greeting / thanks words (Devanagari, Tamil, romanised) - the utterance may add filler words
NATIVE_WORDS: Dict[str, Dict[str, List[str]]] = {
    "greeting": {
        "hi": ["नमस्ते", "नमस्कार", "प्रणाम", "हेलो", "हैलो", "हाय", "सुप्रभात", "शुभ प्रभात", "शुभ संध्या", "नमस्ते जी", "राम राम", "सत श्री अकाल", "आदाब",
               "गुड मॉर्निंग", "गुड मोर्निंग", "गुड आफ्टरनून", "गुड इवनिंग", "वेलकम", "जय हिंद", "नमस्ते सर"],
        "ta": ["வணக்கம்", "ஹலோ", "ஹாய்", "காலை வணக்கம்", "மாலை வணக்கம்", "வணக்கம் சார்", "வணக்கம்ங்க", "வணக்கம் ஐயா", "குட் மார்னிங்",
               "குட் ஆஃப்டர்நூன்", "குட் ஈவினிங்", "வெல்கம்", "நமஸ்காரம்", "நமஸ்தே"],
        "en": ["namaste", "namaskar", "vanakkam", "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "welcome", "greetings"],
    },
    "thanks": {
        "hi": ["धन्यवाद", "शुक्रिया", "थैंक यू", "थैंक्स", "बहुत धन्यवाद", "धन्यवाद जी", "बहुत शुक्रिया", "थैंक यू सो मच", "आभार"],
        "ta": ["நன்றி", "மிக்க நன்றி", "ரொம்ப நன்றி", "தேங்க்ஸ்", "நன்றிங்க", "தேங்க் யூ", "ரொம்ப தேங்க்ஸ்"],
        "en": ["thank you", "thanks", "dhanyavad", "nandri", "shukriya", "thank you very much"],
    },
}

# words that mean the sentence is really about a scheme or welfare topic - never a kiosk intent
DEFAULT_SCHEME_TERMS: Set[str] = {
    "ration", "pds", "onorc", "ayushman", "pmjay", "shram", "eshram", "e-shram", "bocw", "kisan", "pension", "loan", "insurance", "housing",
    "awas", "ujjwala", "gas", "mgnrega", "nrega", "job card", "labour card", "labor card", "jan dhan", "bank", "money", "rupees", "subsidy",
    "scholarship", "widow", "disability", "maternity", "hospital", "treatment", "wages", "wage", "salary", "employer", "accident", "compensation",
    "document", "documents", "aadhaar", "aadhar", "apply", "application", "benefit", "helpline", "complain", "complaint", "card",
    # a person describing themselves is asking about schemes, not about the kiosk
    "worker", "working", "farmer", "labour", "labor", "job", "family", "wife", "husband", "daughter", "son", "children", "years old", "village",
    "district", "state", "migrant", "bihar", "tamil nadu", "uttar pradesh", "jharkhand", "odisha", "bengal", "chennai", "mumbai", "delhi",
}

_INTENT_ORDER = ("greeting", "thanks", "usage_language", "usage_form", "usage_how", "eligibility", "identity", "capability", "help", "schemes_general")

# question words: a greeting word inside a real question is not a greeting ("नमस्ते, राशन कैसे मिलेगा")
_QUESTION_WORDS: Dict[str, Set[str]] = {
    "hi": set("क्या कैसे कब कहाँ कहां कौन क्यों कितना कितने कितनी किस किसे कौनसी कौन सी".split()),
    "ta": set("என்ன எப்படி எப்போது எங்கே யார் ஏன் எவ்வளவு எத்தனை எந்த எது".split()),
    "en": set("what how when where who why which can could is are do does will".split()),
}
# generic words for "scheme" / "government": allowed in general questions about schemes
_GENERIC_SCHEME_WORDS = ("योजना", "योजनाओं", "योजनाएं", "योजनाएँ", "सरकारी", "திட்டம்", "திட்டங்கள்", "திட்டங்கள", "அரசு", "அரசுத்")
_GENERIC_OK_INTENTS = ("schemes_general", "eligibility", "help", "capability", "usage_form")
# welfare words in the person's own script: the sentence belongs to the scheme pipeline
_NATIVE_SCHEME_WORDS = ("कार्ड", "राशन", "पेंशन", "बीमा", "किसान", "आयुष्मान", "श्रम", "आवास", "गैस", "उज्ज्वला", "मनरेगा", "जॉब", "लोन",
                        "बैंक", "खाता", "आधार", "दस्तावेज", "दस्तावेज़", "इलाज", "अस्पताल", "मजदूरी", "मज़दूरी", "पैसा", "पैसे", "छात्रवृत्ति", "सब्सिडी",
                        "ரேஷன்", "அட்டை", "ஓய்வூதியம்", "காப்பீடு", "கிசான்", "ஆயுஷ்மான்", "வீடு", "எரிவாயு", "கடன்", "வங்கி", "கணக்கு",
                        "ஆதார்", "ஆவணம்", "ஆவணங்கள்", "சிகிச்சை", "மருத்துவமனை", "கூலி", "சம்பளம்", "பணம்", "உதவித்தொகை", "மானியம்")


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


def _word_sim(a: str, b: str, lang: str) -> float:
    """Similarity of two words, tolerant of recognition errors: equal 1.0; same consonant
    skeleton (Hindi/Tamil, vowel signs ignored) 0.92; otherwise the spelling ratio for
    words of 3+ letters."""
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if lang in ("hi", "ta"):
        try:
            from pocketinfer.applications.nomad_right.native_correct import skeleton
            ka, kb = skeleton(a, lang), skeleton(b, lang)
            if ka and ka == kb and len(ka) >= 2:
                return 0.92
        except Exception:
            pass
    if min(len(a), len(b)) < 3:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _native_score(utt: List[str], phrase: List[str], lang: str) -> float:
    """How well a phrasing is contained in the utterance: the mean best similarity of the
    phrasing's words, minus a small penalty per unmatched utterance word that is not a
    filler (so "मुझे फॉर्म भरना है और राशन कार्ड भी" does not match a short phrasing)."""
    if not utt or not phrase:
        return 0.0
    fill = FILLERS.get(lang, set())
    sims = []
    used = set()
    for pw in phrase:
        best, best_i = 0.0, -1
        for i, uw in enumerate(utt):
            s = _word_sim(pw, uw, lang)
            if s > best:
                best, best_i = s, i
        sims.append(best)
        if best >= 0.8:
            used.add(best_i)
    extra = sum(1 for i, uw in enumerate(utt) if i not in used and uw not in fill)
    return sum(sims) / len(sims) - 0.08 * extra


def match(native_text: str, english_text: str, lang: str, scheme_terms: Optional[Iterable[str]] = None) -> Optional[KioskHit]:
    lang = (lang or "en").lower()[:2]
    if lang not in ("hi", "ta", "en"):
        lang = "en"
    native = _norm(native_text)
    en = _norm(english_text)
    terms = set(scheme_terms or ()) | DEFAULT_SCHEME_TERMS

    nat_words = native.split()
    scheme_native = any(w in native for w in _NATIVE_SCHEME_WORDS)
    generic_native = any(w in nat_words for w in _GENERIC_SCHEME_WORDS)
    scheme_en = bool(en) and _has_scheme_term(en, terms)
    scheme_typed = bool(native) and _has_scheme_term(native, terms)      # typed / English-language utterance

    # 1. greeting / thanks in the person's own words: exact, near-exact, or a greeting word with
    #    up to three filler or unknown words around it ("मुख्य नमस्ते", "வணக்கம் அண்ணா", "hello madam"),
    #    as long as the utterance is not a question and names no scheme
    if native and not (scheme_native or scheme_typed or generic_native):
        qwords = _QUESTION_WORDS.get(lang, set()) | _QUESTION_WORDS["en"]
        for intent in ("greeting", "thanks"):
            for l in (lang, "en"):
                for w in NATIVE_WORDS[intent].get(l, ()):
                    wn = _norm(w)
                    if native == wn or difflib.SequenceMatcher(None, native, wn).ratio() >= 0.86:
                        return KioskHit(intent, 1.0, RESPONSES[intent][lang], RESPONSES[intent]["en"])
                    ww = wn.split()
                    if len(nat_words) <= len(ww) + 3 and not any(q in nat_words for q in qwords):
                        sc = _native_score(nat_words, ww, l) + 0.08 * max(0, len(nat_words) - len(ww))    # extra words allowed here
                        if sc >= 0.9:
                            return KioskHit(intent, round(min(sc, 1.0), 3), RESPONSES[intent][lang], RESPONSES[intent]["en"])

    if scheme_native or scheme_en or scheme_typed:
        # a scheme question that happens to contain "help me" stays with the pipeline
        return None

    # 2. the other intents in the person's own words (Hindi / Tamil phrasings)
    best_intent, best_score = None, 0.0
    if lang in ("hi", "ta") and nat_words and len(nat_words) <= 12:
        for intent in _INTENT_ORDER:
            if generic_native and intent not in _GENERIC_OK_INTENTS:
                continue
            for ph in NATIVE_PHRASES.get(intent, {}).get(lang, ()):
                sc = _native_score(nat_words, _norm(ph).split(), lang)
                if sc > best_score:
                    best_intent, best_score = intent, sc
        if best_intent and best_score >= 0.86:
            return KioskHit(best_intent, round(best_score, 3), RESPONSES[best_intent][lang], RESPONSES[best_intent]["en"])

    if not en:
        return None
    if len(en.split()) > 12:
        return None

    generic_en = bool(re.search(r"\b(scheme|schemes|yojana|government|plan|plans|program|programme)\b", en))
    best_intent, best_score = None, 0.0
    for intent in _INTENT_ORDER:
        if (generic_en or generic_native) and intent not in _GENERIC_OK_INTENTS:
            continue
        s = _best_phrase(en, PHRASES[intent])
        if s > best_score:
            best_intent, best_score = intent, s
    threshold = 0.9 if best_intent in ("greeting", "thanks", "help") else 0.82
    if best_intent and best_score >= threshold:
        return KioskHit(best_intent, round(best_score, 3), RESPONSES[best_intent][lang], RESPONSES[best_intent]["en"])
    return None
