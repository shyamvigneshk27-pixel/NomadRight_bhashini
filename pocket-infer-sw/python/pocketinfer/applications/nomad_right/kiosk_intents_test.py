"""Kiosk intents (greetings, thanks, questions about the device) and the native-script corrector."""
import unittest

from pocketinfer.applications.nomad_right import kiosk_intents as K
from pocketinfer.applications.nomad_right import native_correct as NC
from pocketinfer.applications.nomad_right import native_correct as N

TERMS = {"ration", "pension", "pds", "wedding", "ayushman"}


class TestKioskIntents(unittest.TestCase):
    def test_english_phrasings(self):
        for text, intent in (("What can you do?", "capability"), ("What do you do", "capability"), ("How can you help me", "capability"),
                             ("What help can I get here", "capability"), ("What is NomadRight", "identity"), ("Who are you", "identity"),
                             ("Tell me about yourself", "identity"), ("Hello", "greeting"), ("Good morning", "greeting"), ("Namaste", "greeting"),
                             ("Thank you", "thanks"), ("How does this work", "usage_how"), ("Can I speak in Tamil", "usage_language"),
                             ("Can you help me fill a form", "usage_form"), ("Can you tell me about government schemes", "schemes_general"),
                             ("Which government scheme can help me", "schemes_general")):
            hit = K.match("", text, "en", TERMS)
            self.assertIsNotNone(hit, text); self.assertEqual(hit.intent, intent, text); self.assertTrue(hit.native)

    def test_hindi_and_tamil_native_words_and_translations(self):
        for lang, native, en, intent in (("hi", "नमस्ते", "Hello", "greeting"), ("hi", "धन्यवाद", "Thank you", "thanks"), ("hi", "प्रणाम", "Pranam", "greeting"),
                                         ("hi", "आप क्या कर सकते हैं", "What can you do", "capability"), ("hi", "आप कौन हैं", "Who are you", "identity"),
                                         ("ta", "வணக்கம்", "Hello", "greeting"), ("ta", "நன்றி", "Thanks", "thanks"),
                                         ("ta", "நீங்கள் என்ன செய்ய முடியும்", "What you can do", "capability")):
            hit = K.match(native, en, lang, TERMS)
            self.assertIsNotNone(hit, native); self.assertEqual(hit.intent, intent, native)
            self.assertEqual(hit.native, K.RESPONSES[intent][lang])

    def test_scheme_questions_are_left_to_the_pipeline(self):
        for native, en in (("", "What documents do I need for a ration card"), ("", "Can you help me get a pension"),
                           ("पीडीएस योजना के लिए मानदंड क्या है", "What is the criterion for a PDS plan?"),
                           ("", "I am a construction worker from Bihar, which scheme can help me"), ("", "hello I need money for my daughter's wedding"),
                           ("", "How much money does Ayushman Bharat give")):
            self.assertIsNone(K.match(native, en, "en", TERMS), en)


class TestNativeCorrection(unittest.TestCase):
    def test_hindi_recognition_errors_snap_to_scheme_words(self):
        fixed, hint = N.correct("इनडिएस योजना के लिए महनदंड क्या है", "hi")
        self.assertEqual(fixed, "पीडीएस योजना के लिए मानदंड क्या है"); self.assertEqual(hint, "SCH_ONORC")
        fixed, hint = N.correct("आयुशमान भारत में इलाज मिलेगा क्या", "hi")
        self.assertEqual(fixed, "आयुष्मान भारत में इलाज मिलेगा क्या"); self.assertEqual(hint, "SCH_PMJAY")

    def test_correct_sentences_are_untouched(self):
        for lang, t in (("hi", "पीडीएस योजना के लिए मानदंड क्या है"), ("hi", "राशन कार्ड के लिए कौन से दस्तावेज़ चाहिए"), ("hi", "मुझे राशन कार्ड कैसे मिलेगा"),
                        ("hi", "मेरा नाम रवि कुमार है"), ("ta", "ரேஷன் கார்டுக்கு என்ன ஆவணங்கள் தேவை"), ("ta", "என் பெயர் ரவி குமார்")):
            self.assertEqual(N.correct(t, lang)[0], t, t)

    def test_tamil(self):
        fixed, hint = N.correct("ஆயுஷ்மன் அட்டை எப்படி பெறுவது", "ta")
        self.assertEqual(fixed, "ஆயுஷ்மான் அட்டை எப்படி பெறுவது"); self.assertEqual(hint, "SCH_PMJAY")
        self.assertEqual(N.correct("பிடிஎஸ் திட்டத்திற்கு என்ன தகுதி", "ta")[1], "SCH_ONORC")

    def test_greetings_are_never_corrected(self):
        for lang, t in (("ta", "வணக்கம்"), ("ta", "நன்றி"), ("hi", "नमस्ते"), ("hi", "धन्यवाद"), ("hi", "नमस्ते जी")):
            self.assertEqual(N.correct(t, lang)[0], t, t)

    def test_english_passes_through(self):
        self.assertEqual(N.correct("What documents do I need", "en"), ("What documents do I need", None))


class TestNativeIntentsV2(unittest.TestCase):
    """2026-09-14: native phrasings for every intent, tolerant greetings, generic scheme words."""

    def hit(self, native, en, lang):
        h = K.match(native, en, lang)
        return h.intent if h else None

    def test_greeting_with_extra_words_from_the_recogniser(self):
        for t in ("मुख्य नमस्ते", "नमस्ते भैया", "अरे नमस्ते", "नमस्ते जी"):
            self.assertEqual(self.hit(t, "", "hi"), "greeting", t)
        for t in ("வணக்கம் அண்ணா", "வணக்கம்ங்க", "குட் மார்னிங்"):
            self.assertEqual(self.hit(t, "", "ta"), "greeting", t)
        self.assertEqual(self.hit("hello madam", "hello madam", "en"), "greeting")

    def test_greeting_inside_a_scheme_question_is_not_a_greeting(self):
        self.assertIsNone(self.hit("नमस्ते राशन कार्ड कैसे बनेगा", "Hello, how will the ration card be made?", "hi"))
        self.assertIsNone(self.hit("வணக்கம் ரேஷன் அட்டை எப்படி பெறுவது", "Hello, how to get a ration card", "ta"))

    def test_native_phrasings_without_translation(self):
        cases = [("hi", "आप कौन हैं", "identity"), ("hi", "यह क्या है", "identity"), ("hi", "क्या कर सकते हो", "capability"),
                 ("hi", "मेरी मदद कैसे करोगे", "capability"), ("hi", "मुझे कैसे मदद मिलेगी", "help"), ("hi", "क्या मैं तमिल में बोल सकता हूँ", "usage_language"),
                 ("ta", "நீங்கள் யார்", "identity"), ("ta", "நீங்க என்ன செய்வீங்க", "capability"), ("ta", "இதை எப்படி பயன்படுத்துவது", "usage_how"),
                 ("ta", "நான் தகுதியானவரா", "eligibility"), ("ta", "எனக்கு எந்த திட்டம் கிடைக்கும்", "schemes_general"), ("ta", "அரசு படிவம் நிரப்ப வேண்டும்", "usage_form")]
        for lang, t, want in cases:
            self.assertEqual(self.hit(t, "", lang), want, t)

    def test_english_eligibility_and_help(self):
        self.assertEqual(self.hit("Can you check whether I am eligible?", "Can you check whether I am eligible?", "en"), "eligibility")
        self.assertEqual(self.hit("I need help", "I need help", "en"), "help")
        self.assertEqual(self.hit("What help can I get?", "What help can I get?", "en"), "capability")

    def test_specific_schemes_and_personal_details_stay_with_the_pipeline(self):
        for lang, t, en in [("en", "Am I eligible for PM Kisan?", "Am I eligible for PM Kisan?"), ("hi", "मुझे पैसों की मदद चाहिए", "I need money help"),
                            ("hi", "मेरी उम्र चालीस साल है", "I am forty years old"), ("ta", "எனக்கு பணம் உதவி வேண்டும்", "I need money help"),
                            ("ta", "என் பெயர் லட்சுமி", "My name is Lakshmi"), ("en", "15 August 1985", "15 August 1985"), ("hi", "हाँ", "yes")]:
            self.assertIsNone(self.hit(t, en, lang), t)


class TestNativeCorrectionV2(unittest.TestCase):
    def test_ordinary_tamil_is_never_changed(self):
        for t in ("நீங்கள் என்ன செய்ய முடியும்", "நீங்க என்ன செய்வீங்க", "இங்கே என்ன உதவி கிடைக்கும்", "அரசு திட்டங்கள் பற்றி சொல்லுங்கள்",
                  "என் தகுதியை சரிபார்க்க முடியுமா", "நான் சென்னையில் வேலை செய்கிறேன்"):
            self.assertEqual(NC.correct(t, "ta")[0], t)

    def test_ordinary_hindi_is_never_changed(self):
        for t in ("मुझे पैसों की मदद चाहिए", "मैं बिहार से आया मजदूर हूँ", "मुख्य नमस्ते", "आप क्या कर सकते हैं"):
            self.assertEqual(NC.correct(t, "hi")[0], t)

    def test_real_recognition_errors_are_fixed(self):
        self.assertEqual(NC.correct("इनडिएस योजना के लिए महनदंड क्या है", "hi")[0], "पीडीएस योजना के लिए मानदंड क्या है")
        self.assertIn("पेंशन", NC.correct("मेरी उम्र साठ साल है क्या फैशन मिलेगी", "hi")[0])
        self.assertEqual(NC.correct("हिल पर लाइय नंबर क्या है", "hi")[0], "हेल्पलाइन नंबर क्या है")
        fixed, hint = NC.correct("आयुशमन कारड के लिए कोनसे दसतावेज चाहिए", "hi")
        self.assertIn("आयुष्मान कार्ड", fixed); self.assertIn("दस्तावेज", fixed); self.assertEqual(hint, "SCH_PMJAY")
        self.assertEqual(NC.correct("என் வேஷன் அட்டையை தமிழ்நாட்டில் பயன்படுத்தலாமா", "ta"), ("என் ரேஷன் அட்டையை தமிழ்நாட்டில் பயன்படுத்தலாமா", "SCH_ONORC"))
        fixed, hint = NC.correct("பிரதம மந்திரிக்கு சான் திட்டம் பற்றி சொல்லுங்கள்", "ta")
        self.assertIn("கிசான்", fixed); self.assertEqual(hint, "SCH_PMKISAN")

    def test_garbled_word_is_never_turned_into_a_generic_title_word(self):
        self.assertNotIn("சட்டம்", NC.correct("நீட்டம் அட்டை பதிவு செய்வது எப்படி", "ta")[0])


if __name__ == "__main__":
    unittest.main()
