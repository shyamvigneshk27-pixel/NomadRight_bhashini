"""Kiosk intents (greetings, thanks, questions about the device) and the native-script corrector."""
import unittest

from pocketinfer.applications.nomad_right import kiosk_intents as K
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


if __name__ == "__main__":
    unittest.main()
