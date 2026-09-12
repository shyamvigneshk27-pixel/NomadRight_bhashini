"""FormFlow with fake device I/O and real OCR: a whole Hindi PMSBY session from a
synthetic photo, a synthetic passbook and Aadhaar copy for the document fields,
Home half-way, a walk-away timeout, and repeated sessions without memory growth."""
import os
import resource
import sys
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from pocketinfer.applications.nomad_right import constants
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.flow import FormFlow, FlowIO, HOME, TIMEOUT
from pocketinfer.applications.nomad_right.formfill.ocr import verhoeff_check_digit

PHOTOS = os.path.expanduser("~/benchmark_data/forms/photos")
AADHAAR = "23456789012" + verhoeff_check_digit("23456789012")


def text_image(lines, width=1400, height=900, size=1.4):
    img = np.full((height, width, 3), 250, np.uint8)
    y = 80
    for line in lines:
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), 3, cv2.LINE_AA)
        y += int(70 * size)
    return img


PASSBOOK = text_image(["STATE BANK OF INDIA", "Branch: Patna Main   IFSC: SBIN0001234", "Account No: 32124567890", "Name: RAVI KUMAR", "CIF 87654321"])
AADHAAR_CARD = text_image(["GOVERNMENT OF INDIA", "Ravi Kumar", "DOB: 15/08/1985  MALE", f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}", "Aadhaar - Aam Aadmi ka Adhikar"])


class FakeIO:
    """Scripted answers; records what was spoken."""

    def __init__(self, answers, documents=None):
        self.answers, self.documents = list(answers), list(documents or [])
        self.spoken, self.ui_states = [], []

    def speak(self, text):
        self.spoken.append(text); return True

    def listen(self, timeout):
        return self.answers.pop(0) if self.answers else TIMEOUT

    def capture(self, timeout):
        return self.documents.pop(0) if self.documents else None

    def ui(self, state):
        self.ui_states.append(state)

    def status(self, text):
        pass

    def io(self):
        return FlowIO(speak=self.speak, listen=self.listen, capture=self.capture, ui=self.ui, status=self.status)


class FakeOutbox:
    def __init__(self):
        self.payloads = []

    def submit(self, payload):
        self.payloads.append(payload); return "sent"


PMSBY_ANSWERS_HI = ["हाँ",                     # confirm the form
                    "मेरा नाम रवि कुमार है", "श्याम कुमार", "मकान 12 गाँधी नगर", "पटना", "पटना", "बिहार", "800001",   # pincode: no confirmation
                    "98765 43210", "हाँ",                                                                          # mobile: confirmed
                    # account number from the passbook -> confirm; ifsc from passbook -> confirm
                    "हाँ", "हाँ",
                    "आधार",                    # kyc document
                    "हाँ",                     # aadhaar read from the card -> confirm
                    "15 अगस्त 1985", "हाँ", "छोड़ो", "नहीं",
                    "सीता देवी", "पत्नी", "2 3 1990", "हाँ", "छोड़ो",
                    "हाँ"]                     # review -> send


class TestFlow(unittest.TestCase):
    def setUp(self):
        self.cat = FormCatalog()
        self.photo = cv2.imread(f"{PHOTOS}/pmsby_hi__photo1.jpg")
        self.assertIsNotNone(self.photo)

    def test_full_hindi_session_from_photo(self):
        io = FakeIO(PMSBY_ANSWERS_HI, documents=[PASSBOOK, PASSBOOK, AADHAAR_CARD])
        ob = FakeOutbox()
        flow = FormFlow(self.cat, io.io(), "hi", outbox=ob)
        t0 = time.time()
        res = flow.run(self.photo, prior_scheme_id="SCH_PMSBY")
        self.assertEqual(res.outcome, "sent", io.spoken[-3:])
        self.assertEqual(res.form_id, "FORM_PMSBY_CONSENT")
        p = ob.payloads[0]
        self.assertEqual(p["fields"]["full_name"], "रवि कुमार")
        self.assertEqual(p["fields"]["account_number"], "32124567890")
        self.assertEqual(p["fields"]["ifsc"], "SBIN0001234")
        self.assertEqual(p["fields"]["aadhaar"], AADHAAR)
        self.assertEqual(p["fields"]["dob"], "1985-08-15")
        self.assertEqual(p["fields"]["nominee_relation"], "wife")
        self.assertEqual(p["unanswered_required"], [])
        # every spoken line is in Hindi script or a number - nothing English leaked from the catalogue
        self.assertTrue(all(any("ऀ" <= ch <= "ॿ" for ch in s) for s in io.spoken), [s for s in io.spoken if not any("ऀ" <= ch <= "ॿ" for ch in s)][:3])
        # the session's values are gone
        self.assertEqual(flow.session.values, {})
        print(f"\n  full session: {len(io.spoken)} prompts, identify {res.timings.get('identify')}s, doc OCR {res.timings.get('doc_ocr')}s, total {time.time() - t0:.1f}s (fake TTS/ASR)")

    def test_home_midway_cancels_and_wipes(self):
        io = FakeIO(["हाँ", "मेरा नाम रवि कुमार है", HOME])
        flow = FormFlow(self.cat, io.io(), "hi", outbox=FakeOutbox())
        res = flow.run(self.photo)
        self.assertEqual(res.outcome, "cancelled")
        self.assertEqual(flow.session.values, {})
        self.assertIn("रद्द", io.spoken[-1])

    def test_walk_away_timeout(self):
        io = FakeIO(["हाँ"])            # nobody answers the first question
        flow = FormFlow(self.cat, io.io(), "hi", outbox=FakeOutbox())
        res = flow.run(self.photo)
        self.assertEqual(res.outcome, "cancelled")

    def test_wrong_form_then_named(self):
        # the person says no to PMSBY and names the Jeevan Jyoti form
        io = FakeIO(["नहीं जीवन ज्योति", "हाँ"] + ["रद्द"])
        flow = FormFlow(self.cat, io.io(), "hi", outbox=FakeOutbox())
        res = flow.run(self.photo)
        self.assertEqual(res.form_id, "FORM_PMJJBY_CONSENT")

    def test_unreadable_photo_asks_which_scheme(self):
        blank = np.full((960, 1280, 3), 240, np.uint8)
        io = FakeIO(["सुरक्षा बीमा", "हाँ", "रद्द"])
        flow = FormFlow(self.cat, io.io(), "hi", outbox=FakeOutbox())
        res = flow.run(blank)
        self.assertEqual(res.form_id, "FORM_PMSBY_CONSENT")
        self.assertTrue(any("पढ़ नहीं पाया" in s for s in io.spoken))

    def test_repeated_sessions_do_not_grow(self):
        rss = []
        for i in range(6):
            io = FakeIO(PMSBY_ANSWERS_HI, documents=[PASSBOOK, PASSBOOK, AADHAAR_CARD])
            FormFlow(self.cat, io.io(), "hi", outbox=FakeOutbox()).run(self.photo, prior_scheme_id="SCH_PMSBY")
            rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        print(f"\n  RSS after sessions 1..6: {rss} MB")
        self.assertLess(rss[-1] - rss[1], 40, "process memory should not keep growing across sessions")


if __name__ == "__main__":
    unittest.main(verbosity=1)
