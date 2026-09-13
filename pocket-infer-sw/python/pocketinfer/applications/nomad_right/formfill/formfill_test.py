"""Unit tests for the assisted form-filling pieces that need no hardware: parsers,
the session state machine (order, retries, corrections, review, payload, cleanup,
no personal data in logs) and the identifier's decision rule on synthetic OCR."""
import io
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from pocketinfer.applications.nomad_right.formfill import validators as V
from pocketinfer.applications.nomad_right.formfill.catalog import FormCatalog
from pocketinfer.applications.nomad_right.formfill.identifier import FormIdentifier
from pocketinfer.applications.nomad_right.formfill.ocr import find_ifsc, OcrLine, OcrResult, verhoeff_check_digit
AADHAAR = "23456789012" + verhoeff_check_digit("23456789012")   # a checksum-valid test number
from pocketinfer.applications.nomad_right.formfill.session import FormSession

CAT = FormCatalog()
W = CAT.words


class TestParsers(unittest.TestCase):
    def test_integers_in_three_languages(self):
        self.assertEqual(V.parse_integer("42", "hi").value, 42)
        self.assertEqual(V.parse_integer("बयालीस", "hi").value, 42)
        self.assertEqual(V.parse_integer("मेरी उम्र पैंतीस साल है", "hi").value, 35)
        self.assertEqual(V.parse_integer("நாற்பத்தி இரண்டு", "ta").value, 42)
        self.assertEqual(V.parse_integer("ஐம்பது", "ta").value, 50)
        self.assertEqual(V.parse_integer("twenty five thousand", "en").value, 25000)
        self.assertEqual(V.parse_integer("पाँच हज़ार", "hi").value, 5000)
        self.assertEqual(V.parse_integer("இரண்டாயிரத்து ஐநூறு", "ta").value, 2500)
        with self.assertRaises(V.Invalid):
            V.parse_integer("पता नहीं", "hi")
        with self.assertRaises(V.Invalid):
            V.parse_integer("150", "hi", 18, 110)

    def test_digit_sequences(self):
        self.assertEqual(V.digits_from_speech("nine eight seven six five four three two one zero", "en"), "9876543210")
        self.assertEqual(V.digits_from_speech("नौ आठ सात छह पाँच चार तीन दो एक शून्य", "hi"), "9876543210")
        self.assertEqual(V.digits_from_speech("ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று பூஜ்யம்", "ta"), "9876543210")
        self.assertEqual(V.digits_from_speech("98765 43210", "hi"), "9876543210")
        self.assertEqual(V.digits_from_speech("double nine eight", "en"), "998")
        self.assertEqual(V.parse_phone("मेरा नंबर 98765 43210 है", "hi").value, "9876543210")
        with self.assertRaises(V.Invalid):
            V.parse_phone("12345", "hi")

    def test_aadhaar_checksum(self):
        self.assertEqual(V.parse_aadhaar(" ".join(AADHAAR[i:i + 4] for i in range(0, 12, 4)), "hi").value, AADHAAR)
        bad = AADHAAR[:-1] + str((int(AADHAAR[-1]) + 1) % 10)
        with self.assertRaises(V.Invalid):
            V.parse_aadhaar(bad, "hi")                                                   # one digit off

    def test_dates(self):
        self.assertEqual(V.parse_date("15/08/1985", "hi").value, "1985-08-15")
        self.assertEqual(V.parse_date("15 अगस्त 1985", "hi").value, "1985-08-15")
        self.assertEqual(V.parse_date("पंद्रह अगस्त उन्नीस सौ पचासी", "hi").value, "1985-08-15")
        self.assertEqual(V.parse_date("15 ஆகஸ்ட் 1985", "ta").value, "1985-08-15")
        self.assertEqual(V.parse_date("3 1 2001", "ta").value, "2001-01-03")
        with self.assertRaises(V.Invalid):
            V.parse_date("अगस्त", "hi")

    def test_choices_yesno_commands(self):
        gender = next(f for f in CAT.form("FORM_PMSVANIDHI_LAF")["fields"] if f["key"] == "gender")
        self.assertEqual(V.parse_choice("महिला", "hi", gender["options"]).value, "female")
        self.assertEqual(V.parse_choice("நான் ஆண்", "ta", gender["options"]).value, "male")
        amt = next(f for f in CAT.form("FORM_APY_REGISTRATION")["fields"] if f["key"] == "apy_pension_amount")
        self.assertEqual(V.parse_choice("दो हज़ार", "hi", amt["options"]).value, "2000")
        self.assertEqual(V.parse_choice("5000", "ta", amt["options"]).value, "5000")
        self.assertTrue(V.parse_yesno("जी हाँ", "hi", W)); self.assertFalse(V.parse_yesno("जी नहीं", "hi", W))
        self.assertTrue(V.parse_yesno("ஆமாம்", "ta", W)); self.assertFalse(V.parse_yesno("இல்லை", "ta", W))
        self.assertIsNone(V.parse_yesno("शायद", "hi", W))
        self.assertEqual(V.detect_command("फिर से बोलो", "hi", W), "repeat")
        self.assertEqual(V.detect_command("மீண்டும்", "ta", W), "repeat")
        self.assertEqual(V.detect_command("छोड़ो", "hi", W), "skip")
        self.assertEqual(V.detect_command("ரத்து", "ta", W), "cancel")
        self.assertIsNone(V.detect_command("रवि कुमार", "hi", W))

    def test_names_strip_fillers(self):
        self.assertEqual(V.parse_text("मेरा नाम रवि कुमार है", "hi", name=True).value, "रवि कुमार")
        self.assertEqual(V.parse_text("என் பெயர் ரவி குமார்", "ta", name=True).value, "ரவி குமார்")
        self.assertEqual(V.parse_text("My name is Ravi Kumar.", "en", name=True).value, "Ravi Kumar")


def ocr_of(text: str, lang: str = "en") -> OcrResult:
    return OcrResult(lang=lang, lines=[OcrLine(l, 85.0, i * 20, 0, 18) for i, l in enumerate(text.splitlines()) if l.strip()])


class TestIdentifier(unittest.TestCase):
    def setUp(self):
        self.ident = FormIdentifier(CAT)

    def test_accepts_clean_title_with_keywords(self):
        r = self.ident.identify({"en": ocr_of("PRADHAN MANTRI SURAKSHA BIMA YOJANA\nNAME OF INSURER NAME OF BANK\nCONSENT-CUM-DECLARATION FORM\naccidental insurance cover Rs. 20")})
        self.assertEqual((r.status, r.best), ("accept", "FORM_PMSBY_CONSENT"))

    def test_lookalike_needs_discriminator(self):
        r = self.ident.identify({"en": ocr_of("PRADHAN MANTRI JEEVAN JYOTI BIMA YOJANA\nCONSENT-CUM-DECLARATION FORM\nlife insurance cover Rs. 436")})
        self.assertEqual((r.status, r.best), ("accept", "FORM_PMJJBY_CONSENT"))
        # only the shared part of the title: must not be accepted outright
        r = self.ident.identify({"en": ocr_of("PRADHAN MANTRI BIMA YOJANA\nCONSENT-CUM-DECLARATION FORM\nNAME OF INSURER")})
        self.assertNotEqual(r.status, "accept")

    def test_hindi_title(self):
        r = self.ident.identify({"en": ocr_of("wert aren dar ats"), "hi": ocr_of("प्रधानमंत्री सुरक्षा बीमा योजना\nबीमाकर्ता का नाम", "hi")})
        self.assertEqual(r.best, "FORM_PMSBY_CONSENT")
        self.assertIn(r.status, ("accept", "confirm"))

    def test_partial_title_confirms_not_guesses(self):
        r = self.ident.identify({"en": ocr_of("Atal Pension\nsubscriber form bank branch")})
        self.assertIn(r.status, ("confirm", "accept"))
        self.assertEqual(r.best, "FORM_APY_REGISTRATION")

    def test_unsupported_and_unreadable(self):
        r = self.ident.identify({"en": ocr_of("Electricity bill for the month of March\nUnits consumed 240 amount payable 1,230 due date 15 April\ncustomer number 88221")})
        self.assertEqual(r.status, "unsupported")
        r = self.ident.identify({"en": ocr_of("ae\nfs")})
        self.assertEqual(r.status, "not_readable")

    def test_prior_breaks_a_tie_but_never_invents(self):
        r = self.ident.identify({"en": ocr_of("Electricity bill\nunits amount due date")}, prior_scheme_id="SCH_PMSBY")
        self.assertNotEqual(r.status, "accept")


class TestEnglishDates(unittest.TestCase):
    def test_spoken_and_written_english_dates(self):
        from pocketinfer.applications.nomad_right.formfill.validators import parse_date
        for text, iso in (("fifteen August nineteen eighty five", "1985-08-15"), ("15th August 1985", "1985-08-15"),
                          ("fifteenth of August nineteen eighty five", "1985-08-15"), ("twenty first March twenty twenty", "2020-03-21"),
                          ("2 3 1990", "1990-03-02"), ("2nd March 1990", "1990-03-02"), ("two thousand five June ten", "2005-06-10")):
            self.assertEqual(parse_date(text, "en").value, iso, text)


class TestIfscConfusion(unittest.TestCase):
    def test_fifth_char_zero_confusions(self):
        for raw in ("SBIN0001234", "SBINQ001234", "SBINO001234", "SBIND001234", "SBIN 0001234"):
            self.assertEqual(find_ifsc(f"Branch Patna IFSC: {raw} MICR 800002"), "SBIN0001234", raw)
        self.assertEqual(find_ifsc("IFSC CODE: BARB0TRIPUR"), "BARB0TRIPUR")      # alphabetic branch code
        self.assertEqual(find_ifsc("IFSC SBIN 0001234"), "SBIN0001234"); self.assertEqual(find_ifsc("code SBIN0 001234"), "SBIN0001234")
        for t in ("no code here", "ACCOUNT 32124567890", "STATE BANK OF INDIA branch Patna Main", "BANK OF INDIA"):
            self.assertIsNone(find_ifsc(t), t)


class TestSession(unittest.TestCase):
    def test_pmsby_hindi_flow_order_and_payload(self):
        log = io.StringIO(); h = logging.StreamHandler(log); logging.getLogger().addHandler(h); logging.getLogger().setLevel(logging.DEBUG)
        s = FormSession(CAT, "FORM_PMSBY_CONSENT", "hi")
        step = s.start()
        self.assertEqual(s.state, "CONFIRM_FORM"); self.assertIn("सुरक्षा", step.speak)
        step = s.handle_answer("हाँ"); self.assertEqual(s.state, "ASKING"); self.assertIn("पूरा नाम", step.speak)
        answers = ["मेरा नाम रवि कुमार है", "श्याम कुमार", "मकान 12 गाँधी नगर", "पटना", "पटना", "बिहार", "800001", "98765 43210"]
        for a in answers:
            step = s.handle_answer(a)
            if s.state == "CONFIRMING":
                step = s.handle_answer("हाँ")
        self.assertEqual(s.values["full_name"], "रवि कुमार"); self.assertEqual(s.values["pincode"], "800001"); self.assertEqual(s.values["mobile"], "9876543210")
        # document field: passbook OCR text -> account number, confirmed
        self.assertEqual(s.state, "DOC_WAIT"); self.assertEqual(step.want_document, "bank_passbook")
        step = s.handle_document("STATE BANK OF INDIA  Account No: 3212 4567 890  IFSC SBIN0001234 branch Patna")
        self.assertEqual(s.state, "CONFIRMING"); step = s.handle_answer("हाँ")
        self.assertEqual(s.values["account_number"], "32124567890")
        # ifsc (optional, doc): unreadable -> left for the officer, no voice fallback (nobody dictates an IFSC)
        self.assertEqual(step.want_document, "bank_passbook")
        step = s.handle_document("no code here"); self.assertFalse(s.doc_failed)
        self.assertIn("ifsc", s.skipped); self.assertIn("अधिकारी", step.speak); self.assertIsNone(step.want_document)
        step = s.handle_answer("आधार")          # kyc document choice
        self.assertEqual(s.values["kyc_document"], "aadhaar")
        # aadhaar: OCR reads a checksum-valid number
        step = s.handle_document(f"Government of India  {AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}  Ravi Kumar DOB 15/08/1985"); step = s.handle_answer("हाँ")
        self.assertEqual(s.values["aadhaar"], AADHAAR)
        step = s.handle_answer("15 अगस्त 1985"); step = s.handle_answer("हाँ"); self.assertEqual(s.values["dob"], "1985-08-15")
        step = s.handle_answer("छोड़ो")          # email optional
        step = s.handle_answer("नहीं")           # disability
        self.assertEqual(s.values["disability"], "no")
        self.assertNotIn("disability_details", [r.key for r in s.plan if s._applicable(r)])   # conditional field hidden
        step = s.handle_answer("सीता देवी"); step = s.handle_answer("पत्नी"); step = s.handle_answer("2 3 1990"); step = s.handle_answer("हाँ")
        step = s.handle_answer("छोड़ो")          # nominee address optional
        self.assertEqual(s.state, "REVIEW"); self.assertIn("रवि कुमार", step.speak)
        step = s.handle_answer("हाँ"); self.assertTrue(step.done); self.assertEqual(s.state, "DONE")
        p = s.payload()
        self.assertEqual(p["form_id"], "FORM_PMSBY_CONSENT"); self.assertEqual(p["fields"]["nominee_relation"], "wife"); self.assertEqual(p["fields"]["nominee_dob"], "1990-03-02")
        self.assertEqual(list(p["fields"])[:3], ["full_name", "father_or_husband_name", "address"])
        logging.getLogger().removeHandler(h)
        for secret in ("रवि कुमार", "9876543210", AADHAAR, "32124567890"):
            self.assertNotIn(secret, log.getvalue(), "personal data must not reach the logs")
        s.clear(); self.assertEqual(s.values, {})

    def test_invalid_retry_then_officer_fallback(self):
        s = FormSession(CAT, "FORM_APY_REGISTRATION", "ta"); s.start(form_confirmed=True)
        # first field is the passbook; fall back to voice and give junk three times
        s.handle_document("nothing")
        for _ in range(3):
            step = s.handle_answer("தெரியல சொல்ல")
        self.assertIn("account_number", s.unanswered_required)
        self.assertEqual(s._current().key, "bank_name")

    def test_change_goes_back_and_review_edit(self):
        s = FormSession(CAT, "FORM_MGNREGA_REGISTRATION", "hi"); s.start(form_confirmed=True)
        step = s.handle_answer("दो")                       # workers_count -> group of 2
        self.assertEqual(s._current().key, "workers.1.name")
        s.handle_answer("राम"); s.handle_answer("तीस"); s.handle_answer("पुरुष")
        self.assertEqual(s._current().key, "workers.2.name")
        step = s.handle_answer("बदलो")                     # back to previous answered field
        self.assertEqual(s._current().key, "workers.1.gender")
        s.handle_answer("महिला")
        self.assertEqual(s.values["workers.1.gender"], "female")
        p = s.payload(); self.assertEqual(p["fields"]["workers"][0]["gender"], "female")

    def test_cancel_clears(self):
        s = FormSession(CAT, "FORM_PMUY_KYC", "ta"); s.start(form_confirmed=True)
        s.handle_answer("ஆம்"); s.handle_answer("என் பெயர் லட்சுமி")
        self.assertIn("full_name", s.values)
        step = s.handle_answer("ரத்து")
        self.assertTrue(step.done); self.assertEqual(s.values, {}); self.assertEqual(s.state, "CANCELLED")

    def test_wrong_form_answer_ends_session(self):
        s = FormSession(CAT, "FORM_SSA1_ACCOUNT_OPENING", "hi"); s.start()
        step = s.handle_answer("नहीं")
        self.assertTrue(step.done); self.assertEqual(s.state, "CANCELLED")


class TestConfirmationRobustness(unittest.TestCase):
    """The review and value confirmations must never repeat forever, must take touch
    answers (the on-screen Yes/No arrive as the language's first yes/no word) and must
    accept the ways people actually say yes."""

    @staticmethod
    def drive_to_review(s):
        """Answer or skip everything until the read-back (any form, any language)."""
        for _ in range(400):
            if s.state in ("REVIEW", "DONE", "CANCELLED"):
                return
            if s.state == "DOC_WAIT":
                s.handle_document("nothing readable")
            elif s.state == "CONFIRMING":
                s.handle_answer(CAT.words["yes"][s.lang][0])
            else:
                s.handle_answer(CAT.words["skip"][s.lang][0])
        raise AssertionError("did not reach the review")

    def test_review_never_loops_forever(self):
        s = FormSession(CAT, "FORM_PMSBY_CONSENT", "en"); s.start(form_confirmed=True)
        self.drive_to_review(s)
        self.assertEqual(s.state, "REVIEW")
        step = s.handle_answer("you")                      # what English Whisper writes for a near-empty clip
        self.assertEqual(s.state, "REVIEW"); self.assertIn("did not catch", step.speak)
        step = s.handle_answer("")
        self.assertEqual(s.state, "REVIEW")
        step = s.handle_answer("Thank you.")
        self.assertTrue(step.done); self.assertEqual(s.state, "DONE"); self.assertTrue(s.unconfirmed)
        p = s.payload("complete_unconfirmed" if s.unconfirmed else "complete")
        self.assertEqual(p["status"], "complete_unconfirmed")

    def test_review_touch_yes_and_change(self):
        for lang in ("hi", "ta", "en"):
            s = FormSession(CAT, "FORM_PMUY_KYC", lang); s.start(form_confirmed=True)
            self.drive_to_review(s)
            self.assertEqual(s.state, "REVIEW", lang)
            step = s.handle_answer(CAT.words["change"][lang][0])      # on-screen Change
            self.assertEqual(s.state, "CHANGE_WHICH", lang)
            s = FormSession(CAT, "FORM_PMUY_KYC", lang); s.start(form_confirmed=True)
            self.drive_to_review(s)
            step = s.handle_answer(CAT.words["yes"][lang][0])         # on-screen Yes
            self.assertTrue(step.done, lang); self.assertFalse(s.unconfirmed, lang)

    def test_value_confirmation_never_loops_forever(self):
        s = FormSession(CAT, "FORM_PMSBY_CONSENT", "hi"); s.start(form_confirmed=True)
        for a in ["रवि कुमार", "श्याम कुमार", "मकान 12", "पटना", "पटना", "बिहार", "800001", "98765 43210"]:
            s.handle_answer(a)
        self.assertEqual(s.state, "CONFIRMING")                       # the mobile number is read back
        for _ in range(3):
            step = s.handle_answer("अरे बाबा क्या पता")                 # neither yes nor no, three times
        self.assertEqual(s.state, "ASKING"); self.assertEqual(s._current().key, "mobile"); self.assertIn("सुनाई नहीं", step.speak)
        s.handle_answer("98765 43210"); s.handle_answer("हाँ")
        self.assertEqual(s.values["mobile"], "9876543210")

    def test_ways_of_saying_yes_and_no(self):
        for text, lang, want in [("Yep.", "en", True), ("yes please", "en", True), ("Send it", "en", True), ("not right", "en", False), ("nah", "en", False),
                                 ("यस", "hi", True), ("हां जी", "hi", True), ("नो", "hi", False), ("गलत है", "hi", False),
                                 ("எஸ்", "ta", True), ("ஆமாமா", "ta", True), ("நோ", "ta", False), ("you", "en", None), ("", "hi", None)]:
            self.assertEqual(V.parse_yesno(text, lang, CAT.words), want, (text, lang))

    def test_payload_pictures_are_numbered_and_cancel_hands_over(self):
        s = FormSession(CAT, "FORM_PMUY_KYC", "ta"); s.start(form_confirmed=True)
        got = []
        s.on_snapshot = got.append
        s.handle_answer("ஆம்"); s.handle_answer("என் பெயர் லட்சுமி")
        p1 = s.payload("in_progress"); p2 = s.payload("in_progress")
        self.assertEqual((p1["seq"], p2["seq"]), (1, 2)); self.assertEqual(p1["status"], "in_progress"); self.assertIsNone(p1["completed_at"])
        self.assertEqual(p1["answered"], len(p1["fields"])); self.assertGreater(p1["total"], p1["answered"])
        step = s.handle_answer("ரத்து")                                # spoken cancel
        self.assertTrue(step.done); self.assertEqual(s.state, "CANCELLED"); self.assertEqual(s.values, {})
        self.assertEqual(len(got), 1); self.assertEqual(got[0]["status"], "cancelled"); self.assertEqual(got[0]["seq"], 3)
        self.assertIn("full_name", got[0]["fields"]); self.assertIn("சேமிக்கப்பட்டுள்ளன", step.speak)


if __name__ == "__main__":
    unittest.main(verbosity=1)
