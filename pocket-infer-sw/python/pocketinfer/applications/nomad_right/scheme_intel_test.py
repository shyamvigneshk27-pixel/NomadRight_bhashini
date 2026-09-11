#!/usr/bin/env python3
"""
NomadRight scheme-intelligence layer tests (offline: no ASR/NMT/TTS, no
Qwen, no embedder - the structured paths never need them).

Usage (run from repo root):
    python python/pocketinfer/applications/nomad_right/scheme_intel_test.py
    NOMADRIGHT_SI_HEAVY=1 python ...   # also runs the WorkflowController integration test
                                       # (loads e5 + ChromaDB, uses temporary DB copies)
"""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from pocketinfer.applications.nomad_right.scheme_intel import metrics, si_constants  # noqa: E402
from pocketinfer.applications.nomad_right.scheme_intel.composer import ResponseComposer  # noqa: E402
from pocketinfer.applications.nomad_right.scheme_intel.models import Intent, Profile, Route, Status  # noqa: E402
from pocketinfer.applications.nomad_right.scheme_intel.scenario_engine import ScenarioEngine  # noqa: E402

metrics.LATENCY._path = None  # tests never overwrite the running app's latency snapshot

T1 = "What is One Nation One Ration Card?"
T2 = "I am from Bihar and now working in Tamil Nadu. I have a ration card. Can I get my ration here?"
T3 = "I am a construction worker. What government benefits can I get?"
T4 = ("I am 32 years old, from Bihar, working in Chennai as a construction worker. "
      "My monthly income is around 12000. What schemes can help me?")
T5 = "What documents are required for this scheme?"
T6 = "I don't know what government scheme I can get. I am working in Tamil Nadu."
NEGATIVE = ["Yapri festival at Kudab Uri", "Man Tarosheva Tamilmevachilaka Minnewala Paite Kaisa",
            "What is the weather today?", "I need to be frozen", "Updating the app", "Hello", "Thank you"]


def _si():
    from pocketinfer.applications.nomad_right.scheme_intel.service import SchemeIntelligence
    si = SchemeIntelligence.create(embedder_provider=None, qwen_client=None)
    if si is None:
        raise unittest.SkipTest("scheme_intel knowledge base not built (run build_kb)")
    return si


class TestScenario(unittest.TestCase):
    def setUp(self):
        self.se = ScenarioEngine()

    def test_origin_and_current_state_are_separate(self):
        p = self.se.extract("I am from Bihar and working in Chennai")
        self.assertEqual(p.get("origin_state"), "Bihar")
        self.assertEqual(p.get("current_city"), "Chennai")
        self.assertEqual(p.get("current_state"), "Tamil Nadu")
        self.assertEqual(p.get("migration_status"), "INTER_STATE_MIGRANT")

    def test_nothing_is_assumed(self):
        p = self.se.extract("I am from Bihar")
        self.assertEqual(p.get("origin_state"), "Bihar")
        for field in ("current_state", "migration_status", "age", "occupation", "ration_card", "citizenship"):
            self.assertIsNone(p.get(field), field)

    def test_reference_query_4_facts(self):
        p = self.se.extract(T4)
        self.assertEqual(p.get("age"), 32)
        self.assertEqual(p.get("occupation"), "CONSTRUCTION_WORKER")
        self.assertEqual(p.get("monthly_income"), 12000)
        self.assertEqual(p.get("origin_state"), "Bihar")
        self.assertEqual(p.get("current_state"), "Tamil Nadu")

    def test_negations_are_respected(self):
        p = self.se.extract("I don't have a ration card and my Aadhaar is not linked")
        self.assertIs(p.get("ration_card"), False)
        self.assertIs(p.get("aadhaar_seeded_ration"), False)
        self.assertIs(self.se.extract("I don't pay income tax").get("income_tax_payer"), False)

    def test_other_peoples_facts_are_not_the_users(self):
        p = self.se.extract("My daughter is 6 years old. My husband is a driver.")
        self.assertIsNone(p.get("age"))
        self.assertIsNone(p.get("occupation"))
        self.assertEqual(p.get("girl_child_ages"), [6])

    def test_card_state_versus_where_it_is_used(self):
        p = self.se.extract("Can I use my ration card in Tamil Nadu?")
        self.assertEqual(p.get("current_state"), "Tamil Nadu")
        self.assertIsNone(p.get("ration_card_state"))
        p = self.se.extract("I have a ration card from Bihar")
        self.assertEqual(p.get("ration_card_state"), "Bihar")
        self.assertIsNone(p.get("origin_state"))

    def test_scheme_names_are_not_occupations(self):
        self.assertIsNone(self.se.extract("Give me details about pm kisan").get("occupation"))
        self.assertEqual(self.se.extract("As a construction worker, what can I get?").get("occupation"),
                         "CONSTRUCTION_WORKER")
        self.assertEqual(self.se.extract("I am a kisan from Bihar").get("occupation"), "FARMER")

    def test_spoken_numbers(self):
        self.assertEqual(self.se.extract("I earn twelve thousand rupees a month").get("monthly_income"), 12000)

    def test_off_topic_text_yields_no_facts(self):
        for q in NEGATIVE:
            self.assertFalse(self.se.extract(q), q)


class TestIntents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.si = _si()

    def detect(self, q, active=None):
        return self.si.intents.detect(q, self.si.scenario.extract(q), active)

    def test_reference_intents(self):
        r = self.detect(T1)
        self.assertEqual(r.intent, Intent.GENERAL_SCHEME_INFORMATION)
        self.assertIn("SCH_ONORC", r.scheme_ids)
        self.assertEqual(self.detect(T2).intent, Intent.RATION_ACCESS)
        self.assertEqual(self.detect(T3).intent, Intent.SCHEME_DISCOVERY)
        self.assertEqual(self.detect(T4).intent, Intent.SCHEME_DISCOVERY)
        self.assertEqual(self.detect(T5, active="SCH_BOCW").intent, Intent.REQUIRED_DOCUMENTS)
        self.assertEqual(self.detect(T6).intent, Intent.SCHEME_DISCOVERY)

    def test_rules_decide_without_classifier(self):
        for q in (T1, T2, T3, T4, T6):
            self.assertEqual(self.detect(q).source, "rules", q)

    def test_off_topic_is_other(self):
        for q in NEGATIVE:
            self.assertEqual(self.detect(q).intent, Intent.OTHER, q)

    def test_service_and_support_intents(self):
        cases = {
            "What is the status of my application for PM Kisan?": Intent.APPLICATION_STATUS,
            "The ration dealer is cheating me": Intent.GRIEVANCE,
            "How do I link Aadhaar with my ration card?": Intent.DOCUMENT_HELP,
            "Is there any free treatment for my family?": Intent.HEALTH_SUPPORT,
            "I am pregnant, is there any help?": Intent.MATERNITY_SUPPORT,
            "Am I eligible for the widow pension?": Intent.ELIGIBILITY_CHECK,
            "How do I register on e-Shram?": Intent.APPLICATION_PROCEDURE,
            "What are the benefits of PM-JAY?": Intent.BENEFIT_INFORMATION,
        }
        for q, want in cases.items():
            self.assertEqual(self.detect(q).intent, want, q)


class TestRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.si = _si()

    def ev(self, sid, text):
        return self.si.rules.evaluate(sid, self.si.scenario.extract(text))

    def test_missing_information_never_satisfies_a_condition(self):
        empty = Profile()
        for sid in self.si.repo.all_ids():
            status = self.si.rules.evaluate(sid, empty).status
            self.assertNotIn(status, (Status.ELIGIBLE, Status.LIKELY_ELIGIBLE), sid)

    def test_onorc(self):
        self.assertEqual(self.ev("SCH_ONORC", "I have a ration card").status, Status.LIKELY_ELIGIBLE)
        full = "I have an Antyodaya ration card and my Aadhaar is linked with my ration card"
        self.assertEqual(self.ev("SCH_ONORC", full).status, Status.ELIGIBLE)
        self.assertEqual(self.ev("SCH_ONORC", "I don't have a ration card").status, Status.INELIGIBLE)

    def test_pmsym_age_income_and_exclusion(self):
        self.assertEqual(self.ev("SCH_PMSYM", T4).status, Status.LIKELY_ELIGIBLE)
        self.assertEqual(self.ev("SCH_PMSYM", "I am 45 years old, a construction worker").status, Status.INELIGIBLE)
        self.assertEqual(self.ev("SCH_PMSYM", "I am 30, a construction worker earning 20000 per month").status,
                         Status.INELIGIBLE)
        self.assertEqual(self.ev("SCH_PMSYM", "I am 30, a construction worker and I pay income tax").status,
                         Status.INELIGIBLE)

    def test_pensions_and_health(self):
        self.assertEqual(self.ev("SCH_NSAP_OA", "I am 65 years old and BPL").status, Status.ELIGIBLE)
        self.assertEqual(self.ev("SCH_NSAP_W", "I am 45, my husband died, we are BPL").status, Status.ELIGIBLE)
        self.assertEqual(self.ev("SCH_PMJAY", "I am 72 years old").status, Status.ELIGIBLE)
        self.assertNotIn(self.ev("SCH_PMJAY", "I am 32 years old").status, (Status.ELIGIBLE, Status.LIKELY_ELIGIBLE))

    def test_ended_scheme_points_to_successor(self):
        ev = self.ev("SCH_MGNREGA", "I am 30 and live in a village")
        self.assertEqual(ev.status, Status.INELIGIBLE)
        self.assertIn("ENDED", ev.notes)
        self.assertIn("REPLACED_BY:SCH_VBG_RAMG", ev.notes)

    def test_girl_child_scheme_uses_the_daughters_age(self):
        self.assertEqual(self.ev("SCH_SSY", "I am 35. My daughter is 6 years old").status, Status.LIKELY_ELIGIBLE)
        self.assertEqual(self.ev("SCH_SSY", "I am 35. My daughter is 12 years old").status, Status.INELIGIBLE)

    def test_self_reported_housing_is_not_official_selection(self):
        ev = self.ev("SCH_PMAYG", "I live in a village and I don't have a house")
        self.assertEqual(ev.status, Status.LIKELY_ELIGIBLE)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.si = _si()

    def check_shape(self, ans):
        self.assertLessEqual(len(ans.voice.split()), si_constants.MAX_VOICE_WORDS, ans.voice)
        self.assertLessEqual(len(ans.top), si_constants.DISPLAY_TOP_MAX)
        self.assertLessEqual(len(ans.bottom), si_constants.DISPLAY_BOTTOM_MAX)
        self.assertNotIn("definitely", ans.voice.lower())
        self.assertFalse(ans.used_llm)

    def test_reference_conversation(self):
        a1 = self.si.handle(T1)
        self.assertEqual((a1.route, a1.scheme_id, a1.legacy_code), (Route.DIRECT_INFORMATION, "SCH_ONORC", "PDS"))
        a2 = self.si.handle(T2)
        self.assertEqual((a2.route, a2.scheme_id, a2.status), (Route.ELIGIBILITY, "SCH_ONORC", "LIKELY_ELIGIBLE"))
        self.assertIn("Tamil Nadu", a2.voice)
        self.assertIn("Bihar", a2.voice)
        a3 = self.si.handle(T3)
        self.assertEqual(a3.route, Route.SCHEME_DISCOVERY)
        self.assertIn("SCH_BOCW", a3.scheme_ids)
        self.assertEqual(a3.asked_slot, "age")
        a4 = self.si.handle(T4)
        self.assertEqual((a4.route, a4.status), (Route.SCHEME_DISCOVERY, "LIKELY_ELIGIBLE"))
        self.assertIn("SCH_BOCW", a4.scheme_ids)
        self.assertIn("SCH_PMSYM", a4.scheme_ids)
        self.assertNotEqual(a4.asked_slot, "age")  # already known - never asked again
        a5 = self.si.handle(T5)
        self.assertEqual((a5.route, a5.scheme_id, a5.status), (Route.DIRECT_INFORMATION, "SCH_BOCW", "DOCUMENTS"))
        for a in (a1, a2, a3, a4, a5):
            self.check_shape(a)

    def test_guided_discovery_asks_one_question_and_remembers_answers(self):
        a = self.si.handle(T6)
        self.assertEqual((a.route, a.asked_slot), (Route.SCHEME_DISCOVERY, "occupation"))
        self.assertEqual(a.voice.count("?"), 1)
        a = self.si.handle("I do construction work")
        self.assertEqual(self.si.session.profile.get("occupation"), "CONSTRUCTION_WORKER")
        self.assertIn("SCH_BOCW", a.scheme_ids)
        self.assertEqual(a.asked_slot, "age")
        a = self.si.handle("32")
        self.assertEqual(self.si.session.profile.get("age"), 32)
        self.assertNotEqual(a.asked_slot, "age")
        self.check_shape(a)

    def test_off_topic_goes_to_legacy_pipeline(self):
        for q in NEGATIVE + ["translate this for the officer", "My contractor is not paying my wages"]:
            self.assertIsNone(self.si.handle(q), q)

    def test_session_expires(self):
        self.si.handle(T4)
        self.si.session.last_ts -= si_constants.SESSION_TTL_S + 1
        self.si.handle(T1)
        self.assertIsNone(self.si.session.profile.get("age"))

    def test_who_runs_a_scheme(self):
        a = self.si.handle("Which ministry runs the Ujjwala scheme?")
        self.assertEqual(a.status, "AUTHORITY")
        self.assertIn("Ministry", a.voice)

    def test_specific_questions_are_told_apart_from_what_is(self):
        from pocketinfer.applications.nomad_right.scheme_intel.service import _asks_more_than_overview as specific
        self.assertFalse(specific("what is one nation one ration card?", ["one nation one ration card"]))
        self.assertFalse(specific("give me details about pm kisan", ["pm kisan"]))
        self.assertFalse(specific("tell me about the ayushman card", ["ayushman"]))
        self.assertTrue(specific("when does the sukanya samriddhi account mature?", ["sukanya samriddhi"]))
        self.assertTrue(specific("what is the interest rate of sukanya samriddhi yojana?", ["sukanya samriddhi yojana"]))

    def test_faq_answer_is_read_without_its_question(self):
        text = ("FAQ for SCH_SSY: Question: When does a Sukanya Samriddhi Yojana (SSY) account mature? "
                "Answer: An SSY account matures 21 years from the date of its opening.")
        a = self.si.composer.from_chunk("SCH_SSY", text, Intent.GENERAL_SCHEME_INFORMATION,
                                        Route.DIRECT_INFORMATION, "FAQ_MATCH")
        self.assertTrue(a.voice.startswith("An SSY account matures 21 years"), a.voice)

    def test_llm_answers_say_where_to_confirm(self):
        a = self.si.composer.llm("A card from another state does not move automatically.", "SCH_PMJAY",
                                 ["SCH_PMJAY"], Intent.HEALTH_SUPPORT)
        self.assertIn("To confirm", a.voice)
        self.assertLessEqual(len(a.voice.split()), si_constants.MAX_VOICE_WORDS)

    def test_faq_naming_other_places_is_not_read_out(self):
        from pocketinfer.applications.nomad_right.scheme_intel.rag_engine import RAGEngine
        from pocketinfer.applications.nomad_right.scheme_intel.scenario_engine import normalize
        faq = {"scheme_id": "SCH_ONORC", "answer": "Yes, the ONORC scheme allows your family to claim their part.",
               "question": "I live in Mumbai but my family lives in Rajasthan, Can my family receive the ration at Rajasthan?"}
        with unittest.mock.patch.object(RAGEngine, "faq_ready", new_callable=unittest.mock.PropertyMock,
                                        return_value=True), \
                unittest.mock.patch.object(self.si.rag, "faq_match", return_value=(faq, 0.95)):
            for q, read_out in (("Can my wife get our ration in Bihar while I take ration in Chennai?", False),
                                ("I live in Mumbai and my family is in Rajasthan. Can they get ration there?", True)):
                ans = self.si._faq_answer(q, normalize(q), ["SCH_ONORC"], Intent.RATION_ACCESS, Route.COMPLEX_SCENARIO)
                self.assertEqual(ans is not None, read_out, q)

    def test_legacy_context_code_is_understood(self):
        a = self.si.handle(T5, context_scheme_code="PMJAY")
        self.assertEqual(a.scheme_id, "SCH_PMJAY")


class TestSwitchesAndFallbacks(unittest.TestCase):
    def test_kill_switch(self):
        from pocketinfer.applications.nomad_right.scheme_intel.service import SchemeIntelligence
        with unittest.mock.patch.object(si_constants, "SCHEME_INTEL_ENABLED", False):
            self.assertIsNone(SchemeIntelligence.create())

    def test_missing_knowledge_base_degrades_to_legacy(self):
        from pocketinfer.applications.nomad_right.scheme_intel import service
        with unittest.mock.patch.object(service, "SchemeRepository",
                                        side_effect=service.KnowledgeBaseMissing("missing")):
            self.assertIsNone(service.SchemeIntelligence.create())


class TestVision(unittest.TestCase):
    def test_downscale(self):
        import cv2
        import numpy as np
        from pocketinfer.applications.nomad_right.scheme_intel.vision import downscale_jpeg
        self.assertEqual(downscale_jpeg(b"fake-jpeg-bytes")[0], b"fake-jpeg-bytes")
        big = cv2.imencode(".jpg", np.full((1080, 1920, 3), 200, np.uint8))[1].tobytes()
        out, info = downscale_jpeg(big)
        self.assertEqual(max(info["out_w"], info["out_h"]), si_constants.VISION_MAX_SIDE)
        small = cv2.imencode(".jpg", np.full((480, 640, 3), 200, np.uint8))[1].tobytes()
        self.assertEqual(downscale_jpeg(small)[0], small)

    def test_photo_facts_feed_scheme_answers(self):
        from pocketinfer.applications.nomad_right.scheme_intel.vision import document_facts
        kind, facts = document_facts("This is a ration card issued by the Government of Bihar, card type PHH.")
        self.assertEqual(kind, "ration_card")
        self.assertEqual(facts.get("ration_card_state"), "Bihar")
        self.assertIs(facts.get("nfsa_beneficiary"), True)
        si = _si()
        # Scheme questions ask Qwen to describe the card; anything else is sent unchanged.
        self.assertIsNotNone(si.vision_question("Can I use this card in Tamil Nadu?"))
        self.assertIsNone(si.vision_question("What does this field mean?"))
        self.assertIsNone(si.handle_vision("What does this field mean?", "This field asks for your Aadhaar number."))
        ans = si.handle_vision("Can I use this card in Tamil Nadu?",
                               "This is a ration card issued by the Government of Bihar.")
        self.assertEqual((ans.route, ans.scheme_id), (Route.VISION, "SCH_ONORC"))
        self.assertIn("Bihar", ans.voice)
        self.assertLessEqual(len(ans.voice.split()), si_constants.MAX_VOICE_WORDS)


class TestText(unittest.TestCase):
    def test_money_and_units(self):
        s = ResponseComposer.speakable
        self.assertIn("5 lakh rupees per family", s("Health coverage of Rs. 5,00,000 per family"))
        self.assertIn("300 rupees a month", s("Rs 300/month widow pension (Central share)"))
        # "rs" inside ordinary words is never money
        self.assertIn("below 10 years, opened", s("for a girl child below 10 years, opened by a parent"))
        self.assertIn("workers 18 to 60", s("construction workers 18 to 60"))
        self.assertIn("up to 2 lakh rupees", s("up to Rs. 2 lakh"))

    def test_clip_keeps_whole_sentences(self):
        c = ResponseComposer(repo=None, max_words=6)
        self.assertEqual(c.clip("One two three. Four five six seven. Eight nine ten."), "One two three.")
        c = ResponseComposer(repo=None, max_words=8)
        self.assertEqual(c.clip("One two three. Four five six seven. Eight nine ten."),
                         "One two three. Four five six seven.")


class TestMetrics(unittest.TestCase):
    def test_bounded_window_and_percentiles(self):
        rec = metrics.LatencyRecorder(window=5, snapshot_path=None)
        for ms in range(1, 11):
            rec.record("stage", ms)
        p = rec.percentiles("stage")
        self.assertEqual(p["n"], 5)
        self.assertEqual((p["min"], p["max"]), (6.0, 10.0))
        self.assertLessEqual(p["p50"], p["p95"])


@unittest.skipUnless(os.environ.get("NOMADRIGHT_SI_HEAVY") == "1", "set NOMADRIGHT_SI_HEAVY=1 to run")
class TestWorkflowIntegration(unittest.TestCase):
    """Real WorkflowController against temporary copies of the legacy KB and ChromaDB."""

    @classmethod
    def setUpClass(cls):
        from pocketinfer.applications.nomad_right.config import NomadRightConfig
        from pocketinfer.applications.nomad_right.workflow import WorkflowController
        data = os.path.join(_REPO_ROOT, "nomadright")
        cls.tmp = tempfile.mkdtemp(prefix="si_test_")
        src = sqlite3.connect(f"file:{os.path.join(data, 'nomadright_kb.db')}?mode=ro", uri=True)
        dst = sqlite3.connect(os.path.join(cls.tmp, "nomadright_kb.db"))
        src.backup(dst)
        src.close()
        dst.close()
        shutil.copytree(os.path.join(data, "chroma_db"), os.path.join(cls.tmp, "chroma_db"))
        cls.wf = WorkflowController(NomadRightConfig(db_path=os.path.join(cls.tmp, "nomadright_kb.db"),
                                                     chroma_persist_dir=os.path.join(cls.tmp, "chroma_db")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.wf.scheme_intel.reset_session()

    def _vision(self, question, qwen_replies):
        asked = []

        def answer_vision(q, image_jpg, context_snippets=None):
            asked.append(q)
            return qwen_replies[len(asked) - 1]
        with unittest.mock.patch.object(self.wf.qwen_client, "answer_vision", side_effect=answer_vision):
            return self.wf.process_vision_query(question, b"not-a-jpeg"), asked

    def test_photo_scheme_question_uses_what_is_on_the_card(self):
        from pocketinfer.applications.nomad_right.scheme_intel.service import _DESCRIBE_DOCUMENT
        pkg, asked = self._vision("Can I use this ration card in Tamil Nadu?",
                                  ["This is a ration card issued by the Government of Bihar."])
        self.assertEqual(asked, [_DESCRIBE_DOCUMENT])
        self.assertEqual(pkg.scheme_code, "PDS")
        self.assertIn("Bihar", pkg.voice_text)

    def test_photo_question_falls_back_to_the_persons_own_question(self):
        from pocketinfer.applications.nomad_right.scheme_intel.service import _DESCRIBE_DOCUMENT
        q = "Can I use this ration card in Tamil Nadu?"
        pkg, asked = self._vision(q, [None, "The card says Priority Household."])
        self.assertEqual(asked, [_DESCRIBE_DOCUMENT, q])
        self.assertEqual(pkg.qr_payload["status"], "VISION_FORM_QUERY")

    def test_form_field_question_is_sent_unchanged(self):
        q = "What does this field mean?"
        pkg, asked = self._vision(q, ["This field asks for your Aadhaar number."])
        self.assertEqual(asked, [q])
        self.assertEqual(pkg.qr_payload["status"], "VISION_FORM_QUERY")

    def test_specific_question_reads_the_matching_faq(self):
        ans = self.wf.scheme_intel.handle("When does the Sukanya Samriddhi account mature?")
        self.assertEqual(ans.status, "FAQ_MATCH")
        self.assertIn("21 years", ans.voice)

    def test_passages_naming_other_places_are_not_read_out(self):
        from pocketinfer.applications.nomad_right.scheme_intel.qwen_adapter import QwenAdapter
        with unittest.mock.patch.object(QwenAdapter, "available", new_callable=unittest.mock.PropertyMock,
                                        return_value=False):
            ans = self.wf.scheme_intel.handle("Can my wife get our ration in Bihar while I take ration in Chennai?")
        voice = ans.voice if ans else ""
        self.assertNotIn("Rajasthan", voice)
        self.assertNotIn("Mumbai", voice)

    def test_scheme_question_answered_by_new_layer(self):
        pkg = self.wf.process(T2, original_query="(hi)", response_language="hi")
        self.assertEqual(pkg.scheme_code, "PDS")
        self.assertEqual(pkg.qr_payload["status"], "LIKELY_ELIGIBLE")
        self.assertTrue(pkg.qr_payload["intent"].startswith("SI_"))

    def test_off_topic_still_uses_legacy_fallback(self):
        with unittest.mock.patch.object(self.wf.qwen_client, "answer_general", return_value=None):
            pkg = self.wf.process("Yapri festival at Kudab Uri")
        self.assertEqual(pkg.qr_payload["status"], "FALLBACK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
