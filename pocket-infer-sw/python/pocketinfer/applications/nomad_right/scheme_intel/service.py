"""
SchemeIntelligence - the facade WorkflowController talks to.

handle() returns an SIAnswer when this layer can answer from the knowledge
base, or None to hand the query back to the legacy pipeline (off-topic
questions, small talk, wage-rights cases the legacy rules already cover,
anything this layer has no grounded answer for). It never raises into the
caller's turn - WorkflowController also wraps it in try/except.

Per turn:  scenario facts -> session merge -> intent -> route -> SQLite /
rules / (RAG) / (Qwen, complex only) -> composer -> session update.
"""

import logging
import re
import time
from typing import Any, Callable, List, Optional

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel import vision as vision_mod
from pocketinfer.applications.nomad_right.scheme_intel.composer import ResponseComposer, words
from pocketinfer.applications.nomad_right.scheme_intel.discovery import (
    is_migrant, pick_question, rank, recommendable,
)
from pocketinfer.applications.nomad_right.scheme_intel.gazetteer import find_places
from pocketinfer.applications.nomad_right.scheme_intel.intent_engine import IntentEngine
from pocketinfer.applications.nomad_right.scheme_intel.metrics import LATENCY
from pocketinfer.applications.nomad_right.scheme_intel.models import (
    DOMAIN_OF_INTENT, Intent, QType, Route, RouteDecision, SIAnswer, Status,
)
from pocketinfer.applications.nomad_right.scheme_intel.qwen_adapter import QwenAdapter
from pocketinfer.applications.nomad_right.scheme_intel.rag_engine import RAGEngine
from pocketinfer.applications.nomad_right.scheme_intel.repository import KnowledgeBaseMissing, SchemeRepository
from pocketinfer.applications.nomad_right.scheme_intel.router import QueryRouter
from pocketinfer.applications.nomad_right.scheme_intel.rules_engine import RulesEngine, derive
from pocketinfer.applications.nomad_right.scheme_intel.scenario_engine import ScenarioEngine, normalize
from pocketinfer.applications.nomad_right.scheme_intel.session import SessionState

logger = logging.getLogger("SchemeIntelligence")

# Scheme questions a photo can meaningfully feed into (a form-field question stays with Qwen alone).
_VISION_SCHEME_INTENTS = {Intent.RATION_ACCESS, Intent.ELIGIBILITY_CHECK, Intent.MIGRANT_SUPPORT,
                          Intent.HEALTH_SUPPORT, Intent.REQUIRED_DOCUMENTS, Intent.APPLICATION_PROCEDURE,
                          Intent.BENEFIT_INFORMATION, Intent.SCHEME_DISCOVERY}
_AADHAAR_SEEDING_RE = re.compile(r"(?=.*\baadhaar\b)(?=.*\bration\b)(?=.*\b(?:link|linked|linking|seed|seeded|seeding|connect|add)\b)")
_WHICH_STATE_RE = re.compile(r"\bwhich (?:board|state)\b|\bas a migrant\b")
_AUTHORITY_RE = re.compile(r"\b(?:ministry|department|agency|who runs|implement\w*|in charge|responsible|nodal)\b")
_PORTABLE_SCHEMES = ("SCH_PMJAY", "SCH_ESHRAM", "SCH_BOCW")
# The scheme a photographed card belongs to ("can I use THIS card here?").
_DOC_SCHEME = {"ration_card": "SCH_ONORC", "labour_card": "SCH_BOCW", "eshram_card": "SCH_ESHRAM",
               "ayushman_card": "SCH_PMJAY", "job_card": "SCH_VBG_RAMG"}
# What Qwen is asked when the spoken question is about a scheme: it can read a
# card, but "can I use this card in Tamil Nadu?" is not visible in the photo
# (measured: Qwen answered the not-found sentinel) - the scheme part comes
# from the knowledge base instead.
_DESCRIBE_DOCUMENT = ("What document is this? Read the issuing state or department, the card type or category, "
                      "and the district if they are written on it.")
# Words that ask nothing beyond "what is <scheme>?" - any other word makes the
# question specific ("when does it mature?", "interest rate").
_OVERVIEW_FILLER = frozenset((
    "a an the is are was what whats s tell me us about give details detail information info explain please "
    "scheme schemes yojana yojna plan programme program abhiyan mission card cards pm pradhan mantri do does i we "
    "you your my can could to of on for in and or know want like more this that it its government govt sarkari "
    "hai kya ke ka ki batao bataiye kuch").split())


def _asks_more_than_overview(t_norm: str, aliases: List[str]) -> bool:
    rest = t_norm
    for alias in sorted(aliases, key=len, reverse=True):
        rest = rest.replace(alias, " ")
    return any(w not in _OVERVIEW_FILLER for w in re.findall(r"[a-z]+", rest))


def _names_other_places(passage: str, t_norm: str) -> bool:
    """True when a passage about to be read out names a state or city the person did
    not ("I live in Mumbai, my family in Rajasthan..." to someone in Bihar and Chennai)."""
    said = {m.state for m in find_places(t_norm) if m.state}
    return bool({m.state for m in find_places(passage.lower()) if m.state} - said)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class SchemeIntelligence:
    def __init__(self, repo: SchemeRepository, embedder_provider: Callable[[], Any], qwen_client=None):
        self.repo = repo
        self.scenario = ScenarioEngine()
        self.rag = RAGEngine(repo, embedder_provider)
        self.intents = IntentEngine(repo, embed_fn=self.rag.embed_query)
        self.rules = RulesEngine(repo)
        self.router = QueryRouter(repo)
        self.composer = ResponseComposer(repo)
        self.qwen = QwenAdapter(qwen_client)
        self.session = SessionState()

    @classmethod
    def create(cls, embedder_provider: Optional[Callable[[], Any]] = None, qwen_client=None) -> Optional["SchemeIntelligence"]:
        if not C.SCHEME_INTEL_ENABLED:
            logger.info("[SI] disabled (SCHEME_INTEL_ENABLED / NOMADRIGHT_SCHEME_INTEL=0) - legacy pipeline only")
            return None
        t0 = time.perf_counter()
        try:
            si = cls(SchemeRepository(), embedder_provider or (lambda: None), qwen_client)
        except KnowledgeBaseMissing as exc:
            logger.warning(f"[SI] {exc} - legacy pipeline only")
            return None
        except Exception as exc:
            logger.warning(f"[SI] could not start ({exc}) - legacy pipeline only", exc_info=True)
            return None
        logger.info(f"[SI] ready in {_ms(t0):.0f} ms: {si.repo.stats()} vector_index={si.rag.index_ready} "
                    f"classifier={si.intents.classifier_ready} built={si.repo.meta.get('built_at')}")
        return si

    def reset_session(self) -> None:
        self.session.reset()

    # ── text turn ─────────────────────────────────────────────────────────
    def handle(self, query_en: str, context_scheme_code: Optional[str] = None,
               original_query: Optional[str] = None, response_language: Optional[str] = None) -> Optional[SIAnswer]:
        t0 = time.perf_counter()
        s = self.session
        if s.expire_if_idle():
            logger.info("[SI] session idle > %ss - profile cleared", int(C.SESSION_TTL_S))
        text = (query_en or "").strip()
        if not text:
            return None
        t_norm = normalize(text)

        ts = time.perf_counter()
        facts = self.scenario.extract(text)
        if s.pending_slot and not facts.has(s.pending_slot) and s.pending_slot not in facts.bounds:
            yn = s.boolean_reply(t_norm)
            if yn is not None:
                facts.set(s.pending_slot, yn, f"reply: {t_norm[:24]}")
            else:
                stmt = s.pending_reply_as_statement(t_norm)
                if stmt:
                    facts.merge(self.scenario.extract(stmt))
        LATENCY.record("si_scenario", _ms(ts))
        changed = s.absorb(facts)

        context_sid = self.repo.sid_for_legacy(context_scheme_code)
        ti = time.perf_counter()
        ir = self.intents.detect(text, facts, s.active_scheme_id or context_sid, s.last_candidates)
        LATENCY.record("si_intent", _ms(ti))
        # A turn that only answers the pending question continues the previous intent.
        if ir.intent == Intent.OTHER and changed and s.current_intent and s.current_intent != Intent.OTHER:
            ir.intent, ir.source, ir.confidence = s.current_intent, "session", 0.7

        decision = self.router.route(ir, t_norm, s.active_scheme_id, context_sid)
        answer: Optional[SIAnswer] = None
        if decision.route != Route.DEFER:
            try:
                answer = self._execute(decision, ir, text, t_norm)
            except Exception as exc:
                logger.warning(f"[SI] route {decision.route.value} failed ({exc}) - deferring to legacy", exc_info=True)
                answer = None
        total = _ms(t0)
        LATENCY.record("si_total", total)
        s.touch()
        if answer is None:
            logger.info(f"[SI] DEFER intent={ir.intent.value}({ir.source}) route={decision.route.value} "
                        f"reason='{decision.reason}' facts={','.join(changed) or '-'} ms={total:.1f}")
            return None

        s.remember(ir.intent, scheme_id=answer.scheme_id,
                   candidates=answer.scheme_ids if answer.route == Route.SCHEME_DISCOVERY else None,
                   asked_slot=answer.asked_slot)
        answer.trace = {"intent": ir.intent.value, "intent_source": ir.source, "confidence": ir.confidence,
                        "qtype": ir.qtype.value, "domains": ir.domains, "route": decision.route.value,
                        "reason": decision.reason, "facts_this_turn": changed, "ms": round(total, 1),
                        "response_language": response_language, "has_original_query": bool(original_query)}
        logger.info(f"[SI] intent={ir.intent.value}({ir.source},{ir.confidence:.2f}) route={answer.route.value} "
                    f"scheme={answer.scheme_id or '-'} status={answer.status or '-'} "
                    f"facts={','.join(changed) or '-'} ask={answer.asked_slot or '-'} "
                    f"llm={'yes' if answer.used_llm else 'no'} words={words(answer.voice)} ms={total:.1f}")
        return answer

    # ── routes ────────────────────────────────────────────────────────────
    def _execute(self, decision: RouteDecision, ir, text: str, t_norm: str) -> Optional[SIAnswer]:
        if decision.route == Route.CLARIFY:
            if decision.reason == "unverified":
                return self.composer.unverified(ir.unverified_name, ir.intent)
            return self.composer.clarify_scheme(self.session.last_candidates, ir.intent)
        if decision.route == Route.DIRECT_INFORMATION:
            return self._direct(decision, ir, text, t_norm)
        if decision.route == Route.ELIGIBILITY:
            return self._eligibility(decision, ir)
        if decision.route == Route.SCHEME_DISCOVERY:
            return self._discovery(ir, text)
        if decision.route == Route.COMPLEX_SCENARIO:
            return self._complex(decision, ir, text, t_norm)
        return None

    def _direct(self, decision: RouteDecision, ir, text: str, t_norm: str) -> Optional[SIAnswer]:
        sid, intent, d = decision.scheme_id, ir.intent, self.session.profile
        if decision.reason == "migrant overview":
            if sid and sid in (d.get("existing_scheme_membership") or []):
                ans = self.composer.portability(sid, d, intent)
                if ans:
                    return ans
            return self.composer.migrant_overview(d, intent)
        if intent == Intent.DOCUMENT_HELP:
            if _AADHAAR_SEEDING_RE.search(t_norm):
                return self.composer.aadhaar_seeding_help(intent)
            hit = self._best_chunk(text, [sid] if sid else None, C.RAG_DIRECT_ANSWER_SCORE)
            if not hit or _names_other_places(hit[0]["text"], t_norm):
                return None
            return self.composer.from_chunk(hit[0]["scheme_id"], hit[0]["text"], intent, Route.DIRECT_INFORMATION,
                                            "DOC_HELP")
        if intent == Intent.GRIEVANCE:
            return self.composer.grievance(sid, intent)
        if intent == Intent.APPLICATION_STATUS:
            if sid:
                return self.composer.contact(sid, intent, status="STATUS")
            return self.composer.clarify_scheme(self.session.last_candidates, intent)
        if not sid:
            return None
        # "Which board do I register with?", "does it work in another state?" for
        # the schemes whose knowledge-base records say how they cross states.
        if (sid in _PORTABLE_SCHEMES and ir.qtype not in (QType.DOCUMENTS, QType.FEES, QType.CONTACT)
                and intent in (Intent.APPLICATION_PROCEDURE, Intent.GENERAL_SCHEME_INFORMATION, Intent.BENEFIT_INFORMATION)
                and (ir.portability_cue or _WHICH_STATE_RE.search(t_norm))):
            ans = self.composer.portability(sid, d, intent)
            if ans:
                return ans
        if intent == Intent.REQUIRED_DOCUMENTS:
            return self.composer.documents(sid, intent)
        if intent == Intent.APPLICATION_PROCEDURE:
            return self.composer.procedure(sid, intent, d)
        if intent == Intent.BENEFIT_INFORMATION:
            return self.composer.benefits(sid, intent)
        if ir.qtype == QType.FEES:
            return self.composer.fees(sid, intent)
        if ir.qtype == QType.CONTACT:
            if _AUTHORITY_RE.search(t_norm):
                ans = self.composer.authority(sid, intent)
                if ans:
                    return ans
            return self.composer.contact(sid, intent)
        aliases = [alias for found, _strength, alias, _pos in self.repo.find_schemes(t_norm) if found == sid]
        if _asks_more_than_overview(t_norm, aliases):
            # A specific question the structured fields do not cover ("when does the
            # Sukanya account mature?"): the scheme's own FAQ answers it if one matches.
            ans = self._faq_answer(text, t_norm, [sid], intent, Route.DIRECT_INFORMATION)
            if ans:
                return ans
        return self.composer.overview(sid, intent)

    def _eligibility(self, decision: RouteDecision, ir) -> Optional[SIAnswer]:
        sid, intent, d = decision.scheme_id, ir.intent, self.session.profile
        if intent == Intent.HEALTH_SUPPORT and "Punjab" in (d.get("current_state"), d.get("origin_state")):
            sid = "SCH_ABPMJAY" if self.repo.has("SCH_ABPMJAY") else sid
        if sid in _PORTABLE_SCHEMES and (
                (sid in (d.get("existing_scheme_membership") or []) and (ir.portability_cue or d.get("current_state")))
                or (sid == "SCH_BOCW" and ir.portability_cue and is_migrant(d))):
            ans = self.composer.portability(sid, d, intent)
            if ans:
                return ans
        tr = time.perf_counter()
        derived = derive(d)
        ev = self.rules.evaluate(sid, d, derived)
        LATENCY.record("si_rules", _ms(tr))
        if sid == "SCH_ONORC":
            return self.composer.ration_access(ev, d, intent)
        slot = None
        if ev.status in (Status.POTENTIALLY_ELIGIBLE, Status.INSUFFICIENT_INFORMATION):
            slot = pick_question([ev], derived, self.session.asked_slots, intent, restrict=False)
        ans = self.composer.eligibility(ev, d, intent, slot)
        # A support-area question ("is there help for treatment?") first says what the scheme is.
        if (intent in DOMAIN_OF_INTENT and ir.qtype != QType.ELIGIBILITY
                and ev.status not in (Status.INELIGIBLE, Status.INSUFFICIENT_INFORMATION)):
            combined = f"{self.composer.speakable(self.composer._summary_sentence(sid))} {ans.voice}"
            if words(combined) <= C.MAX_VOICE_WORDS:
                ans.voice = combined
        return ans

    def _candidate_pool(self, domains: List[str]) -> List[str]:
        d = self.session.profile
        states = [st for st in (d.get("current_state"), d.get("origin_state")) if st] or [None]
        pool: List[str] = []
        for st in states:
            for sid in self.repo.candidates(domains=domains or None, state=st):
                if sid not in pool:
                    pool.append(sid)
        return pool

    def _discovery(self, ir, text: str) -> Optional[SIAnswer]:
        d = self.session.profile
        derived = derive(d)
        domains = [DOMAIN_OF_INTENT[ir.intent]] if ir.intent in DOMAIN_OF_INTENT else []
        tq = time.perf_counter()
        cands = self._candidate_pool(domains)
        LATENCY.record("si_sql", _ms(tq))
        tr = time.perf_counter()
        evals = [self.rules.evaluate(sid, d, derived) for sid in cands]
        LATENCY.record("si_rules", _ms(tr))
        rag_scores = None
        if ir.source == "classifier" and self.rag.index_ready:
            tv = time.perf_counter()
            rag_scores = self.rag.scheme_scores(self.rag.embed_query(text), cands)
            LATENCY.record("si_rag", _ms(tv))
        ranked = rank(evals, self.repo, self.rules, derived, domains, rag_scores)
        top = recommendable(ranked, 3)
        slot = pick_question(ranked, derived, self.session.asked_slots, ir.intent)
        if not top and domains:
            # Nothing matches yet: offer the topic's schemes that are open to this
            # person (not ones targeted at a different occupation).
            fitting = [ev for ev in ranked
                       if not self.repo.scheme(ev.scheme_id).get("occupation") or self.rules.targets_person(ev.scheme_id, derived)]
            if len(fitting) == 1:
                one = fitting[0]
                q = pick_question([one], derived, self.session.asked_slots, ir.intent, restrict=False)
                ans = self.composer.eligibility(one, d, ir.intent, q)
                ans.route = Route.SCHEME_DISCOVERY
                return ans
            return self.composer.discovery(top, d, ir.intent, slot, domains[0], [ev.scheme_id for ev in fitting[:2]])
        return self.composer.discovery(top, d, ir.intent, slot)

    def _best_chunk(self, text: str, scheme_ids: Optional[List[str]], min_score: float):
        if not self.rag.index_ready:
            return None
        tv = time.perf_counter()
        hits = self.rag.search(self.rag.embed_query(text), scheme_ids=scheme_ids, top_k=1, min_score=min_score)
        LATENCY.record("si_rag", _ms(tv))
        return hits[0] if hits else None

    def _faq_answer(self, text: str, t_norm: str, scheme_ids: Optional[List[str]], intent: Intent,
                    route: Route) -> Optional[SIAnswer]:
        """Reads out the knowledge-base FAQ whose question matches this one, unless it
        names places the person did not (its example places would mislead)."""
        if not self.rag.faq_ready:
            return None
        tv = time.perf_counter()
        hit = self.rag.faq_match(self.rag.embed_query(text), scheme_ids)
        LATENCY.record("si_rag", _ms(tv))
        if hit is None:
            return None
        faq, _score = hit
        if _names_other_places(f"{faq['question']} {faq['answer']}", t_norm):
            return None
        return self.composer.from_chunk(faq["scheme_id"], faq["answer"], intent, route, "FAQ_MATCH")

    def _complex(self, decision: RouteDecision, ir, text: str, t_norm: str) -> Optional[SIAnswer]:
        d = self.session.profile
        derived = derive(d)
        cands = [c for c in decision.candidates if self.repo.has(c)] or self.session.last_candidates[:3]
        rule_lines = []
        for sid in cands[:3]:
            ev = self.rules.evaluate(sid, d, derived)
            need = ev.missing_fields(include_verify=False)[:2]
            rule_lines.append(f"{self.repo.spoken(sid)}: {ev.status.value.replace('_', ' ').lower()}"
                              + (f" (unknown: {', '.join(need)})" if need else ""))
        if not self.rag.index_ready:
            return None
        faq = self._faq_answer(text, t_norm, cands or None, ir.intent, Route.COMPLEX_SCENARIO)
        if faq:
            return faq
        tv = time.perf_counter()
        hits = self.rag.search(self.rag.embed_query(text), scheme_ids=cands or None, top_k=3)
        LATENCY.record("si_rag", _ms(tv))
        if not hits:
            return None
        best_chunk = hits[0][0]
        if self.qwen.available:
            tl = time.perf_counter()
            answer, declined = self.qwen.complex_answer(text, derived.summary(8), rule_lines, [h[0]["text"] for h in hits])
            LATENCY.record("si_llm", _ms(tl))
            if answer:
                return self.composer.llm(answer, cands[0] if cands else best_chunk["scheme_id"], cands, ir.intent)
            if declined:
                return None
        # Qwen unavailable: read out the best passage that names no other places.
        readable = [c for c, _score in hits if not _names_other_places(c["text"], t_norm)]
        if not readable:
            return None
        return self.composer.from_chunk(readable[0]["scheme_id"], readable[0]["text"], ir.intent,
                                        Route.COMPLEX_SCENARIO, "RAG_TEMPLATE")

    # ── vision turn (Camera button) ───────────────────────────────────────
    @staticmethod
    def prepare_image(jpg: bytes):
        return vision_mod.downscale_jpeg(jpg)

    def vision_question(self, query_en: str) -> Optional[str]:
        """
        The question to send to Qwen3-VL with the photo: for a scheme question
        ("can I use this card in Tamil Nadu?") ask it to describe the document;
        for anything else (a form field, "what is written here?") None, i.e.
        send the person's own question exactly as before.
        """
        text = (query_en or "").strip()
        if not text:
            return None
        ir = self.intents.detect(text, self.scenario.extract(text), self.session.active_scheme_id,
                                 self.session.last_candidates)
        return _DESCRIBE_DOCUMENT if ir.intent in _VISION_SCHEME_INTENTS else None

    def handle_vision(self, query_en: str, vision_answer: str,
                      context_scheme_code: Optional[str] = None) -> Optional[SIAnswer]:
        """
        Camera -> Qwen3-VL -> structured facts -> scheme intelligence. Facts read
        from the photo (a Bihar ration card, an Ayushman card...) join the session;
        if the spoken question is a scheme question, the answer is grounded in the
        knowledge base and prefixed with what Qwen saw. Otherwise None (the plain
        form-reading answer stands).
        """
        t0 = time.perf_counter()
        doc_type, facts = vision_mod.document_facts(vision_answer)
        if facts:
            self.session.absorb(facts)
        text = (query_en or "").strip()
        if not text:
            return None
        t_norm = normalize(text)
        text_facts = self.scenario.extract(text)
        self.session.absorb(text_facts)
        # The photographed card names the scheme "this card" refers to.
        context_sid = _DOC_SCHEME.get(doc_type) or self.repo.sid_for_legacy(context_scheme_code)
        if _DOC_SCHEME.get(doc_type):
            self.session.active_scheme_id = context_sid
        ir = self.intents.detect(text, text_facts, self.session.active_scheme_id or context_sid, self.session.last_candidates)
        if ir.intent not in _VISION_SCHEME_INTENTS:
            logger.info(f"[SI] vision: doc={doc_type or '-'} intent={ir.intent.value} - plain photo answer kept")
            return None
        decision = self.router.route(ir, t_norm, self.session.active_scheme_id, context_sid)
        if decision.route not in (Route.ELIGIBILITY, Route.DIRECT_INFORMATION, Route.SCHEME_DISCOVERY):
            return None
        answer = self._execute(decision, ir, text, t_norm)
        if answer is None:
            return None
        seen = self.composer.first_clause(self.composer.speakable(vision_answer), 12)
        combined = f"{seen} {answer.voice}"
        answer.voice = combined if words(combined) <= C.MAX_VOICE_WORDS else answer.voice
        answer.route = Route.VISION
        self.session.remember(ir.intent, scheme_id=answer.scheme_id, asked_slot=answer.asked_slot)
        LATENCY.record("si_vision", _ms(t0))
        logger.info(f"[SI] vision: doc={doc_type or '-'} intent={ir.intent.value} scheme={answer.scheme_id or '-'} "
                    f"status={answer.status or '-'} words={words(answer.voice)}")
        return answer

    def health(self) -> dict:
        return {"kb": self.repo.stats(), "built_at": self.repo.meta.get("built_at"),
                "vector_index": self.rag.index_ready, "classifier": self.intents.classifier_ready,
                "qwen": self.qwen.available, "session": self.session.snapshot()}
