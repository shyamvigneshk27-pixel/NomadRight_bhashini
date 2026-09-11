"""
IntentEngine - lightweight intent detection. Rules first; a small
nearest-neighbour classifier over e5 embeddings of labelled example questions
only when the rules are inconclusive. Qwen is never used for intent.

Two independent signals are combined:
  * question type (QType): documents / procedure / benefits / eligibility /
    overview / contact / fees / status / grievance / document help
  * topic (domain): ration, health, worker rights, pension, maternity,
    education, housing, disability, finance, migration
plus explicit scheme mentions (alias lexicon from the knowledge base).
"""

import logging
import os
import re
from typing import Callable, List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel.models import (
    INTENT_OF_DOMAIN, Intent, IntentResult, Profile, QType,
)
from pocketinfer.applications.nomad_right.scheme_intel.scenario_engine import normalize

logger = logging.getLogger(__name__)

# Order = precedence when several match.
_QTYPE_RULES: List[Tuple[QType, "re.Pattern"]] = [(qt, re.compile(p)) for qt, p in [
    (QType.STATUS,
     r"\b(?:status|track(?:ing)? my|when will i (?:get|receive)|(?:has|have) not (?:yet )?(?:been )?(?:received|credited|come|issued)"
     r"|not (?:yet )?(?:received|credited|issued)|(?:money|payment|installment|instalment|pension|amount|card) (?:has |is )?not (?:come|received|credited|issued|arrived)"
     r"|application (?:is )?(?:still )?pending|where is my (?:application|card|money|pension))\b"),
    (QType.GRIEVANCE,
     r"\b(?:complain(?:t|ts)?|grievance|cheat(?:ed|ing)?|bribe|harass(?:ed|ment)?|fraud|refus(?:ed|es|ing)|den(?:ied|ying)"
     r"|(?:not|n't) (?:being )?(?:given|giving|allowed|allowing)|rejected unfairly|misbehav\w*)\b"),
    (QType.CONTACT,
     r"\b(?:helpline|help line|toll[- ]?free|contact number|phone number|call (?:number|centre|center)"
     r"|whom (?:should|do|can) i (?:call|contact)|who (?:should|do|can) i (?:call|contact)|website|web site|official site|portal"
     r"|(?:which|what) (?:ministry|department|agency)|who (?:runs|implements|manages|handles|is in charge of)"
     r"|implemented by|implementing agency|nodal (?:ministry|agency|officer))\b"),
    (QType.DOC_HELP,
     r"\b(?:lost|misplaced|stolen|damaged) (?:my |our )?(?:[\w-]+ )?(?:card|aadhaar|document|certificate|passbook)\b"
     r"|\b(?:link|seed|update|correct|change) (?:my |the )?(?:aadhaar|mobile number|name|address)\b"
     r"|\b(?:get|make|apply for|obtain|create) (?:a |an |my |new )+(?:aadhaar|ration card|pan card|voter id|income certificate"
     r"|caste certificate|disability certificate|udid|birth certificate|domicile certificate)\b|\bnew ration card\b"
     r"|\bwrong (?:name|details) (?:in|on) (?:my )?(?:aadhaar|ration card|card)\b"),
    (QType.DOCUMENTS,
     r"\b(?:documents?|papers?|proofs?|kaagaz|kagaz|dastavez|certificates? (?:needed|required)"
     r"|what (?:do|should) i (?:need to )?(?:carry|bring|submit|show)|what to (?:carry|bring|submit)"
     r"|what (?:all )?(?:is|are) (?:needed|required) (?:to|for))\b"),
    (QType.PROCEDURE,
     r"\b(?:how (?:do|can|to|should) (?:i |we )?(?:make|get|obtain|apply for|create) (?:a |an |my |the |new )?(?:[\w'-]+ ){0,3}card\b"
     r"|how (?:do|can|to|should|will) (?:i |we )?(?:apply|register|enrol+|join|sign up|get registered|get enrolled"
     r"|get (?:it|this|the card|a card|the benefit|a job card|a labou?r card|a shram card))|how to (?:apply|register|join|get|enrol+)"
     r"|where (?:do|can|should) (?:i|we) (?:apply|register|go|enrol+)|where to (?:apply|register|go)|application (?:process|procedure|form)"
     r"|procedure|process to|steps? (?:to|for)|register|registration|enrol+ment|apply (?:online|offline)|online application|which office)\b"),
    (QType.FEES,
     r"\b(?:fees?|charges?|is (?:it|this|registration|the card) free|free of cost|do i (?:have|need) to pay"
     r"|how much (?:do|will|should) i (?:have to )?pay|premium|contribution|cost to apply|how much does it cost)\b"),
    (QType.BENEFITS,
     r"\b(?:benefits?|advantages?|what (?:do|will|can|would) (?:i|we) get|how much (?:money|pension|amount|will i get|do i get"
     r"|is given|will be given|rice|grain|loan|subsidy|cover|insurance)|amount|assistance|coverage|covers?"
     r"|what does (?:it|this|the scheme) (?:give|provide|offer|cover)|entitle(?:d|ment)s?|subsidy|how much)\b"),
    (QType.ELIGIBILITY,
     r"\b(?:eligib(?:le|ility)|qualif(?:y|ied|ication)|am i (?:allowed|entitled|able|covered)"
     r"|can (?:i|we|my family) (?:get|avail|apply|join|use|claim|take|receive|collect|register|buy)|will (?:i|we) get|do (?:i|we) get"
     r"|who (?:can|is eligible to) (?:apply|get|join)|criteria|conditions?|requirements?|age limit|income limit"
     r"|will (?:it|my card|my [\w-]+ card) work|does (?:it|my card) work|is (?:it|my card) valid)\b"),
    (QType.OVERVIEW,
     r"\b(?:what is|what's|what are|tell me (?:about|more)|explain|details (?:of|about)|information (?:on|about)"
     r"|meaning of|about (?:the )?(?:scheme|yojana)|kya hai|what does .{1,30} mean)\b"),
]]

# Topic lexicon; list order breaks ties.
_DOMAIN_RULES: List[Tuple[str, "re.Pattern"]] = [(d, re.compile(p)) for d, p in [
    ("RATION", r"\b(?:ration|rations|food ?grains?|rice|wheat|pds|fair price shop|ration shop|ration dealer|onorc|annapurna|food security|nfsa|anaj)\b"),
    ("HEALTH", r"\b(?:health|hospital|hospitali[sz]ation|treatment|medical|medicine|doctor|surgery|operation|illness|sick|disease|ayushman|pm-?jay|golden card)\b"),
    ("MATERNITY", r"\b(?:pregnan(?:t|cy)|maternity|delivery|childbirth|newborn|expecting a baby|baby is due)\b"),
    ("DISABILITY", r"\b(?:disab\w*|handicap\w*|divyang|blind|deaf|wheelchair|differently abled)\b"),
    ("PENSION", r"\b(?:pension|old age|retire(?:ment|d)?|after 60|budhapa)\b"),
    ("HOUSING", r"\b(?:housing|a house|own house|new house|pucca house|kutcha house|build (?:a )?(?:house|home)|shelter|awaas|awas|homeless|houseless|roof over)\b"),
    ("EDUCATION", r"\b(?:education|school|scholarship|college|studies|studying|tuition|skill training|training|course|learn a skill|skills?)\b"),
    ("WORKER", r"\b(?:wages?|salary not|not paid|unpaid|contractor|employer|labou?r rights|worker'?s? rights|minimum wage|worksite|work site|accident at work|injured at work|labou?r office|labou?r department|overtime)\b"),
    ("MIGRANT", r"\b(?:migrant|migrated|migration|moved to|shifted to|another state|other state|home state|native state|outside my state|different state|portab\w*)\b"),
    ("FINANCIAL", r"\b(?:loan|loans|bank account|savings|credit|money|finance|financial|insurance|subsidy|cash|business|mudra|debit card|overdraft)\b"),
]]

_DISCOVERY_RE = re.compile(
    r"\b(?:(?:which|what|any|all) (?:\w+ ){0,2}(?:government |sarkari |central |state )?(?:schemes?|yojanas?|benefits?|help|support|facilit(?:y|ies)|programmes?|programs?|welfare)"
    r"|(?:schemes?|benefits?|help|support|facilities) (?:for|available (?:for|to)) (?:me|us|my family|people like me|workers like me|someone like me|a person like me)"
    r"|what (?:can|could|do) (?:i|we) get|how can (?:the )?government help|(?:don't|do not|dont) know (?:what|which|about any|anything about) (?:\w+ ){0,2}(?:scheme|schemes|benefits?|yojana)"
    r"|suggest (?:a |some |any )?schemes?|recommend (?:a |some )?schemes?|help me (?:find|get|choose)|am i (?:eligible|entitled) (?:for|to) (?:any|anything|some)"
    r"|what (?:all )?(?:does|can|will) the government (?:give|provide|offer|do)|is there (?:any|a) (?:scheme|yojana|benefit|help)|what help|can you help me)\b")
_PORTABILITY_RE = re.compile(
    r"\b(?:here|this (?:state|city|place)|another (?:state|city|place)|other (?:state|states|city)|different (?:state|city)|new (?:state|city|place)"
    r"|outside (?:my|the|our) (?:state|home|village)|portab\w*|transfer\w*|moved|shifted|migrat\w*|anywhere in india|all over india|across (?:india|states)"
    r"|(?:will|does|would|can|is) (?:it|this|that|my (?:[\w-]+ )?card|the card) (?:\w+ )?(?:work|be valid|valid|be accepted|accepted)"
    r"|use (?:it|this|my (?:[\w-]+ )?card) (?:\w+ ){0,2}(?:in|at))\b")
# "Am I eligible / who can apply / criteria": outranks the looser benefit and procedure words.
_STRONG_ELIG_RE = re.compile(
    r"\b(?:eligib(?:le|ility)|qualif(?:y|ied|ication)|am i (?:allowed|entitled|covered)"
    r"|who (?:can|is eligible to|are eligible to) (?:apply|get|join|avail)|who is eligible|criteria|age limit|income limit"
    r"|can (?:i|we) (?:join|apply for|register for|avail))\b")
_CONTEXT_REF_RE = re.compile(
    r"\b(?:this|that|the same|the above|above|previous|same) (?:scheme|yojana|one|card|programme|program|plan|pension|insurance)\b"
    r"|\b(?:for|about|of|join|apply for|register for|get|use) (?:it|this|that)\b|^(?:and |what about |how about )?(?:it|this|that)\b"
    r"|\b(?:is|does|can) it\b")
_ORDINAL_RE = re.compile(r"\b(?:the )?(first|second|third|1st|2nd|3rd|last) (?:one|scheme|option|yojana)\b")
_COMPLEX_RE = re.compile(
    r"\b(?:what (?:happens|will happen)|what if|why|both|instead of|compare|difference between|which is better|better than|while i"
    r"|if i (?:go|move|return|shift|change|leave|die|get|lose|quit)|after i|at the same time|can my (?:wife|husband|family|son|daughter|mother|father)|together)\b")
_QUESTION_RE = re.compile(
    r"\?|^(?:what|which|how|who|whom|where|when|why|can|could|is|are|am|do|does|did|will|would|should|shall|may|tell|explain|give|please|help)\b"
    r"|\b(?:can i|could i|should i|do i|will i|am i|is there|are there|tell me|help me)\b")
_SCHEME_CARD_RE = re.compile(r"\b(?:labou?r|bocw|construction worker'?s?|job|e-shram|uan|shram|ayushman|golden|pm-?jay) card\b")
_ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": -1}


class IntentEngine:
    def __init__(self, repo, embed_fn: Optional[Callable[[str], object]] = None,
                 prototypes_path: str = C.INTENT_PROTOTYPES_PATH):
        self.repo = repo
        self._embed = embed_fn
        self._protos = None
        self._labels = None
        if embed_fn is not None and os.path.exists(prototypes_path):
            try:
                import numpy as np
                data = np.load(prototypes_path)
                self._protos = data["embeddings"].astype("float32")
                self._labels = [str(x) for x in data["labels"]]
            except Exception as exc:  # classifier is optional - rules still work
                logger.warning(f"[SI] intent prototypes unavailable ({exc}); rules only")

    @property
    def classifier_ready(self) -> bool:
        return self._protos is not None

    # ── public ────────────────────────────────────────────────────────────
    def detect(self, text: str, facts: Profile, active_scheme_id: Optional[str] = None,
               last_candidates: Optional[List[str]] = None) -> IntentResult:
        t = normalize(text)
        res = IntentResult(intent=Intent.OTHER, confidence=0.0, source="none")
        if not t:
            return res
        found = self.repo.find_schemes(t)
        res.scheme_ids = [sid for sid, strength, _a, _p in found if strength >= 0.8]
        res.scheme_hints = [sid for sid, strength, _a, _p in found if strength < 0.8]
        res.unverified_name = self.repo.find_unverified(t)
        res.refers_to_context = bool(_CONTEXT_REF_RE.search(t))
        m = _ORDINAL_RE.search(t)
        if m and last_candidates:
            idx = _ORDINALS[m.group(1)]
            if -len(last_candidates) <= idx < len(last_candidates):
                res.scheme_ids.insert(0, last_candidates[idx])
                res.matched.append(f"ordinal:{m.group(1)}")
        res.discovery_cue = bool(_DISCOVERY_RE.search(t))
        res.portability_cue = bool(_PORTABILITY_RE.search(t))
        res.complex_cue = bool(_COMPLEX_RE.search(t))
        res.has_question = bool(_QUESTION_RE.search(t))
        qtypes = [qt for qt, rx in _QTYPE_RULES if rx.search(t)]
        res.domains = self._domains(t)
        res.qtype = qtypes[0] if qtypes else QType.NONE
        res.matched += [f"q:{q.value}" for q in qtypes] + [f"d:{d}" for d in res.domains]

        intent = self._resolve(res, qtypes, t, facts, active_scheme_id)
        if intent is not None:
            res.intent = intent
            res.source = "rules"
            anchored = bool(res.scheme_ids or res.domains or res.discovery_cue or res.refers_to_context)
            res.confidence = 0.92 if (qtypes and anchored) else 0.8
            return res

        label, sim, share = self._classify(t, found)
        if label and sim >= C.INTENT_MIN_SIMILARITY and share >= C.INTENT_MIN_VOTE_SHARE:
            res.matched.append(f"knn:{label}:{sim:.2f}:{share:.2f}")
            if label.startswith("I:"):
                res.intent = Intent(label[2:])
            else:
                qt = QType(label[2:])
                res.qtype = qt
                res.intent = self._resolve(res, [qt], t, facts, active_scheme_id) or Intent.OTHER
            res.source = "classifier"
            res.confidence = round(sim * share, 3)
        elif label:
            res.matched.append(f"knn_low:{label}:{sim:.2f}:{share:.2f}")
        return res

    # ── internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _domains(t: str) -> List[str]:
        scored = []
        for order, (domain, rx) in enumerate(_DOMAIN_RULES):
            n = len(rx.findall(t))
            if n:
                scored.append((-n, order, domain))
        return [d for _n, _o, d in sorted(scored)]

    def _resolve(self, res: IntentResult, qtypes: List[QType], t: str, facts: Profile,
                 active_scheme_id: Optional[str]) -> Optional[Intent]:
        q = set(qtypes)
        ctx = bool(res.refers_to_context and active_scheme_id)
        domain = res.domains[0] if res.domains else None
        if QType.STATUS in q:
            return Intent.APPLICATION_STATUS
        if QType.GRIEVANCE in q:
            return Intent.GRIEVANCE
        if QType.DOC_HELP in q and not _SCHEME_CARD_RE.search(t):
            return Intent.DOCUMENT_HELP
        if _STRONG_ELIG_RE.search(t) and QType.DOCUMENTS not in q:
            res.qtype = QType.ELIGIBILITY
            if res.scheme_ids or ctx or (res.scheme_hints and not res.discovery_cue):
                return Intent.ELIGIBILITY_CHECK
            if domain and domain != "MIGRANT":
                return INTENT_OF_DOMAIN[domain]
            res.qtype = QType.DISCOVERY
            return Intent.SCHEME_DISCOVERY
        if QType.DOCUMENTS in q:
            res.qtype = QType.DOCUMENTS
            return Intent.REQUIRED_DOCUMENTS
        if QType.PROCEDURE in q or QType.DOC_HELP in q:
            res.qtype = QType.PROCEDURE
            return Intent.APPLICATION_PROCEDURE
        if QType.FEES in q:
            res.qtype = QType.FEES
            return Intent.GENERAL_SCHEME_INFORMATION
        if QType.CONTACT in q:
            res.qtype = QType.CONTACT
            return Intent.GENERAL_SCHEME_INFORMATION
        if res.discovery_cue and not res.scheme_ids:
            res.qtype = QType.DISCOVERY
            if domain and (domain != "MIGRANT" or res.portability_cue):
                return INTENT_OF_DOMAIN[domain]
            return Intent.SCHEME_DISCOVERY
        if QType.BENEFITS in q:
            if res.scheme_ids or ctx:
                res.qtype = QType.BENEFITS
                return Intent.BENEFIT_INFORMATION
            if domain:
                return INTENT_OF_DOMAIN[domain]
            res.qtype = QType.DISCOVERY
            return Intent.SCHEME_DISCOVERY
        if QType.ELIGIBILITY in q:
            res.qtype = QType.ELIGIBILITY
            if res.scheme_ids:
                return Intent.ELIGIBILITY_CHECK
            if domain:
                return INTENT_OF_DOMAIN[domain]
            if ctx or res.scheme_hints:
                return Intent.ELIGIBILITY_CHECK
            return Intent.SCHEME_DISCOVERY
        if QType.OVERVIEW in q:
            res.qtype = QType.OVERVIEW
            if res.scheme_ids or ctx:
                return Intent.GENERAL_SCHEME_INFORMATION
            if domain:
                return INTENT_OF_DOMAIN[domain]
            if res.scheme_hints or res.unverified_name:
                return Intent.GENERAL_SCHEME_INFORMATION
            return None
        if res.unverified_name:
            return Intent.GENERAL_SCHEME_INFORMATION
        if domain and (res.has_question or bool(facts)):
            return INTENT_OF_DOMAIN[domain]
        if res.scheme_ids:
            res.qtype = QType.OVERVIEW
            return Intent.GENERAL_SCHEME_INFORMATION
        if facts and (res.has_question or len(facts.known_fields()) >= 2):
            res.qtype = QType.DISCOVERY
            return Intent.SCHEME_DISCOVERY
        return None

    def _classify(self, t: str, found) -> Tuple[Optional[str], float, float]:
        if self._protos is None or self._embed is None:
            return None, 0.0, 0.0
        masked = t
        for _sid, _s, alias, _p in found:
            masked = re.sub(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", "this scheme", masked)
        try:
            qv = self._embed(masked)
        except Exception as exc:
            logger.warning(f"[SI] intent embedding failed: {exc}")
            return None, 0.0, 0.0
        if qv is None:
            return None, 0.0, 0.0
        import numpy as np
        sims = self._protos @ qv
        k = min(C.INTENT_KNN_K, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        weights = {}
        best_sim = {}
        for i in top:
            lab = self._labels[i]
            s = float(sims[i])
            weights[lab] = weights.get(lab, 0.0) + s
            best_sim[lab] = max(best_sim.get(lab, 0.0), s)
        label = max(weights, key=weights.get)
        return label, best_sim[label], weights[label] / sum(weights.values())
