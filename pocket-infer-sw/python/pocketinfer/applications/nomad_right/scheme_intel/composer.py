"""
ResponseComposer - short, grounded, TTS-friendly answers.

Every sentence is built from knowledge-base fields (voice summaries,
documents, procedure steps, benefits, contacts, rule outcomes). Wording
follows the eligibility status exactly:
  ELIGIBLE -> "you meet the conditions"   LIKELY -> "you are likely eligible, if ..."
  POTENTIALLY -> "you may be eligible, it depends on ..."
  INSUFFICIENT -> what the scheme is + the one question that decides it
  INELIGIBLE -> "you are not eligible, because ..."
"Definitely eligible" is never said.

Voice text is capped at MAX_VOICE_WORDS (<= 20 s of Hindi/Tamil speech, see
si_constants) and cut only at sentence or clause boundaries. Display lines
follow the legacy 30/60-character limits.
"""

import re
from typing import Dict, List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel.discovery import (
    SLOT_QUESTIONS, is_migrant, question_text,
)
from pocketinfer.applications.nomad_right.scheme_intel.models import (
    CondResult, Intent, Profile, Route, SchemeEvaluation, SIAnswer, Status,
)

_ABBREVIATIONS = [
    (r"\bePoS[- ]enabled\b", "point-of-sale"), (r"\bePoS\b", "point-of-sale machine"),
    (r"\bFPS\b", "Fair Price Shop"), (r"\bNFSA\b", "National Food Security Act"),
    (r"\bCSCs?\b", "Common Service Centre"), (r"\bDBT\b", "direct bank transfer"),
    (r"\bUAN\b", "Universal Account Number"), (r"\bULB\b", "urban local body"),
    (r"\bVLE\b", "Common Service Centre operator"), (r"\bOMCs?\b", "oil company"),
    (r"\bBDO office\b", "Block Development Office"), (r"\bBDO\b", "Block Development Office"),
    (r"\bGRS\b", "Gram Rozgar Sahayak"), (r"\bPMAM\b", "Arogya Mitra"), (r"\bSHG\b", "self-help group"),
    (r"\bKYC\b", "K Y C"), (r"\bBOCW [Bb]oard\b", "construction workers board"),
    (r"\bBOCW\b", "construction workers board"), (r"\bIGNOAPS\b", "old age pension"),
    (r"\bIGNWPS\b", "widow pension"), (r"\bIGNDPS\b", "disability pension"),
    (r"\bOTP\b", "O T P"), (r"\bIFSC\b", "I F S C"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
]
_KEEP_CASE = {"Aadhaar", "PM", "PM-JAY", "Jan", "Dhan", "BPL", "APL", "Ayushman", "RuPay", "UDID", "KYC",
              "IFSC", "LPG", "GSTIN", "e-Shram", "Mera", "SECC", "NFSA", "IGNOAPS", "ESIC", "EPFO", "PAN"}

# How an unconfirmed condition is phrased in "...if <phrase>".
_VERIFY_PHRASES: Dict[str, str] = {
    "aadhaar_seeded_ration": "your Aadhaar is linked to your ration card",
    "nfsa_beneficiary": "your card is under the National Food Security Act",
    "construction_days_12m": "you worked at least 90 days in construction in the last 12 months",
    "has_bank_account": "you have a bank account",
    "vending_since_march_2020": "you were vending on or before 24 March 2020",
    "cov_or_id_card": "you have a vending certificate or a recommendation letter from the town vending committee",
    "aadhaar": "you have an Aadhaar card with a linked mobile number",
    "citizenship": "you are an Indian citizen",
    "lpg_connection": "your family has no gas connection yet",
    "willing_manual_work": "you are willing to do manual work",
    "housing_deprivation": "your family is on the SECC 2011 housing list or the Awaas Plus list",
    "in_awasplus_list": "your family is on the Awaas Plus list",
    "has_electricity_connection": "you have an electricity connection",
    "land_in_family_name": "the land is in your family's name",
    "ignoaps_member": "you do not already get the old age pension",
}
_EXCLUSION_PHRASES: Dict[str, str] = {
    "epfo_esic_member": "you are not covered by PF or ESI",
    "income_tax_payer": "you do not pay income tax",
    "nps_govt_member": "you are not in a government pension scheme",
    "govt_employee_or_family": "no one in your family is a government employee",
    "govt_employee": "you are not a government employee",
    "pmsym_member": "you are not already in PM Shram Yogi Maan-dhan",
    "lpg_connection": "your family has no gas connection yet",
    "housing_deprivation": "you do not already have a pucca house",
}
_EXCLUSION_REASONS: Dict[str, str] = {
    "epfo_esic_member": "it is not open to people covered by PF or ESI",
    "income_tax_payer": "it is not open to income tax payers",
    "nps_govt_member": "it is not open to people in a government pension scheme",
    "govt_employee_or_family": "it is not open to families with a government employee",
    "govt_employee": "it is not open to government employees",
    "pmsym_member": "people already in PM Shram Yogi Maan-dhan cannot join",
    "lpg_connection": "the family must not already have a gas connection",
}
_VERIFY_ORDER = ["construction_days_12m", "housing_deprivation", "aadhaar_seeded_ration", "has_bank_account", "vending_since_march_2020",
                 "cov_or_id_card", "epfo_esic_member", "income_tax_payer", "nfsa_beneficiary", "lpg_connection",
                 "aadhaar", "citizenship"]
_CORE_PHRASES: Dict[str, str] = {
    "age": "your age", "monthly_income": "your monthly income", "ration_card": "whether you have a ration card",
    "is_bpl": "whether your family is below the poverty line", "poor_household": "whether your family is below the poverty line",
    "is_unorganised_worker": "the kind of work you do", "occupation": "the kind of work you do",
    "area_type": "whether you live in a village", "owns_land": "whether you own farm land",
    "in_secc_2011": "whether your family is in the SECC 2011 list", "gender": "whether you are a woman",
    "construction_days_12m": "whether you worked 90 days in construction in the last year",
    "girl_child_under_10": "whether you have a daughter below 10 years", "disability_percentage": "your disability percentage",
    "marital_status": "whether you are a widow", "residence_state": "which state you live in",
    "housing_deprivation": "your housing condition", "trade": "your trade", "small_marginal_farmer": "how much land you have",
    "land_area_ha": "how much land you have", "farmer_type": "whether you farm the land",
}
_DOMAIN_WORDS = {"MATERNITY": "pregnancy and delivery", "HEALTH": "health", "HOUSING": "housing",
                 "EDUCATION": "education and skills", "DISABILITY": "disability", "PENSION": "pension",
                 "FINANCIAL": "loans, savings and insurance", "WORKER": "workers", "RATION": "ration",
                 "MIGRANT": "migrant workers"}


def words(text: str) -> int:
    return len(text.split())


def _money(m: "re.Match") -> str:
    num, unit = m.group(1).replace(",", ""), m.group(2)
    if unit:
        return f" {num} {unit.lower()} rupees "
    try:
        v = float(num)
    except ValueError:
        return f" {num} rupees "
    if v >= 100000:
        return f" {v / 100000:g} lakh rupees "
    return f" {v:g} rupees "


def _tidy(t: str) -> str:
    """Formatting shared by speech and display: no brackets, readable money, no dashes."""
    t = re.sub(r"\([^)]*\)", "", t or "")
    t = t.replace("—", ", ").replace("–", ", ")
    # "Rs"/"INR" only as a standalone word followed by a number - never the
    # "rs" inside "years," or "workers 18".
    t = re.sub(r"(?i)(?<![a-z])(?:rs\.?|inr|₹)\s*(\d[\d,]*(?:\.\d+)?)\s*(lakh|crore)?\b", _money, t)
    t = re.sub(r"(\d),(?=\d)", r"\1", t)
    t = re.sub(r"\s*/\s*(month|day|year|week)\b", r" a \1", t)
    t = re.sub(r"\s*/\s*kW\b", " per kilowatt", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\d])", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()


class ResponseComposer:
    def __init__(self, repo, max_words: int = C.MAX_VOICE_WORDS):
        self.repo = repo
        self.max_words = max_words

    # ── text utilities ────────────────────────────────────────────────────
    @staticmethod
    def speakable(text: str) -> str:
        if not text:
            return ""
        t = _tidy(text)
        for pat, rep in _ABBREVIATIONS:
            t = re.sub(pat, rep, t)
        t = t.replace("&", " and ").replace("%", " percent").replace(" / ", " or ").replace("->", " then ")
        t = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " or ", t)
        t = re.sub(r"\betc\.?", "", t)
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def displayable(text: str) -> str:
        return _tidy(text).replace("&", "and")

    @staticmethod
    def first_clause(text: str, limit: int = 14) -> str:
        """The first sentence, shortened at a natural break if it is longer than `limit` words."""
        t = (text or "").strip()
        if not t:
            return ""
        first = re.split(r"(?<=[.!?])\s+", t)[0]
        w = first.split()
        if len(w) <= limit:
            return first.rstrip(" ,;:") + ("" if first.endswith((".", "!", "?")) else ".")
        head = " ".join(w[:limit])
        for sep in (" with ", " by ", " using ", " through ", " so that ", " for ", "; ", ", "):
            idx = head.rfind(sep)
            if idx > 0 and len(head[:idx].split()) >= 5:
                return head[:idx].rstrip(" ,;:") + "."
        return head.rstrip(" ,;:") + "."

    def clip(self, text: str, max_words: Optional[int] = None) -> str:
        limit = max_words or self.max_words
        text = re.sub(r"\s+", " ", text or "").strip()
        if words(text) <= limit:
            return text
        out: List[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if words(" ".join(out + [sent])) <= limit:
                out.append(sent)
            else:
                break
        return " ".join(out) if out else self.first_clause(text, limit)

    @staticmethod
    def join(items: List[str], conj: str = "and") -> str:
        items = [i for i in items if i]
        if len(items) <= 1:
            return items[0] if items else ""
        return ", ".join(items[:-1]) + f" {conj} " + items[-1]

    @staticmethod
    def doc_name(name: str) -> str:
        n = re.sub(r"\([^)]*\)", "", name or "")
        n = n.replace(" / ", " or ").replace("/", " or ")
        out = []
        for w in n.split():
            core = w.strip(",.")
            keep = core in _KEEP_CASE or core.startswith("Aadhaar") or (len(core) >= 2 and core.isupper())
            out.append(w if keep else w.lower())
        return " ".join(out)

    def _answer(self, voice: str, top: str, bottom: str, route: Route, intent: Intent, status: str = "",
                sid: Optional[str] = None, sids: Optional[List[str]] = None, asked: Optional[str] = None,
                severity: str = "INFO") -> SIAnswer:
        voice = self.clip(self.speakable(voice))
        return SIAnswer(voice=voice, top=top[:C.DISPLAY_TOP_MAX], bottom=self.displayable(bottom)[:C.DISPLAY_BOTTOM_MAX],
                        route=route, intent=intent, status=status, scheme_id=sid,
                        legacy_code=self.repo.legacy_code(sid), scheme_ids=sids or ([sid] if sid else []),
                        severity=severity, asked_slot=asked)

    def _spoken(self, sid: str, d: Optional[Profile] = None) -> str:
        name = self.repo.spoken(sid)
        if sid == "SCH_BOCW" and d is not None and d.get("current_state"):
            name = f"{name} in {d.get('current_state')}"
        return name

    def _contacts(self, sid: str, kind: str) -> List[str]:
        out: List[str] = []
        for c in self.repo.contacts(sid):
            if c["kind"] != kind:
                continue
            parts = [p.strip() for p in re.split(r"\s*/\s*", c["value"])] if kind == "helpline" else [c["value"]]
            for p in parts:
                if p and p not in out:
                    out.append(p)
        return out

    def _contact(self, sid: str, kind: str) -> Optional[str]:
        vals = self._contacts(sid, kind)
        return vals[0] if vals else None

    @staticmethod
    def _site(url: str) -> str:
        return re.sub(r"^https?://(?:www\.)?", "", url or "").split("/")[0]

    def _summary_sentence(self, sid: str) -> str:
        s = self.repo.scheme(sid)
        return self.first_clause(s.get("voice_summary") or s.get("description") or "", 30)

    def _benefit_sentence(self, sid: str) -> str:
        for b in self.repo.benefits(sid):
            if b.get("value_inr"):
                text = self.first_clause(self.speakable(b["text"]), 14)
                first = text.split()[0] if text else ""
                if first and not (first in _KEEP_CASE or first.isupper() or first[0].isdigit()):
                    text = first.lower() + text[len(first):]
                return f"It gives {text}" if text else ""
        return ""

    def _check_line(self, sid: str) -> str:
        portal, helpline = self._contact(sid, "portal"), self._contact(sid, "helpline")
        if portal and helpline:
            return f"To check, visit {self._site(portal)} or call {helpline}."
        if portal:
            return f"To check, visit {self._site(portal)}."
        if helpline:
            return f"To check, call {helpline}."
        return ""

    def _confirm_line(self, sid: Optional[str]) -> str:
        """Where to confirm an answer the knowledge base did not word: a helpline if
        there is one (shortest to say), else the scheme's website, else a plain
        reminder (15 schemes have no contact in the knowledge base - none is invented)."""
        if not sid:
            return ""
        s = self.repo.scheme(sid)
        helpline = self._contact(sid, "helpline") or re.split(r"\s*/\s*", s.get("helpline") or "")[0]
        if helpline:
            return f"To confirm, call {helpline}."
        site = self._contact(sid, "portal") or s.get("official_url")
        return f"To confirm, visit {self._site(site)}." if site else "Please confirm this before you apply."

    def _next_step(self, sid: str, d: Optional[Profile] = None) -> str:
        if sid == "SCH_BOCW" and d is not None and d.get("current_state"):
            for c in self.repo.contacts(sid):
                if c["kind"] == "state_portal" and c.get("description") == d.get("current_state"):
                    return f"Register with the {d.get('current_state')} board at {c['value']}."
        proc = self.repo.procedure(sid)
        steps = proc["offline"] or proc["online"]
        if steps:
            return "To apply: " + self.first_clause(self.speakable(steps[0]), 16)
        helpline = self._contact(sid, "helpline")
        return f"For help, call {helpline}." if helpline else ""

    def _fit(self, parts: List[str], optional_idx: List[int]) -> str:
        """Joins parts, dropping the optional ones (last first) until the answer fits the budget."""
        parts = list(parts)
        for i in sorted(optional_idx, reverse=True):
            if words(" ".join(p for p in parts if p)) <= self.max_words:
                break
            if i < len(parts):
                parts[i] = ""
        return " ".join(p for p in parts if p)

    # ── DIRECT_INFORMATION ────────────────────────────────────────────────
    def overview(self, sid: str, intent: Intent) -> SIAnswer:
        s = self.repo.scheme(sid)
        voice = s.get("voice_summary") or s.get("description") or ""
        return self._answer(voice, f"{self.repo.short(sid)}: ABOUT", voice, Route.DIRECT_INFORMATION, intent,
                            "OVERVIEW", sid)

    def documents(self, sid: str, intent: Intent) -> Optional[SIAnswer]:
        mandatory, optional = self.repo.documents(sid)
        if not mandatory and not optional:
            return None
        names = [self.doc_name(d["name"]) for d in (mandatory or optional)][:6]
        voice = f"For {self.repo.spoken(sid)}, you need {self.join(names)}."
        extra = [self.doc_name(d["name"]) for d in optional][:2] if mandatory else []
        if extra and words(voice) + 6 + sum(words(e) for e in extra) <= self.max_words:
            voice += f" If you have it, also bring {self.join(extra)}."
        bottom = ", ".join(self.doc_name(d["name"]) for d in (mandatory or optional))
        return self._answer(voice, f"{self.repo.short(sid)}: DOCUMENTS", bottom, Route.DIRECT_INFORMATION,
                            intent, "DOCUMENTS", sid)

    def procedure(self, sid: str, intent: Intent, d: Profile) -> Optional[SIAnswer]:
        proc = self.repo.procedure(sid)
        steps = proc["offline"] or proc["online"]
        if not steps:
            return None
        parts = [f"To apply for {self._spoken(sid, d)}:"]
        if sid == "SCH_BOCW" and d.get("current_state"):
            step = self._next_step(sid, d)
            if step.startswith("Register"):
                parts.append(step)
        for st in steps[:4]:
            sentence = self.first_clause(self.speakable(st), 18)
            if words(" ".join(parts + [sentence])) > self.max_words - 6:
                break
            parts.append(sentence)
        if proc["offline"] and proc["online"] and words(" ".join(parts)) <= self.max_words - 6:
            parts.append("You can also apply online.")
        return self._answer(" ".join(parts), f"{self.repo.short(sid)}: HOW TO APPLY",
                            self.first_clause(steps[0], 10), Route.DIRECT_INFORMATION, intent, "PROCEDURE", sid)

    def benefits(self, sid: str, intent: Intent) -> Optional[SIAnswer]:
        rows = self.repo.benefits(sid)
        if not rows:
            return None
        quantified = [b for b in rows if b.get("value_inr")]
        chosen = (quantified + [b for b in rows if not b.get("value_inr")])[:3]
        parts = [f"Under {self.repo.spoken(sid)}:"]
        for b in chosen:
            sentence = self.first_clause(self.speakable(b["text"]), 18)
            if words(" ".join(parts + [sentence])) > self.max_words:
                break
            parts.append(sentence)
        if len(parts) == 1:
            parts.append(self.first_clause(self.speakable(chosen[0]["text"]), self.max_words - 6))
        return self._answer(" ".join(parts), f"{self.repo.short(sid)}: BENEFITS", chosen[0]["text"],
                            Route.DIRECT_INFORMATION, intent, "BENEFITS", sid)

    def contact(self, sid: str, intent: Intent, status: str = "CONTACT") -> Optional[SIAnswer]:
        helplines = self._contacts(sid, "helpline")[:2]
        portal = self._contact(sid, "portal")
        if not helplines and not portal:
            return None
        spoken = self.repo.spoken(sid)
        lead = f"To check your {spoken} application," if status == "STATUS" else f"For {spoken},"
        bits = []
        if helplines:
            bits.append(f"call {self.join(helplines, 'or')}")
        if portal:
            bits.append(f"visit {self._site(portal)}")
        voice = f"{lead} {' or '.join(bits)}."
        return self._answer(voice, f"{self.repo.short(sid)}: {'STATUS' if status == 'STATUS' else 'HELPLINE'}",
                            " | ".join(helplines + ([self._site(portal)] if portal else [])),
                            Route.DIRECT_INFORMATION, intent, status, sid)

    def authority(self, sid: str, intent: Intent) -> Optional[SIAnswer]:
        """Which ministry / agency runs the scheme (canonical ministry, department, implementing_agency)."""
        s = self.repo.scheme(sid)
        ministry, dept, agency = s.get("ministry"), s.get("department"), s.get("implementing_agency")
        if not (ministry or agency):
            return None
        parts = []
        if ministry:
            parts.append(f"{self.repo.spoken(sid)} is run by the {ministry}" + (f", {dept}" if dept else "") + ".")
        if agency:
            parts.append(f"It is implemented by {agency}." if ministry else
                         f"{self.repo.spoken(sid)} is implemented by {agency}.")
        return self._answer(" ".join(parts), f"{self.repo.short(sid)}: WHO RUNS IT", ministry or agency,
                            Route.DIRECT_INFORMATION, intent, "AUTHORITY", sid)

    def fees(self, sid: str, intent: Intent) -> Optional[SIAnswer]:
        text = self.repo.procedure(sid).get("fees")
        if not text:
            for f in self.repo.faqs(sid):
                if re.search(r"\b(?:fee|fees|free|charge|premium|pay)\b", f["question"].lower()):
                    text = f["answer"]
                    break
        if not text:
            return None
        voice = f"For {self.repo.spoken(sid)}: {self.first_clause(self.speakable(text), 30)}"
        return self._answer(voice, f"{self.repo.short(sid)}: FEES", text, Route.DIRECT_INFORMATION, intent, "FEES", sid)

    def grievance(self, sid: Optional[str], intent: Intent) -> Optional[SIAnswer]:
        if sid:
            griev = self._contacts(sid, "grievance")
            offices = [g for g in griev if not re.search(r"https?://|\.(?:gov|nic)\.in", g)]
            urls = [self._site(g) for g in griev if g not in offices]
            helplines = self._contacts(sid, "helpline")[:2]
            where = offices[0] if offices else (urls[0] if urls else None)
            if where or helplines:
                voice = f"To complain about {self.repo.spoken(sid)}, "
                if where:
                    voice += f"contact {where}"
                if helplines:
                    voice += (", or " if where else "") + f"call {self.join(helplines, 'or')}"
                voice += "."
                if urls and offices and words(voice) + 8 <= self.max_words:
                    voice += f" You can also complain online at {urls[0]}."
                return self._answer(voice, f"{self.repo.short(sid)}: COMPLAINT", where or " / ".join(helplines),
                                    Route.DIRECT_INFORMATION, intent, "GRIEVANCE", sid)
        # Central grievance portal (CPGRAMS) - listed in pds.json / mgnregs.json official_contacts.
        for code in ("SCH_ONORC", "SCH_MGNREGA"):
            if self.repo.has(code) and any("pgportal" in g for g in self._contacts(code, "grievance")):
                voice = ("You can file a complaint on the central government grievance portal, "
                         "pgportal.gov.in. Tell me the scheme name for its own helpline.")
                return self._answer(voice, "COMPLAINT", "pgportal.gov.in (CPGRAMS)", Route.DIRECT_INFORMATION,
                                    intent, "GRIEVANCE")
        return None

    def migrant_overview(self, d: Profile, intent: Intent) -> SIAnswer:
        dest = d.get("current_state")
        where = f"in {dest}" if dest else "in any state"
        voice = (f"Your ration card works {where} under One Nation One Ration Card. "
                 "Ayushman Bharat hospital cover and the e-Shram card are valid across India. "
                 "Construction workers must register with the board of the state where they work.")
        return self._answer(voice, "MIGRANT: WHAT MOVES WITH YOU", "Ration card, PM-JAY, e-Shram work in all states",
                            Route.DIRECT_INFORMATION, intent, "PORTABILITY",
                            sids=["SCH_ONORC", "SCH_PMJAY", "SCH_ESHRAM", "SCH_BOCW"])

    def aadhaar_seeding_help(self, intent: Intent) -> SIAnswer:
        # pds.json application_process.offline[0] and online[1].
        voice = ("To link Aadhaar with your ration card, visit the local Food and Civil Supplies office "
                 "or designated centre. You can check the linking status in the Mera Ration app.")
        return self._answer(voice, "AADHAAR - RATION CARD", "Food and Civil Supplies office / Mera Ration app",
                            Route.DIRECT_INFORMATION, intent, "DOC_HELP", "SCH_ONORC")

    def unverified(self, name: str, intent: Intent) -> SIAnswer:
        voice = (f"I do not have verified information about {name} yet. "
                 "I can help with ration, health, pension, worker and housing schemes.")
        return self._answer(voice, "NOT IN VERIFIED DATA", f"No verified data for {name}",
                            Route.CLARIFY, intent, "UNVERIFIED", severity="WARNING")

    def clarify_scheme(self, options: List[str], intent: Intent) -> SIAnswer:
        opts = [self.repo.spoken(s) for s in options[:3] if self.repo.has(s)]
        voice = f"Which scheme do you mean: {self.join(opts, 'or')}?" if opts else "Which scheme do you want to know about?"
        return self._answer(voice, "WHICH SCHEME?", ", ".join(self.repo.short(s) for s in options[:3]) or "Name the scheme",
                            Route.CLARIFY, intent, "CLARIFY", asked="scheme")

    def from_chunk(self, sid: Optional[str], text: str, intent: Intent, route: Route, status: str) -> SIAnswer:
        # FAQ chunks are "Q: ... A: ..." or "FAQ for X: Question: ... Answer: ..." - read only the answer.
        body = re.sub(r"^.*?\b(?:Q|Question):.*?\b(?:A|Answer):\s*", "", text, flags=re.S)
        voice = self.clip(self.speakable(body))
        top = f"{self.repo.short(sid)}: INFO" if sid else "INFO"
        return self._answer(voice, top, voice, route, intent, status, sid)

    # ── ELIGIBILITY ───────────────────────────────────────────────────────
    def _first_open(self, ev: SchemeEvaluation) -> Tuple[Optional[str], Optional[str]]:
        items: List[Tuple[int, str, str]] = []
        for c in ev.unknown_verify:
            for f in c.field.split("|"):
                if f in _VERIFY_PHRASES:
                    items.append((_VERIFY_ORDER.index(f) if f in _VERIFY_ORDER else 50, f, _VERIFY_PHRASES[f]))
                    break
        for c in ev.exclusions_unknown:
            for f in c.field.split("|"):
                if f in _EXCLUSION_PHRASES:
                    items.append((_VERIFY_ORDER.index(f) if f in _VERIFY_ORDER else 60, f, _EXCLUSION_PHRASES[f]))
                    break
        if not items:
            return None, None
        items.sort()
        return items[0][1], items[0][2]

    def _core_phrase(self, conds: List[CondResult]) -> Tuple[Optional[str], Optional[str]]:
        for c in conds:
            for f in c.field.split("|"):
                if f in _CORE_PHRASES:
                    return f, _CORE_PHRASES[f]
        if conds:
            return conds[0].field.split("|")[0], self.first_clause(conds[0].desc.lower(), 10).rstrip(".")
        return None, None

    def _reason(self, sid: str, ev: SchemeEvaluation) -> str:
        s = self.repo.scheme(sid)
        if ev.exclusions_hit:
            c = ev.exclusions_hit[0]
            for f in c.field.split("|"):
                if f in _EXCLUSION_REASONS:
                    return _EXCLUSION_REASONS[f]
            return self.first_clause(self.speakable(c.desc).lower(), 12).rstrip(".")
        c = ev.failed[0]
        if c.field == "age" and (s.get("age_min") or s.get("age_max")):
            if s.get("age_min") and s.get("age_max"):
                return f"it is for people aged {s['age_min']} to {s['age_max']}"
            return f"the minimum age is {s['age_min']}" if s.get("age_min") else f"the maximum age is {s['age_max']}"
        if c.field == "monthly_income" and s.get("income_limit"):
            return f"it is for people earning up to {s['income_limit']} rupees a month"
        if c.field == "area_type":
            return "it is only for people living in rural areas"
        if c.field == "residence_state":
            return f"it is only for residents of {c.expected}"
        if c.field in _VERIFY_PHRASES:
            return "it requires that " + _VERIFY_PHRASES[c.field]
        return self.first_clause(self.speakable(c.desc).lower(), 12).rstrip(".")

    def eligibility(self, ev: SchemeEvaluation, d: Profile, intent: Intent, question_slot: Optional[str]) -> SIAnswer:
        sid = ev.scheme_id
        spoken = self._spoken(sid, d)
        short = self.repo.short(sid)
        status = ev.status
        asked = question_slot if question_slot in SLOT_QUESTIONS else None
        question = question_text(asked, d) if asked else ""
        if "ENDED" in ev.notes:
            succ = next((n.split(":", 1)[1] for n in ev.notes if n.startswith("REPLACED_BY:")), None)
            voice = f"{self.repo.spoken(sid)} has ended."
            if succ and self.repo.has(succ):
                voice += f" It was replaced by {self.repo.spoken(succ)}. {self._summary_sentence(succ)}"
            return self._answer(voice, f"{short}: ENDED", f"Replaced by {self.repo.short(succ)}" if succ else "Scheme ended",
                                Route.ELIGIBILITY, intent, "ENDED", sid, severity="WARNING")
        if status == Status.ELIGIBLE:
            step = self._next_step(sid, d)
            voice = self._fit([f"Yes. Based on what you told me, you meet the conditions for {spoken}.",
                               self._benefit_sentence(sid), step], optional_idx=[1])
            bottom = "Meets the conditions. " + step
        elif status == Status.LIKELY_ELIGIBLE:
            _f, phrase = self._first_open(ev)
            lead = f"You are likely eligible for {spoken}" + (f", if {phrase}." if phrase else ".")
            voice = self._fit([lead, self._benefit_sentence(sid), self._next_step(sid, d)], optional_idx=[1])
            bottom = "Likely eligible" + (f" if {phrase}" if phrase else "")
        elif status == Status.POTENTIALLY_ELIGIBLE:
            _f, phrase = self._core_phrase(ev.unknown_core)
            voice = f"You may be eligible for {spoken}." + (f" It depends on {phrase}." if phrase else "")
            voice += " " + (question or self._check_line(sid))
            bottom = f"May be eligible, depends on {phrase}" if phrase else "May be eligible"
        elif status == Status.INSUFFICIENT_INFORMATION:
            _f, phrase = self._core_phrase(ev.unknown_core)
            voice = f"{self._summary_sentence(sid)} {question or self._check_line(sid)}"
            bottom = f"Need: {phrase}" if phrase else "Need more details"
        elif status == Status.INELIGIBLE:
            reason = self._reason(sid, ev)
            voice = f"Based on what you told me, you are not eligible for {spoken}, because {reason}."
            bottom = f"Not eligible: {reason}"
        else:
            voice = f"I do not have verified eligibility rules for {spoken}. {self._check_line(sid)}"
            bottom = "Eligibility rules not in verified data"
        return self._answer(voice, f"{short}: {status.value.replace('_', ' ')}", bottom, Route.ELIGIBILITY,
                            intent, status.value, sid, asked=asked if asked and question in voice else None,
                            severity="WARNING" if status in (Status.UNKNOWN, Status.INSUFFICIENT_INFORMATION) else "INFO")

    def ration_access(self, ev: SchemeEvaluation, d: Profile, intent: Intent) -> SIAnswer:
        """One Nation One Ration Card, phrased for someone asking about getting ration where they are."""
        card_state = d.get("ration_card_state") or (d.get("origin_state") if is_migrant(d) else None)
        here = d.get("current_state")
        card = f"your {card_state} ration card" if card_state else "your ration card"
        where = f"any Fair Price Shop in {here}" if here else "any Fair Price Shop in India"
        seeded = d.get("aadhaar_seeded_ration")
        if d.get("ration_card") is False:
            voice = ("One Nation One Ration Card works only with an existing ration card under the National Food "
                     "Security Act. For a new ration card, please contact your state Food and Civil Supplies Department.")
            return self._answer(voice, "ONORC: NEEDS RATION CARD", "Needs an NFSA ration card", Route.ELIGIBILITY,
                                intent, Status.INELIGIBLE.value, "SCH_ONORC")
        if seeded is False:
            voice = ("Not yet. One Nation One Ration Card works only when your Aadhaar is linked to your ration card. "
                     "Get it linked at the local Food and Civil Supplies office or designated centre.")
            return self._answer(voice, "ONORC: LINK AADHAAR FIRST", "Link Aadhaar to ration card first",
                                Route.ELIGIBILITY, intent, Status.INELIGIBLE.value, "SCH_ONORC")
        if d.get("ration_card") is True and ev.status in (Status.ELIGIBLE, Status.LIKELY_ELIGIBLE):
            lead = "Yes." if ev.status == Status.ELIGIBLE else "Yes, most likely."
            voice = f"{lead} Under One Nation One Ration Card, you can use {card} at {where}."
            if seeded is not True:
                voice += " Your Aadhaar must be linked to the card."
            voice += " Give your fingerprint at the shop."
            return self._answer(voice, f"ONORC: {ev.status.value.replace('_', ' ')}",
                                f"Use {card.replace('your ', '')} at {where}", Route.ELIGIBILITY, intent,
                                ev.status.value, "SCH_ONORC")
        if not here and not is_migrant(d):
            # pds.json eligibility[0], [2]: NFSA ration cardholders in the AAY or PHH category.
            voice = ("Ration food grains are for families with a National Food Security Act ration card, in the "
                     f"Antyodaya or Priority Household category. {question_text('ration_card', d)}")
            return self._answer(voice, "RATION: WHO GETS IT", "NFSA ration card, AAY or PHH category",
                                Route.ELIGIBILITY, intent, ev.status.value, "SCH_ONORC", asked="ration_card")
        voice = (f"If you have a ration card under the National Food Security Act, One Nation One Ration Card "
                 f"lets you collect your ration at {where}. {question_text('ration_card', d)}")
        return self._answer(voice, "ONORC: NEED RATION CARD INFO", "Do you have an NFSA ration card?",
                            Route.ELIGIBILITY, intent, ev.status.value, "SCH_ONORC", asked="ration_card")

    def portability(self, sid: str, d: Profile, intent: Intent) -> Optional[SIAnswer]:
        """Whether a scheme card works where the person is (pmjay/eshram/bocw portability fields)."""
        here = d.get("current_state")
        if sid == "SCH_PMJAY":
            where = f"any empanelled hospital in {here}" if here else "any empanelled hospital in India"
            voice = (f"Yes. Ayushman Bharat PM-JAY works across India. You can get cashless treatment at {where}. "
                     "Carry your Ayushman card and Aadhaar.")
        elif sid == "SCH_ESHRAM":
            voice = "Yes. Your e-Shram card and Universal Account Number are valid in every state."
        elif sid == "SCH_BOCW":
            board = f"the construction workers board of {here}" if here else "the construction workers board of the state"
            voice = (f"Register with {board} where you are working now, once you have 90 days of construction work there. "
                     "A card from another state does not move automatically.")
            step = self._next_step(sid, d)
            if step.startswith("Register with the") and words(voice) + words(step) <= self.max_words:
                voice = f"{voice} {step}"
        else:
            return None
        return self._answer(voice, f"{self.repo.short(sid)}: ANY STATE", self.first_clause(voice, 10),
                            Route.ELIGIBILITY, intent, "PORTABILITY", sid)

    # ── DISCOVERY ─────────────────────────────────────────────────────────
    def discovery(self, top: List[SchemeEvaluation], d: Profile, intent: Intent, slot: Optional[str],
                  domain: Optional[str] = None, domain_cands: Optional[List[str]] = None) -> SIAnswer:
        q = question_text(slot, d) if slot else ""
        if not top:
            where = f" in {d.get('current_state')}" if d.get("current_state") else ""
            if domain and domain_cands:
                names = [self._spoken(s, d) for s in domain_cands[:2]]
                voice = f"For {_DOMAIN_WORDS.get(domain, domain.lower())}, my verified schemes include {self.join(names)}. {q}"
                return self._answer(voice, f"{domain}: SCHEMES", ", ".join(self.repo.short(s) for s in domain_cands[:2]),
                                    Route.SCHEME_DISCOVERY, intent, "NEED_INFO", domain_cands[0], domain_cands[:2],
                                    asked=slot)
            if slot == "occupation":
                voice = (f"I can help you find schemes{where}. Many schemes depend on your work, "
                         "for example construction, domestic work, farming or vending. What work do you do?")
            else:
                voice = f"I can help you find schemes{where}. {q}".strip()
            return self._answer(voice, "FIND SCHEMES", q or "Tell me about your work", Route.SCHEME_DISCOVERY,
                                intent, "NEED_INFO", asked=slot)
        if len(top) == 1:
            ans = self.eligibility(top[0], d, intent, slot)
            ans.route = Route.SCHEME_DISCOVERY
            return ans
        sure = [ev for ev in top if ev.status in (Status.ELIGIBLE, Status.LIKELY_ELIGIBLE)]
        maybe = [ev for ev in top if ev.status == Status.POTENTIALLY_ELIGIBLE]
        occ = d.get("occupation")
        occ_lead = (f"As a {occ.replace('_', ' ').lower()}, you may get help from"
                    if occ and occ not in ("UNEMPLOYED", "STUDENT", "RETIRED", "HOMEMAKER", "ARTISAN") else "You may be eligible for")
        voice, shown = "", []
        for n_sure, n_maybe in ((len(sure), len(maybe)), (len(sure), 0), (min(2, len(sure)), 0), (1, 0)):
            parts, shown = [], []
            if sure[:n_sure]:
                parts.append(f"You are likely eligible for {self.join([self._spoken(e.scheme_id, d) for e in sure[:n_sure]])}.")
                shown += sure[:n_sure]
            if maybe[:n_maybe]:
                lead = "You may also be eligible for" if parts else occ_lead
                parts.append(f"{lead} {self.join([self._spoken(e.scheme_id, d) for e in maybe[:n_maybe]])}.")
                shown += maybe[:n_maybe]
            if not parts:
                parts.append(f"{occ_lead} {self._spoken(top[0].scheme_id, d)}.")
                shown = top[:1]
            voice = " ".join(parts + ([q] if q else []))
            if words(voice) <= self.max_words:
                break
        sids = [ev.scheme_id for ev in shown]
        top_line = f"{len(sids)} SCHEME{'S' if len(sids) > 1 else ''} FOR YOU"
        bottom = ", ".join(self.repo.short(s) for s in sids) + (f" | {q}" if q else "")
        return self._answer(voice, top_line, bottom, Route.SCHEME_DISCOVERY, intent, shown[0].status.value,
                            sids[0], sids, asked=slot if q and q in voice else None)

    def llm(self, text: str, sid: Optional[str], sids: List[str], intent: Intent) -> SIAnswer:
        # Qwen's wording is not a knowledge-base field (measured: it can add a
        # clause no passage supports), so it always ends with where to confirm.
        confirm = self._confirm_line(sid)
        body = self.clip(self.speakable(text), self.max_words - words(confirm))
        ans = self._answer(f"{body} {confirm}".strip(), f"{self.repo.short(sid)}: INFO" if sid else "SCHEME INFO",
                           text, Route.COMPLEX_SCENARIO, intent, "LLM_GROUNDED", sid, sids, severity="WARNING")
        ans.used_llm = True
        return ans
