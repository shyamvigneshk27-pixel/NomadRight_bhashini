"""
ScenarioEngine - deterministic extraction of the user's situation from one
English utterance (post-NMT).

Only facts the person actually states are recorded. Nothing is assumed:
"I am from Bihar" sets origin_state only; current_state stays unknown until
they say where they are. Every fact carries the phrase it came from
(Profile.evidence) so a wrong extraction is easy to diagnose.

Spec scenario fields (models.SCENARIO_FIELDS) plus supplementary fields the
rules need:
  trade, is_unorganised_worker, has_bank_account, income_tax_payer,
  epfo_esic_member, area_type, is_bpl, ration_card_type, ration_card_state,
  ration_card_lost, aadhaar_seeded_ration, disability_percentage,
  household_disability, citizenship, construction_days_12m, owns_land,
  land_area_ha, farmer_type, govt_employee, lpg_connection, housing_status,
  monthly_pension, daily_wage, family_monthly_income, family_annual_income,
  children_count, has_daughter, girl_child_ages, has_electricity_connection,
  mentioned_state.
"""

import re
from typing import List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel import gazetteer
from pocketinfer.applications.nomad_right.scheme_intel.models import Profile

# ─────────────────────────────────────────────────────────────────────────
# Normalisation (numbers are the main thing NMT output varies on)
# ─────────────────────────────────────────────────────────────────────────

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000,
           "lacs": 100000, "crore": 10000000, "crores": 10000000}
_NUMWORD_RE = re.compile(
    r"\b(?:(?:" + "|".join(list(_UNITS) + list(_TENS) + list(_SCALES)) + r")(?:[\s-]+(?:and[\s-]+)?|\b))+"
)


def _parse_number_words(words: List[str]) -> Optional[int]:
    total, current, seen = 0, 0, False
    for w in words:
        if w == "and":
            continue
        if w in _UNITS:
            current += _UNITS[w]
            seen = True
        elif w in _TENS:
            current += _TENS[w]
            seen = True
        elif w in _SCALES:
            scale = _SCALES[w]
            current = max(current, 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def _words_to_digits(text: str) -> str:
    def repl(m: "re.Match") -> str:
        chunk = m.group(0)
        words = [w for w in re.split(r"[\s-]+", chunk.strip()) if w]
        # "one" alone is too often an article-like word ("the one scheme") -
        # only convert multi-word numbers or clearly numeric single words.
        if len(words) == 1 and words[0] in ("one", "hundred", "lakh", "lac", "crore"):
            return chunk
        value = _parse_number_words(words)
        if value is None:
            return chunk
        trailing = " " if chunk.endswith((" ", "-")) else ""
        return f"{value}{trailing}"
    return _NUMWORD_RE.sub(repl, text)


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("₹", " rs ")
    t = re.sub(r"\brs\.", "rs ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _words_to_digits(t)
    # 12,000 / 1,20,000 -> 12000 / 120000
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"(\d),(\d{2,3})\b", r"\1\2", t)
    # 12k -> 12000, 1.5 lakh -> 150000, 15 thousand -> 15000
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?k\b", lambda m: str(int(float(m.group(1)) * 1000)), t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?(?:thousand)\b", lambda m: str(int(float(m.group(1)) * 1000)), t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?(?:lakhs?|lacs?)\b",
               lambda m: str(int(float(m.group(1)) * 100000)) + " lakhmark", t)
    t = t.replace("e shram", "e-shram").replace("eshram", "e-shram").replace("e-sharm", "e-shram")
    t = re.sub(r"\baadhar\b", "aadhaar", t)
    return t


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

_NEG_RE = re.compile(
    r"\b(?:not|no|never|don't|dont|do not|doesn't|does not|didn't|did not|haven't|have not|"
    r"hasn't|has not|without|nor|neither|nahi|nahin|none)\b"
)
_SELF_RE = re.compile(r"\b(?:i|i'm|im|me|my|myself|we|we're|our|us|mine)\b")
_SENT_SPLIT_RE = re.compile(r"[.!?;]")


def _negated(t: str, start: int, window: int = 28) -> bool:
    """A negation word shortly before `start`, within the same clause."""
    seg = t[max(0, start - window):start]
    seg = re.split(r"[.,;!?]| but | and ", seg)[-1]
    return bool(_NEG_RE.search(seg))


def _clause_before(t: str, start: int) -> str:
    return _SENT_SPLIT_RE.split(t[:start])[-1]


def _self_stated(t: str, start: int) -> bool:
    return bool(_SELF_RE.search(_clause_before(t, start)))


def _snip(t: str, m: "re.Match") -> str:
    return t[m.start():m.end()]


# ─────────────────────────────────────────────────────────────────────────
# Lexicons
# ─────────────────────────────────────────────────────────────────────────

# Order matters: the first (most specific) match inside a self-statement wins.
_OCCUPATIONS: List[Tuple[str, str]] = [
    ("CONSTRUCTION_WORKER",
     r"construction (?:site )?(?:worker|labou?rer|labou?r|work|helper|site|line|field)|building (?:construction )?worker"
     r"|work(?:ing|s)? (?:in|at|on) (?:a |the )?(?:construction|building site)|do construction"
     r"|mason|raj ?mistri|brick ?layer|bar bender|shuttering|centering work|scaffold"),
    ("DOMESTIC_WORKER",
     r"domestic (?:worker|help|work)|house ?maid|\bmaid\b|house help|household help"
     r"|work(?:ing|s)? (?:in|at) (?:other )?(?:people'?s )?(?:houses|homes)|cook in (?:houses|homes)"),
    ("STREET_VENDOR",
     r"street vendor|\bvendor\b|hawker|thela|rehri|pheri ?wala|sabzi ?wala|vegetable (?:seller|vendor)"
     r"|fruit (?:seller|vendor)|sell(?:ing|s)? (?:\w+ ){0,3}(?:on|at|by) (?:the )?(?:street|road|roadside|footpath|cart)"
     r"|tea stall|push ?cart"),
    ("GIG_WORKER",
     r"delivery (?:boy|partner|executive|agent|job|work|rider)|gig worker|platform worker|swiggy|zomato"
     r"|zepto|blinkit|dunzo|rapido"),
    ("TRANSPORT_WORKER",
     r"(?:auto|rickshaw|e-rickshaw|taxi|cab|truck|lorry|bus|tempo|tractor) ?(?:driver|puller|wala)"
     r"|rickshaw puller|\bdriver\b|\bconductor\b"
     # "I drive an auto", "driving a taxi in Mumbai"
     r"|\b(?:drive|drives|driving) (?:an? |the |my |our )?(?:auto|rickshaw|e-rickshaw|taxi|cab|truck|lorry|bus|tempo|tractor|ola|uber)\b"),
    ("HEAD_LOADER", r"head ?loader|\bporter\b|\bcoolie\b|\bhamal\b|loading (?:and )?unloading|\bloader\b"),
    ("BRICK_KILN_WORKER", r"brick kiln|\bbhatta\b"),
    ("RAG_PICKER", r"rag ?picker|waste picker|kabadi|scrap (?:collector|picker)"),
    ("SANITATION_WORKER", r"manual scavenger|sanitation worker|safai karmachari|\bsweeper\b|cleaning worker"),
    ("WEAVER", r"handloom weaver|power ?loom|handloom|\bweaver\b|weaving"),
    ("FISHERMAN", r"fisher ?man|fishermen|fish ?worker|\bfishing\b"),
    ("BEEDI_WORKER", r"\bbeedi\b|bidi roll"),
    ("LEATHER_WORKER", r"leather (?:worker|work)|tannery"),
    ("AGRICULTURAL_LABOURER",
     r"farm (?:worker|labou?rer|labou?r)|agricultural (?:worker|labou?rer|labou?r)|khet ?mazdoor"
     r"|landless labou?rer|work(?:ing|s)? (?:in|on) (?:other people'?s |someone'?s )?(?:fields|farms)"),
    ("FARMER",
     r"\bfarmer\b|(?<!pm )(?<!pm-)\bkisan\b(?! samman| mandhan| maandhan| credit| card)|cultivator|agriculturist|\bfarming\b"
     r"|i grow (?:crops|rice|wheat|paddy|vegetables)"
     r"|tenant farmer|sharecropper"),
    ("FACTORY_WORKER",
     r"factory (?:worker|job|work)|work(?:ing|s)? (?:in|at) (?:a |the )?(?:factory|mill|garment unit)"
     r"|mill worker|garment worker|textile worker"),
    ("SECURITY_GUARD", r"security guard|watchman|chowkidar"),
    ("SHOPKEEPER", r"shop ?keeper|shop owner|own (?:a |my )?(?:small )?shop|\bkirana\b|small shop|run (?:a )?(?:small )?shop"),
    ("SMALL_TRADER", r"\btrader\b|small business|business ?man|business owner|retailer|merchant|vyapari"),
    ("GOVT_EMPLOYEE",
     r"government (?:employee|job|servant|service|teacher)|sarkari naukri|work for the government"
     r"|central government employee|state government employee"),
    ("PRIVATE_EMPLOYEE",
     r"private (?:job|company|sector)|company job|office job"
     r"|work(?:ing|s)? (?:in|at|for) (?:a |an )?(?:company|office|it company|firm)|salaried"
     r"|software engineer"),
    ("STUDENT", r"(?:i am|i'm|im) (?:a )?student|studying in|college student|school student"),
    ("HOMEMAKER", r"housewife|home ?maker"),
    ("RETIRED", r"(?:i am|i'm|im) retired|retired (?:from|person|employee)|\bpensioner\b"),
    ("UNEMPLOYED",
     r"unemployed|no job|jobless|lost my job|(?:i am|i'm|im) not working|looking for (?:a )?(?:job|work)"
     r"|no work|without work|out of work"),
    ("SELF_EMPLOYED", r"self[- ]employed|own business|my own (?:work|business)"),
    ("DAILY_WAGE_WORKER",
     r"daily wage|daily wager|dihadi|labou?rer|mazdoor|majdoor|casual (?:labou?r|work)|\bhelper\b"
     r"|hard labou?r|labou?r work|unskilled work|manual work"),
]
_OCCUPATIONS_RE = [(code, re.compile(p)) for code, p in _OCCUPATIONS]

# The 18 PM Vishwakarma trades (canonical rule RULE_VISHWA_ELIG).
_TRADES: List[Tuple[str, str]] = [
    ("CARPENTER", r"carpenter|carpentry|badhai"), ("BOAT_MAKER", r"boat ?maker"),
    ("ARMOURER", r"armou?rer"), ("BLACKSMITH", r"blacksmith|\blohar\b"),
    ("HAMMER_AND_TOOL_KIT_MAKER", r"tool ?kit maker|hammer (?:and|&) tool"),
    ("LOCKSMITH", r"locksmith"), ("SCULPTOR_STONE_CARVER", r"sculptor|stone carver|stone breaker"),
    ("GOLDSMITH", r"goldsmith|\bsonar\b"), ("POTTER", r"\bpotter\b|kumhar|pottery"),
    ("COBBLER", r"cobbler|\bmochi\b|shoe ?maker|shoe repair"), ("MASON", r"\bmason\b|raj ?mistri"),
    ("BASKET_MAT_BROOM_MAKER_COIR_WEAVER", r"basket ?maker|mat ?maker|broom ?maker|coir"),
    ("DOLL_AND_TOY_MAKER", r"toy maker|doll maker"), ("BARBER", r"\bbarber\b|hair ?cutting|\bsalon\b"),
    ("GARLAND_MAKER", r"garland maker|mala ?maker|flower garland"),
    ("WASHERMAN", r"washer ?man|\bdhobi\b|laundry"), ("TAILOR", r"\btailor\b|\bdarzi\b|stitching"),
    ("FISHING_NET_MAKER", r"fishing net maker|net maker"),
]
_TRADES_RE = [(code, re.compile(p)) for code, p in _TRADES]

_SELF_EMPLOYED_OCCUPATIONS = {"STREET_VENDOR", "SHOPKEEPER", "SMALL_TRADER", "ARTISAN", "FARMER",
                              "SELF_EMPLOYED"}
_NOT_WORKING = {"UNEMPLOYED", "STUDENT", "RETIRED", "HOMEMAKER"}


# ─────────────────────────────────────────────────────────────────────────
# Location roles
# ─────────────────────────────────────────────────────────────────────────

_ORIGIN_CUE = re.compile(
    r"(?:\bfrom|\bnative (?:place )?(?:is |of )?|\bhome ?town (?:is )?|\bhome state (?:is )?"
    r"|\bmy village (?:is )?(?:in )?|\bvillage in|\bbelong(?:s|ing)? to|\boriginally(?: from)?"
    r"|\bborn in|\bhail from|\bmy home is in|\bmy family (?:lives|is|stays) in)\s*"
    r"(?:the )?(?:state of |district of |city of )?$"
)
# Only issuing phrases describe the card's state. "use my ration card in
# Tamil Nadu" is where the person wants to use it (a current-location cue).
_CARD_CUE = re.compile(
    r"(?:(?:ration card|card|aadhaar|job card)\s+(?:\w+\s+){0,2}(?:from|of)"
    r"|(?:ration card|card)\s+(?:is |was )?(?:issued|made|registered)\s+(?:\w+\s+){0,2}(?:in|from)"
    r"|(?:got|made|registered)\s+(?:my |a |the |our )?(?:ration )?card\s+(?:\w+\s+)?in)\s*(?:the )?(?:state of )?$"
)
_CURRENT_CUE = re.compile(
    r"(?:\bin|\bat|\bto|\bworking in|\bwork in|\bworking at|\bwork at|\blive in|\bliving in|\bstay in"
    r"|\bstaying in|\bnow in|\bcurrently in|\bhere in|\bbased in|\bsettled in|\bmoved to|\bshifted to"
    r"|\bmigrated to|\bcame to|\bcome to|\bcoming to|\breached|\barrived in|\bjob in|\bemployed in)\s*"
    r"(?:the )?(?:state of |city of |district of )?$"
)


class ScenarioEngine:
    """Stateless; one instance is shared."""

    def extract(self, text: str) -> Profile:
        p = Profile()
        t = normalize(text)
        if not t:
            return p
        self._age(t, p)
        self._gender_marital(t, p)
        self._family(t, p)
        self._pregnancy_disability(t, p)
        self._occupation(t, p)
        self._income(t, p)
        self._locations(t, p)
        self._documents_and_memberships(t, p)
        self._economic_flags(t, p)
        self._land_housing(t, p)
        return p

    # ── demographics ──────────────────────────────────────────────────────
    _AGE_RES = [
        re.compile(r"\b(?:i am|i'm|im|i m)\s+(\d{1,3})\s*(?:years?|yrs?)?\s*(?:old|of age)?\b"
                   r"(?!\s*(?:months?|days?|weeks?|kg|km|rs|rupees|thousand|lakhmark|%|percent|members|children|people|persons|kids))"),
        re.compile(r"\bmy age is\s*(?:about |around |nearly )?(\d{1,3})\b"),
        re.compile(r"\bage\s*(?:is|:|of)?\s*(\d{1,3})\b(?!\s*(?:limit|to|-))"),
        re.compile(r"\baged\s*(\d{1,3})\b"),
        re.compile(r"\b(\d{1,3})\s*[- ]?(?:years?|yrs?)\s*[- ]?old\b(?!\s*(?:daughter|son|child|girl|boy|kid|mother|father))"),
        re.compile(r"\b(\d{1,3})\s*years?\s*of\s*age\b"),
    ]
    _AGE_LO_RE = re.compile(r"\b(?:i am|i'm|im|my age is|age is)\s+(?:above|over|more than|older than)\s+(\d{1,3})\b")
    _AGE_HI_RE = re.compile(r"\b(?:i am|i'm|im|my age is|age is)\s+(?:below|under|less than|younger than)\s+(\d{1,3})\b")
    _SENIOR_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:a\s+)?senior citizen\b")
    _OTHER_PERSON_RE = re.compile(
        r"\b(?:daughter|son|child|children|kid|kids|baby|wife|husband|mother|father|brother|sister|girl|boy)\b")

    def _age(self, t: str, p: Profile) -> None:
        for i, rx in enumerate(self._AGE_RES):
            for m in rx.finditer(t):
                # Patterns 0/1 are first-person by construction; the generic
                # ones must not pick up "my daughter is 6 years old".
                if i >= 2 and self._OTHER_PERSON_RE.search(_clause_before(t, m.start())):
                    continue
                val = int(m.group(1))
                if 1 <= val <= 110:
                    p.set("age", val, _snip(t, m))
                    return
        m = self._AGE_LO_RE.search(t)
        if m:
            p.set_bound("age", lo=int(m.group(1)) + 1, evidence=_snip(t, m))
            return
        m = self._AGE_HI_RE.search(t)
        if m:
            p.set_bound("age", hi=int(m.group(1)) - 1, evidence=_snip(t, m))
            return
        m = self._SENIOR_RE.search(t)
        if m:
            p.set_bound("age", lo=60, evidence=_snip(t, m))

    _FEMALE_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:a\s+|an\s+)?(?:[\w-]+\s+){0,3}(woman|lady|female|girl|mother|widow|housewife|homemaker)\b")
    _MALE_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:a\s+|an\s+)?(?:[\w-]+\s+){0,3}(man|male|boy|father|widower)\b")
    # "I am a widow", "a 62 year old widow from Bihar", "widow, 62, no income".
    _WIDOW_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:an?\s+)?(?:\d+\s*(?:years?|yrs?)?\s*old\s+)?widow(?:ed)?\b"
                           r"|\bmy husband (?:has )?(?:died|passed away|expired|is dead|is no more)\b"
                           r"|(?:^|[,.]\s*)(?:an?\s+)?\d+\s*(?:years?|yrs?)?\s*old\s+widow\b")
    _WIDOWER_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:a\s+)?widower\b|\bmy wife (?:has )?(?:died|passed away|expired|is dead|is no more)\b")

    def _gender_marital(self, t: str, p: Profile) -> None:
        m = self._FEMALE_RE.search(t)
        if m and not _negated(t, m.start(1)):
            p.set("gender", "FEMALE", _snip(t, m))
        else:
            m = self._MALE_RE.search(t)
            if m and not _negated(t, m.start(1)):
                p.set("gender", "MALE", _snip(t, m))
        m = self._WIDOW_RE.search(t)
        if m:
            p.set("marital_status", "WIDOW", _snip(t, m))
            return
        m = self._WIDOWER_RE.search(t)
        if m:
            p.set("marital_status", "WIDOWER", _snip(t, m))
            return
        for status, rx in (
            ("DIVORCED", r"\b(?:i am|i'm|im) (?:a )?divorc(?:ed|ee)\b"),
            ("SEPARATED", r"\b(?:i am|i'm|im) separated\b"),
            ("SINGLE", r"\b(?:i am|i'm|im) (?:unmarried|single|not married)\b"),
            ("MARRIED", r"\b(?:i am|i'm|im) married\b|\bmy (?:wife|husband)\b"),
        ):
            m = re.search(rx, t)
            if m:
                p.set("marital_status", status, _snip(t, m))
                return

    _FAMILY_RES = [
        re.compile(r"\bfamily of (\d{1,2})\b"),
        re.compile(r"\b(\d{1,2}) (?:members|people|persons) in (?:my|our) (?:family|house|household|home)\b"),
        re.compile(r"\bwe are (\d{1,2}) (?:people|members|persons)\b"),
        re.compile(r"\b(?:my|our) family has (\d{1,2}) (?:members|people|persons)\b"),
        re.compile(r"\b(\d{1,2}) family members\b"),
    ]
    _CHILDREN_RE = re.compile(r"\b(?:i have|we have|with|have got) (\d{1,2}|a|an) (?:small |little |young )?(children|kids|sons|daughters|son|daughter|child)\b")
    _DAUGHTER_AGE_RES = [
        re.compile(r"\bdaughter(?:'s age)?\s+(?:is|who is|aged|of)\s+(\d{1,2})\b"),
        re.compile(r"\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]?old (?:daughter|girl child|baby girl)\b"),
        re.compile(r"\bdaughter (?:aged |of )?(\d{1,2}) (?:years?|yrs?)\b"),
    ]

    def _family(self, t: str, p: Profile) -> None:
        for rx in self._FAMILY_RES:
            m = rx.search(t)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 30:
                    p.set("family_size", n, _snip(t, m))
                break
        m = self._CHILDREN_RE.search(t)
        if m:
            n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
            p.set("children_count", n, _snip(t, m))
            if "daughter" in m.group(2):
                p.set("has_daughter", True, _snip(t, m))
        if re.search(r"\bmy daughter\b", t):
            p.set("has_daughter", True, "my daughter")
        ages = []
        for rx in self._DAUGHTER_AGE_RES:
            for m in rx.finditer(t):
                a = int(m.group(1))
                if 0 <= a <= 30 and a not in ages:
                    ages.append(a)
        if ages:
            p.set("girl_child_ages", ages, "daughter age")
            p.set("has_daughter", True, "daughter age")

    def _pregnancy_disability(self, t: str, p: Profile) -> None:
        m = re.search(r"\b(?:i am|i'm|im)\s+(?:\d+\s+months?\s+)?pregnant\b|\bi am expecting (?:a )?(?:baby|child)\b", t)
        if m and not _negated(t, m.start()):
            p.set("pregnancy_status", "PREGNANT", _snip(t, m))
        else:
            m = re.search(r"\bmy wife is (?:\d+ months? )?(?:pregnant|expecting)\b", t)
            if m:
                p.set("pregnancy_status", "SPOUSE_PREGNANT", _snip(t, m))
        m = re.search(r"\b(\d{1,3})\s*(?:%|percent|per cent)\s*(?:disabled|disability|handicap)", t)
        if m:
            pct = int(m.group(1))
            if 0 < pct <= 100:
                p.set("disability_percentage", pct, _snip(t, m))
                p.set("disability_status", True, _snip(t, m))
        m = re.search(
            r"\b(?:i am|i'm|im)\s+(?:\w+\s+)?(?:physically |visually |mentally )?(?:disabled|handicapped|blind|deaf|divyang|differently abled)\b"
            r"|\bi have (?:a )?(?:\d{1,3} ?(?:%|percent) )?disability\b|\bmy disability\b|\bperson with (?:a )?disability\b", t)
        if m:
            p.set("disability_status", not _negated(t, m.start() + 2), _snip(t, m))
        m = re.search(r"\bmy (?:son|daughter|father|mother|wife|husband|brother|sister|child) is (?:\w+ )?(?:disabled|handicapped|blind|deaf)\b", t)
        if m:
            p.set("household_disability", True, _snip(t, m))

    # ── work ──────────────────────────────────────────────────────────────
    def _occupation(self, t: str, p: Profile) -> None:
        occupation, trade = None, None
        # Lexicon order is priority order: the first code with a first-person,
        # non-negated mention wins ("mason" -> CONSTRUCTION_WORKER before ARTISAN).
        for code, rx in _OCCUPATIONS_RE:
            for m in rx.finditer(t):
                if _negated(t, m.start()) or not self._occupation_is_self(t, m.start()):
                    continue
                occupation = code
                p.evidence["occupation"] = _snip(t, m)[:60]
                break
            if occupation:
                break
        for code, rx in _TRADES_RE:
            m = rx.search(t)
            if m and not _negated(t, m.start()) and self._occupation_is_self(t, m.start()):
                trade = code
                p.set("trade", code, _snip(t, m))
                break
        if occupation is None and trade:
            occupation = "ARTISAN"
            p.evidence["occupation"] = p.evidence.get("trade", "")
        if occupation:
            p.set("occupation", occupation, p.evidence.get("occupation", ""))
            if occupation in _NOT_WORKING:
                p.set("employment_status", occupation if occupation != "HOMEMAKER" else "NOT_WORKING")
            elif occupation in _SELF_EMPLOYED_OCCUPATIONS:
                p.set("employment_status", "SELF_EMPLOYED", p.evidence.get("occupation", ""))
            else:
                p.set("employment_status", "EMPLOYED", p.evidence.get("occupation", ""))
            if occupation == "GOVT_EMPLOYEE":
                p.set("govt_employee", True, p.evidence.get("occupation", ""))
        if not p.has("employment_status"):
            m = re.search(r"\b(?:i|we) (?:am |are )?(?:working|work|employed)\b|\bmy job\b|\bi have a job\b", t)
            if m and not _negated(t, m.start() + 1):
                p.set("employment_status", "EMPLOYED", _snip(t, m))
        m = re.search(r"\bunorgani[sz]ed (?:sector|worker|labou?r)|\binformal (?:sector|worker)|\bdaily wage", t)
        if m and not _negated(t, m.start()) and _self_stated(t, m.start() + 1):
            p.set("is_unorganised_worker", True, _snip(t, m))

    _OTHER_SUBJECT_RE = re.compile(
        r"\bmy (?:husband|wife|son|daughter|father|mother|brother|sister|child|children|kid|kids|parents?)\b"
        r"(?:\s+\w+){0,2}\s+(?:is|was|works|worked|does|did)\b"
        # "my husband drives an auto" - the verb is part of the occupation match itself
        r"|\bmy (?:husband|wife|son|daughter|father|mother|brother|sister|child|children|kid|kids|parents?)\s*$")

    # Subject pronouns only: "give me details" / "tell us" say nothing about the speaker's job.
    _FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|im|my|myself|we|we're|our|mine)\b")

    @classmethod
    def _occupation_is_self(cls, t: str, start: int) -> bool:
        clause = re.split(r",| and | but ", _clause_before(t, start))[-1]
        if cls._OTHER_SUBJECT_RE.search(clause):
            return False  # "my husband is a driver" - someone else's job
        if re.search(r"\b(?:for|about|of)\s+(?:the\s+)?(?:\w+\s+)?$", clause) and not cls._FIRST_PERSON_RE.search(clause):
            return False  # "schemes for farmers" - asking about others
        whole = _clause_before(t, start)
        if cls._FIRST_PERSON_RE.search(whole):
            return True
        # "As a construction worker, what can I get?" / "Construction worker from
        # Bihar, what can I get?" - a leading role phrase in an utterance about "I".
        leading = whole.strip() == "" or bool(re.search(r"(?:^|[.!?]\s*)as an?\s*$", whole))
        return leading and bool(cls._FIRST_PERSON_RE.search(t))

    _INCOME_RE = re.compile(
        r"(?P<ctx>monthly (?:income|salary|earnings?|wages?|pay)|(?:family |household )?(?:income|salary|earnings?|wages?|pay)"
        r"|(?:i|we) (?:earn|make|get paid|am paid|are paid|receive)|earning|earns)"
        r"(?:\s+(?:is|of|about|around|approximately|approx|nearly|almost|only|just|roughly|rs|rupees|inr|per month|per day|a month|a day|monthly))*"
        r"\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<lakh>lakhmark)?\s*(?:rs|rupees|inr)?"
        r"(?:\s*(?P<period>(?:per|a|every|each|in a|in one) (?:month|day|year|annum|week)|monthly|daily|yearly|annually|per annum))?"
    )
    _INCOME_TAIL_RE = re.compile(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<lakh>lakhmark)?\s*(?:rs|rupees|inr)?\s*(?P<period>(?:per|a|every|each) (?:month|day|year|annum|week)|monthly|daily|yearly|annually)\b"
    )

    def _income(self, t: str, p: Profile) -> None:
        m = self._INCOME_RE.search(t)
        ctx = ""
        if m:
            ctx = m.group("ctx")
        else:
            m = self._INCOME_TAIL_RE.search(t)
            if m and not re.search(r"\b(?:earn|income|salary|wage|get|paid|pay|make)\b", t[max(0, m.start() - 40):m.start()]):
                m = None
        if m and re.search(r"\bno (?:income|earning|salary)\b", t):
            m = None
        if not m:
            if re.search(r"\b(?:i have|we have) no (?:income|earnings?)\b|\bno income\b", t):
                p.set("monthly_income", 0, "no income")
            return
        value = float(m.group("num"))
        period = (m.group("period") or "")
        window = t[max(0, m.start() - 25):m.end() + 12]
        family = bool(re.search(r"\b(?:family|household)\b", ctx + " " + t[max(0, m.start() - 20):m.start()]))
        if "month" in ctx or "month" in period:
            kind = "MONTHLY"
        elif "day" in period or "daily" in period or "daily" in window:
            kind = "DAILY"
        elif any(w in period for w in ("year", "annum", "annual", "yearly")) or "annual" in window or "yearly" in window:
            kind = "ANNUAL"
        elif m.group("lakh"):
            kind = "ANNUAL"
        elif "salary" in ctx:
            kind = "MONTHLY"
        elif 1000 <= value <= 60000:
            kind = "MONTHLY_ASSUMED"
        else:
            return
        ev = _snip(t, m).replace("lakhmark", "lakh")
        amount = int(value)
        if kind == "DAILY":
            p.set("daily_wage", amount, ev)
        elif kind == "ANNUAL":
            p.set("family_annual_income" if family else "annual_income", amount, ev)
        else:
            if kind == "MONTHLY_ASSUMED":
                ev = ev + " (assumed monthly)"
            p.set("family_monthly_income" if family else "monthly_income", amount, ev)

    # ── places ────────────────────────────────────────────────────────────
    def _locations(self, t: str, p: Profile) -> None:
        places = gazetteer.find_places(t)
        if not places:
            return
        for i, pl in enumerate(places):
            before = t[max(0, pl.start - 45):pl.start]
            after = t[pl.end:pl.end + 16]
            if re.match(r"\s*(?:ration card|card)\b", after):
                p.set("ration_card_state", pl.state, pl.surface + after[:12])
                continue
            role = None
            # "ration card from Bihar" describes the card, not the person.
            if _CARD_CUE.search(before):
                p.set("ration_card_state", pl.state, before[-20:] + pl.surface)
                continue
            # "from Bihar to Tamil Nadu" / "from X ... now in Y"
            if _ORIGIN_CUE.search(before):
                role = "origin"
            elif _CURRENT_CUE.search(before):
                role = "current"
            elif re.search(r"\bfrom\s+\S+(?:\s+\S+)?\s+to\s*$", before):
                role = "current"
            cue = ""
            if role:
                cm = (_ORIGIN_CUE if role == "origin" else _CURRENT_CUE).search(before)
                cue = (cm.group(0) if cm else "").strip() + " "
            if role == "origin":
                if not p.has("origin_state"):
                    p.set("origin_state", pl.state, cue + pl.surface)
                if pl.district and not p.has("origin_district"):
                    p.set("origin_district", pl.district, pl.surface)
            elif role == "current":
                if pl.city:
                    if not p.has("current_city"):
                        p.set("current_city", pl.city, cue + pl.surface)
                        p.set("area_type", "URBAN", "city " + pl.city)
                    if pl.district and not p.has("current_district"):
                        p.set("current_district", pl.district, pl.surface)
                if not p.has("current_state"):
                    p.set("current_state", pl.state, cue + pl.surface)
            else:
                if not p.has("mentioned_state"):
                    p.set("mentioned_state", pl.state, pl.surface)
        m = re.search(r"\b([a-z]+) district\b", t)
        if m and not p.has("current_district") and not p.has("origin_district"):
            name = m.group(1)
            if name not in ("my", "the", "this", "our", "which", "what", "same", "home", "native"):
                before = t[max(0, m.start() - 30):m.start()]
                target = "origin_district" if _ORIGIN_CUE.search(before + " ") or "from" in before else "current_district"
                p.set(target, name.title(), _snip(t, m))
        self._migration(t, p)

    @staticmethod
    def _migration(t: str, p: Profile) -> None:
        origin, current = p.get("origin_state"), p.get("current_state")
        if origin and current:
            if origin != current:
                p.set("migration_status", "INTER_STATE_MIGRANT", f"{origin} -> {current}")
            elif p.get("origin_district") and p.get("current_district") and p.get("origin_district") != p.get("current_district"):
                p.set("migration_status", "INTRA_STATE_MIGRANT", f"{p.get('origin_district')} -> {p.get('current_district')}")
            return
        m = re.search(r"\b(?:i am|i'm|im|we are)\s+(?:a\s+)?migrant|\bmigrant (?:worker|labou?rer|family)\b"
                      r"|\b(?:i|we) (?:have )?(?:migrated|moved|shifted|came) (?:here |to \w+ )?(?:from|for work)", t)
        if m and _self_stated(t, m.start() + 1):
            p.set("migration_status", "MIGRANT", _snip(t, m))

    # ── cards, ids, memberships ───────────────────────────────────────────
    _MEMBERSHIPS = [
        ("SCH_PMJAY", r"\b(?:have|got|hold|with) (?:an? |my |our )?(?:ayushman|golden|pm-?jay|pmjay|abha) card\b|\bmy (?:ayushman|golden|pm-?jay) card\b|\bayushman card holder\b"),
        ("SCH_BOCW", r"\b(?:have|got|hold|with) (?:an? |my |our )?(?:labou?r|bocw|construction worker'?s?|nirman shramik) card\b|\bmy (?:labou?r|bocw) card\b"
                     r"|\bregistered (?:with|in|under|at) (?:the )?(?:bocw|construction workers'? welfare board|welfare board|labou?r board)\b"),
        ("SCH_PMJDY", r"\bjan ?dhan (?:account|khata)\b"),
        ("SCH_NSAP_OA", r"\b(?:i|we) (?:get|receive|am getting|am receiving|draw) (?:the |an? )?old age pension\b"),
        ("SCH_NSAP_W", r"\b(?:i|we) (?:get|receive|am getting|am receiving|draw) (?:the |a )?widow pension\b"),
        ("SCH_NSAP_D", r"\b(?:i|we) (?:get|receive|am getting|am receiving|draw) (?:the |a )?disability pension\b"),
        ("SCH_PMUY", r"\bujjwala (?:gas |lpg )?connection\b"),
        ("SCH_PMKISAN", r"\b(?:i|we) (?:get|receive|am getting) (?:the )?(?:pm[- ]?kisan|kisan samman)\b"),
        ("SCH_APY", r"\b(?:joined|enrolled in|member of|contribute to|paying for) (?:the )?atal pension\b"),
        ("SCH_PMSYM", r"\b(?:joined|enrolled in|member of|contribute to) (?:the )?(?:pm[- ]?sym|shram yogi)\b"),
    ]
    _MEMBERSHIPS_RE = [(sid, re.compile(rx)) for sid, rx in _MEMBERSHIPS]

    def _documents_and_memberships(self, t: str, p: Profile) -> None:
        # Ration card
        m = re.search(r"\blost (?:my |our )?(?:\w+ )?ration card\b", t)
        if m:
            p.set("ration_card_lost", True, _snip(t, m))
        else:
            m = re.search(
                r"\b(?:i|we) (?:[\w']+ ){0,2}(?:have|hold|got|possess|own|carry)\s+(?:a |an |my |our |the |any )?(?:\w+ ){0,2}ration card\b"
                r"|\bmy (?:\w+ )?ration card\b|\bration card holder\b|\bwith (?:my |our |a )?(?:\w+ )?ration card\b"
                r"|\b(?:i|we) (?:[\w']+ ){0,2}(?:have|hold) (?:a |an )?(?:aay|phh|bpl|apl|antyodaya|priority) card\b", t)
            if re.search(r"\bno ration card\b|\bwithout (?:a |any )?ration card\b", t):
                p.set("ration_card", False, "no ration card")
            elif m:
                p.set("ration_card", not _negated(t, m.start() + 1, window=30) and not _NEG_RE.search(_snip(t, m)),
                      _snip(t, m))
        for code, rx in (("AAY", r"\b(?:aay|antyodaya anna|antyodaya) (?:ration )?card\b|\bantyodaya anna yojana card\b"),
                         ("PHH", r"\b(?:phh|priority household)\b"),
                         ("BPL", r"\bbpl (?:ration )?card\b"),
                         ("APL", r"\bapl (?:ration )?card\b")):
            m = re.search(rx, t)
            if m and not _negated(t, m.start()):
                p.set("ration_card_type", code, _snip(t, m))
                if code in ("AAY", "PHH"):
                    # NFSA beneficiaries are exactly the AAY and PHH categories (pds.json eligibility).
                    p.set("nfsa_beneficiary", True, _snip(t, m))
                if code == "BPL":
                    p.set("is_bpl", True, _snip(t, m))
                if not p.has("ration_card") and _self_stated(t, m.start() + 1):
                    p.set("ration_card", True, _snip(t, m))
                break
        m = re.search(r"\b(?:nfsa|food security act) (?:beneficiary|card|ration card)\b|\bunder (?:the )?(?:nfsa|food security act)\b", t)
        if m and _self_stated(t, m.start() + 1) and not _negated(t, m.start()):
            p.set("nfsa_beneficiary", True, _snip(t, m))

        # Aadhaar and its ration-card seeding
        m = re.search(
            r"\baadhaar(?: card| number)? (?:is |has been |was |got |is already |has already been )?(?:linked|seeded|connected|added|attached|mapped)"
            r"(?: \w+){0,2} (?:with|to|in) (?:my |our |the )?ration(?: card)?"
            r"|\bration card (?:is |has been )?(?:already )?(?:linked|seeded|connected) (?:with|to) (?:my |our |the )?aadhaar", t)
        if m:
            p.set("aadhaar_seeded_ration", not (_negated(t, m.start()) or " not " in f" {_snip(t, m)} "), _snip(t, m))
            p.set("aadhaar", True, _snip(t, m))
        elif "ration" in t:
            m = re.search(r"\baadhaar(?: card| number)? (?:is |was |has |has been )?(?:still )?not (?:been |yet )?"
                          r"(?:linked|seeded|connected|added)\b", t)
            if m:
                p.set("aadhaar_seeded_ration", False, _snip(t, m))
                p.set("aadhaar", True, _snip(t, m))
        m = re.search(r"\b(?:i|we) (?:[\w']+ ){0,2}(?:have|got|hold) (?:an? |my |our |any )?aadhaar\b|\bmy aadhaar(?: card| number)?\b|\bwith (?:my )?aadhaar\b", t)
        if m and not p.has("aadhaar"):
            p.set("aadhaar", not (_negated(t, m.start() + 1) or _NEG_RE.search(_snip(t, m))), _snip(t, m))
        elif re.search(r"\bno aadhaar\b|\bwithout (?:an? )?aadhaar\b", t) and not p.has("aadhaar"):
            p.set("aadhaar", False, "no aadhaar")

        # e-Shram registration
        m = re.search(r"\b(?:registered|enrolled|registration done|signed up)\b[^.]{0,20}\be-shram\b"
                      r"|\b(?:have|got|hold) (?:an? |my )?(?:e-shram|uan|shram) card\b|\bmy (?:e-shram|uan) (?:card|number)\b", t)
        if m:
            p.set("e_shram_registration", not _negated(t, m.start() + 1, window=30), _snip(t, m))
            if p.get("e_shram_registration"):
                p.set("existing_scheme_membership", ["SCH_ESHRAM"], _snip(t, m))

        for sid, rx in self._MEMBERSHIPS_RE:
            m = rx.search(t)
            if m and not _negated(t, m.start()) and _self_stated(t, m.start() + 1):
                p.set("existing_scheme_membership", [sid], _snip(t, m))
                if sid == "SCH_PMJDY":
                    p.set("has_bank_account", True, _snip(t, m))

        # Bank account
        m = re.search(
            r"\b(?:i|we) (?:[\w']+ ){0,2}(?:have|hold|got|opened) (?:an? |my |our |any )?(?:bank|savings?|jan ?dhan|post office|zero balance) (?:bank )?(?:account|a/c|khata)\b"
            r"|\bmy (?:bank|savings?|jan ?dhan|post office) (?:account|a/c|khata|passbook)\b", t)
        if m:
            p.set("has_bank_account", not (_negated(t, m.start() + 1, window=30) or _NEG_RE.search(_snip(t, m))),
                  _snip(t, m))
        elif re.search(r"\bno bank account\b|\bwithout (?:a )?bank account\b", t):
            p.set("has_bank_account", False, "no bank account")

    # ── money, tax, social security, poverty ─────────────────────────────
    def _economic_flags(self, t: str, p: Profile) -> None:
        m = re.search(r"\b(?:i|we) (?:[\w']+ ){0,2}(?:pay|file) (?:any )?(?:income )?tax(?:es)?\b|\btax ?payer\b|\bfile (?:itr|income tax return)", t)
        if m:
            neg = (_negated(t, m.start() + 1, window=30) or bool(_NEG_RE.search(_snip(t, m)))
                   or bool(re.search(r"\bnot (?:an? )?(?:income )?tax ?payer", t)))
            p.set("income_tax_payer", not neg, _snip(t, m))
        m = re.search(r"\b(?:pf|epf|provident fund)\b[^.]{0,20}\b(?:deducted|cut|account|member)\b|\bepfo\b|\besic?\b|\besi card\b"
                      r"|\bcovered (?:by|under) (?:esi|esic|epf|pf|epfo)\b", t)
        if m and _self_stated(t, m.start() + 1):
            p.set("epfo_esic_member", not _negated(t, m.start() + 1, window=30), _snip(t, m))
        m = re.search(r"\bbpl\b|\bbelow (?:the )?poverty line\b", t)
        if m and not p.has("is_bpl") and _self_stated(t, m.start() + 1):
            p.set("is_bpl", not _negated(t, m.start()), _snip(t, m))
        m = re.search(r"\bapl\b|\babove (?:the )?poverty line\b", t)
        if m and not p.has("is_bpl") and _self_stated(t, m.start() + 1) and not _negated(t, m.start()):
            p.set("is_bpl", False, _snip(t, m))
        m = re.search(r"\bindian citizen\b|\bcitizen of india\b|\b(?:i am|i'm|im) (?:an )?indian\b", t)
        if m and not _negated(t, m.start()):
            p.set("citizenship", "INDIAN", _snip(t, m))
        m = re.search(r"\b(?:i|we) (?:get|receive|draw) (?:a )?(?:monthly )?pension of (?:rs |rupees )?(\d+)", t)
        if m:
            p.set("monthly_pension", int(m.group(1)), _snip(t, m))
        # Construction days in the last 12 months
        m = re.search(r"\b(\d{1,3}) days\b[^.]{0,30}\b(?:construction|site|building)\b|\b(?:construction|site|building)\b[^.]{0,30}\b(\d{1,3}) days\b", t)
        if m:
            days = int(m.group(1) or m.group(2))
            if 0 < days <= 366:
                p.set("construction_days_12m", days, _snip(t, m))
        elif p.get("occupation") == "CONSTRUCTION_WORKER":
            m = re.search(r"\b(?:for|since) (?:the )?(?:last |past )?(\d{1,2}) (months?|years?)\b", t)
            if m:
                n, unit = int(m.group(1)), m.group(2)
                if (unit.startswith("year") and n >= 1) or (unit.startswith("month") and n >= 3):
                    p.set_bound("construction_days_12m", lo=90, evidence=_snip(t, m))
        m = re.search(r"\b(?:someone|anyone|a member|one member|my (?:husband|wife|father|son|brother)) (?:in my family )?(?:is|works as) (?:an? )?government employee\b", t)
        if m and not _negated(t, m.start()):
            p.set("govt_employee_in_family", True, _snip(t, m))

    # ── land, housing, utilities ──────────────────────────────────────────
    def _land_housing(self, t: str, p: Profile) -> None:
        m = re.search(r"\b(\d+(?:\.\d+)?) ?(acres?|hectares?|ha)\b(?: of)?(?: \w+)? (?:land|farm|field)?", t)
        if m and _self_stated(t, m.start() + 1):
            area = float(m.group(1)) * (0.4047 if m.group(2).startswith("acre") else 1.0)
            p.set("land_area_ha", round(area, 3), _snip(t, m))
            p.set("owns_land", True, _snip(t, m))
        else:
            m = re.search(r"\b(?:i|we) (?:[\w']+ ){0,2}(?:have|own) (?:some |any |a little |a small |our own |my own )?(?:farm ?land|land|agricultural land|a farm|fields?)\b|\bmy (?:farm|land|fields?)\b", t)
            if m:
                p.set("owns_land", not (_negated(t, m.start() + 1) or _NEG_RE.search(_snip(t, m))), _snip(t, m))
            elif re.search(r"\blandless\b", t) and _SELF_RE.search(t):
                p.set("owns_land", False, "landless")
        m = re.search(r"\btenant farmer\b|\b(?:on )?lease(?:d)? land\b", t)
        if m:
            p.set("farmer_type", "TENANT_FARMER", _snip(t, m))
        elif re.search(r"\bsharecropp(?:er|ing)\b|\bbatai\b", t):
            p.set("farmer_type", "SHARECROPPER", "sharecropper")
        m = re.search(r"\b(?:i|we) (?:[\w']+ ){0,2}(?:have|got) (?:an? |any )?(?:gas|lpg) (?:connection|cylinder|stove)\b", t)
        if m:
            p.set("lpg_connection", not (_negated(t, m.start() + 1) or _NEG_RE.search(_snip(t, m))), _snip(t, m))
        elif re.search(r"\bcook (?:on|with|using) (?:wood|firewood|chulha|coal|kerosene|cow ?dung)\b", t):
            p.set("lpg_connection", False, "cooks on wood/chulha")
        m = re.search(r"\b(?:no |don't have (?:a |any |my )?|do not have (?:a |any |my )?|dont have (?:a |any )?|without (?:a )?)"
                      r"(?:own |proper |pucca )?(?:house|home)\b|\bhomeless\b|\bhouseless\b", t)
        if m and _SELF_RE.search(t):
            p.set("housing_status", "HOUSELESS", _snip(t, m))
        else:
            m = re.search(r"\bkutcha (?:house|home)\b|\bmud house\b|\bjhuggi\b|\bslum\b|\btemporary (?:house|shelter)\b", t)
            if m and _SELF_RE.search(t):
                p.set("housing_status", "KUTCHA", _snip(t, m))
            else:
                m = re.search(r"\bpucca (?:house|home)\b|\b(?:i|we) own (?:a |our )?(?:pucca )?house\b", t)
                if m and not _negated(t, m.start()):
                    p.set("housing_status", "PUCCA", _snip(t, m))
        m = re.search(r"\b(?:have|got) (?:an? )?electricity connection\b|\belectricity bill (?:is )?in my name\b", t)
        if m:
            p.set("has_electricity_connection", not _negated(t, m.start() + 1), _snip(t, m))
        # rural / urban residence (explicit statements only)
        if not p.has("area_type"):
            m = re.search(r"\b(?:i|we) (?:\w+ )?(?:live|stay|am living|are living) in (?:a |my |our |the )?village\b|\brural area\b"
                          r"|\b(?:in|at) (?:my|our) village\b", t)
            if m:
                p.set("area_type", "RURAL", _snip(t, m))
            else:
                m = re.search(r"\b(?:i|we) (?:\w+ )?(?:live|stay) in (?:a |the )?(?:city|town)\b|\burban area\b", t)
                if m:
                    p.set("area_type", "URBAN", _snip(t, m))
