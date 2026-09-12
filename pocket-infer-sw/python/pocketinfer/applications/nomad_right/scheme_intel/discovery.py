"""
Discovery ranking and "the single most useful question".

Ranking = eligibility status x relevance to the person (occupation the
scheme targets, topic asked about, portability for migrants, state match,
optional semantic similarity from the RAG index). INELIGIBLE schemes are
never recommended.

The follow-up question is the missing fact that would settle the most
(relevance-weighted) open conditions among the schemes actually being
recommended - asked once, never repeated while unanswered.
"""

from typing import Dict, List, Optional

from pocketinfer.applications.nomad_right.scheme_intel.models import (
    Intent, Profile, SchemeEvaluation, Status,
)
from pocketinfer.applications.nomad_right.scheme_intel.rules_engine import GATE_FIELDS, IDENTITY_FIELDS

STATUS_WEIGHT = {
    Status.ELIGIBLE: 1.0,
    Status.LIKELY_ELIGIBLE: 0.85,
    Status.POTENTIALLY_ELIGIBLE: 0.6,
    Status.INSUFFICIENT_INFORMATION: 0.25,
    Status.UNKNOWN: 0.2,
    Status.INELIGIBLE: 0.0,
}
RECOMMENDABLE = {Status.ELIGIBLE, Status.LIKELY_ELIGIBLE, Status.POTENTIALLY_ELIGIBLE}

# Slots a person can actually answer, with the question to ask (<= 12 words).
SLOT_QUESTIONS: Dict[str, str] = {
    "occupation": "What work do you do?",
    "age": "How old are you?",
    "current_state": "Which state are you living in now?",
    "origin_state": "Which state are you from?",
    "monthly_income": "About how much do you earn in a month?",
    "ration_card": "Do you have a ration card?",
    "has_bank_account": "Do you have a bank account?",
    "construction_days_12m": "Did you do construction work for 90 days in the last year?",
    "is_bpl": "Does your family have a BPL card?",
    "poor_household": "Does your family have a BPL card?",
    "gender": "Are you a man or a woman?",
    "area_type": "Do you live in a village or a city?",
    "owns_land": "Do you own farm land?",
    "epfo_esic_member": "Is PF or ESI deducted from your pay?",
    "income_tax_payer": "Do you pay income tax?",
    "aadhaar_seeded_ration": "Is your Aadhaar linked to your ration card?",
    "girl_child_under_10": "Do you have a daughter below 10 years?",
    "disability_percentage": "What percentage of disability is on your certificate?",
    "lpg_connection": "Does your family already have a gas connection?",
    "marital_status": "Are you married, single or widowed?",
}
# Ties broken in this order (earlier = more broadly useful).
SLOT_PRIORITY = [
    "occupation", "age", "current_state", "ration_card", "monthly_income", "has_bank_account",
    "construction_days_12m", "is_bpl", "poor_household", "gender", "area_type", "owns_land",
    "girl_child_under_10", "disability_percentage", "lpg_connection",
    "marital_status", "epfo_esic_member", "income_tax_payer", "aadhaar_seeded_ration", "origin_state",
]
_DISCOVERY_LIKE = {Intent.SCHEME_DISCOVERY, Intent.WORKER_SUPPORT, Intent.PENSION_SUPPORT,
                   Intent.FINANCIAL_SUPPORT, Intent.EDUCATION_SUPPORT}


def is_migrant(d: Profile) -> bool:
    return bool(d.get("migration_status")) or bool(
        d.get("origin_state") and d.get("current_state") and d.get("origin_state") != d.get("current_state"))


def relevance(repo, rules, sid: str, d: Profile, domains: List[str], rag_score: Optional[float]) -> float:
    s = repo.scheme(sid)
    targets = s.get("occupation") or []
    scheme_domains = s.get("domains") or []
    rel = 0.0
    if d.get("occupation") in targets or d.get("trade") in targets:
        rel += 1.2
    elif "UNORGANISED" in targets and d.get("is_unorganised_worker") is True:
        rel += 0.8
    if domains and set(domains) & set(scheme_domains):
        rel += 0.6
    if is_migrant(d) and "MIGRANT" in scheme_domains:
        rel += 0.4
    if d.get("housing_status") in ("HOUSELESS", "KUTCHA") and "HOUSING" in scheme_domains:
        rel += 0.6
    state = s.get("state")
    if state and state != "ALL" and state in (d.get("current_state"), d.get("origin_state")):
        rel += 0.3
    if rag_score is not None:
        rel += max(0.0, rag_score - 0.78) * 3.0
    return round(rel, 3)


def rank(evals: List[SchemeEvaluation], repo, rules, d: Profile, domains: List[str],
         rag_scores: Optional[Dict[str, float]] = None) -> List[SchemeEvaluation]:
    rag_scores = rag_scores or {}
    for ev in evals:
        ev.relevance = relevance(repo, rules, ev.scheme_id, d, domains, rag_scores.get(ev.scheme_id))
        # Stated life situation that a scheme is built around (widow, disabled,
        # 60 or older ...) outranks a scheme that merely does not exclude the person.
        ident = sum(1 for c in ev.passed for f in c.field.split("|")
                    if f in IDENTITY_FIELDS or (f == "age" and c.op == ">=" and (c.expected or 0) >= 60))
        ev.relevance = round(ev.relevance + min(0.6, 0.3 * ident), 3)
        score = STATUS_WEIGHT[ev.status] * (0.5 + ev.relevance) + 0.05 * len(ev.passed)
        if "VALIDITY_CHECK" in ev.notes:
            score *= 0.7
        ev.rank_score = round(score, 4)
    keep = [ev for ev in evals if ev.status != Status.INELIGIBLE]
    keep.sort(key=lambda e: (-e.rank_score, -e.relevance, e.scheme_id))
    return keep


def recommendable(ranked: List[SchemeEvaluation], limit: int = 3) -> List[SchemeEvaluation]:
    return [ev for ev in ranked if ev.status in RECOMMENDABLE][:limit]


def pick_question(ranked: List[SchemeEvaluation], d: Profile, asked: List[str], intent: Intent,
                  restrict: bool = True) -> Optional[str]:
    """
    restrict=True (discovery): only the schemes being recommended, plus clearly
    relevant ones, may drive the question. restrict=False: a single-scheme
    eligibility check - its own open conditions decide.
    """
    if not d.has("occupation") and "occupation" not in asked and intent in _DISCOVERY_LIKE:
        return "occupation"
    if restrict:
        top = recommendable(ranked, 3)
        pool = top + [ev for ev in ranked if ev not in top and ev.relevance >= 0.6]
    else:
        pool = list(ranked)
    scores: Dict[str, float] = {}
    for ev in pool:
        w = 0.5 + ev.relevance
        seen = set()  # each fact counts once per scheme
        for weight, conds in ((1.5, ev.unknown_core), (0.5, ev.unknown_verify), (0.2, ev.exclusions_unknown)):
            for c in conds:
                members = c.field.split("|")
                # An "either/or" condition whose other branch the person cannot
                # answer (SECC list, Awaas+ list ...) is not settled by asking.
                if len(members) > 1 and any(m not in SLOT_QUESTIONS for m in members):
                    continue
                for f in members:
                    if f in seen or f not in SLOT_QUESTIONS or d.has(f) or f in asked:
                        continue
                    seen.add(f)
                    # "Who is this scheme for" facts settle more than generic ones.
                    bonus = 1.2 if weight == 1.5 and f in GATE_FIELDS else 1.0
                    scores[f] = scores.get(f, 0.0) + weight * w * bonus
    if not scores:
        return None
    best = max(scores.values())
    for slot in SLOT_PRIORITY:
        if scores.get(slot) == best:
            return slot
    return max(scores, key=scores.get)


def question_text(slot: str, d: Profile) -> str:
    if slot == "ration_card" and is_migrant(d) and d.get("origin_state"):
        return f"Do you have a ration card from {d.get('origin_state')}?"
    return SLOT_QUESTIONS.get(slot, "")
