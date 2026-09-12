"""
QueryRouter - decides HOW a query is answered, never the answer itself.

  DIRECT_INFORMATION  SQLite fact + template        ("What is ONORC?", documents, how to apply)
  ELIGIBILITY         rules engine + template       ("Can I get my ration here?")
  SCHEME_DISCOVERY    SQLite filter -> rules -> vector ranking -> template
  COMPLEX_SCENARIO    SQLite -> rules -> vector search -> Qwen only if nothing else answers
  CLARIFY             ask one question (which scheme / unverified scheme name)
  DEFER               hand back to the legacy pipeline (off-topic, small talk,
                      wage-rights cases the legacy rules already cover)
VISION is routed by SchemeIntelligence.handle_vision(), not here.
"""

import re
from typing import Optional

from pocketinfer.applications.nomad_right.scheme_intel.models import (
    DOMAIN_OF_INTENT, Intent, IntentResult, Route, RouteDecision,
)

_NEEDS_SCHEME = {Intent.GENERAL_SCHEME_INFORMATION, Intent.REQUIRED_DOCUMENTS,
                 Intent.APPLICATION_PROCEDURE, Intent.BENEFIT_INFORMATION, Intent.ELIGIBILITY_CHECK,
                 Intent.APPLICATION_STATUS, Intent.GRIEVANCE}
# Wage theft / unpaid wages / worksite injury: the legacy RulesEngine has
# dedicated, sourced OSH-Code answers for these (ESHRAM_WAGE_RIGHTS).
# One-word / bare-need inputs that name no scheme and no topic worth guessing at.
_CARD_ONLY_RE = re.compile(r"^(?:(?:the|my|a|our) )?cards?(?: please)?\.?$")
_VAGUE_NEED_RE = re.compile(
    r"^(?:(?:i|we) (?:need|want|require|am looking for|are looking for)(?: some| more| a| any)? "
    r"(?:money|help|support|benefits?|cash|assistance|scheme|schemes|yojana|a scheme|government help|financial help)"
    r"(?: please)?|(?:money|help|benefits?|scheme|schemes|yojana|apply|application|form|forms|please help|help me|assistance))\.?$")
_WAGE_RIGHTS_RE = re.compile(
    r"\b(?:wages?|salary|payment|pay|paid|contractor|employer|minimum wage|form ?11|labou?r office|injur\w*|accident)\b")


class QueryRouter:
    def __init__(self, repo):
        self.repo = repo

    def route(self, ir: IntentResult, text_norm: str, active_sid: Optional[str],
              context_sid: Optional[str]) -> RouteDecision:
        intent = ir.intent
        if intent == Intent.OTHER:
            # Too little to answer, but clearly about the kiosk's topics: ask a
            # short clarifying question instead of guessing a scheme or leaving it
            # to the general language model.
            if _CARD_ONLY_RE.match(text_norm) and not ir.scheme_ids:
                return RouteDecision(Route.CLARIFY, reason="which card")
            if _VAGUE_NEED_RE.match(text_norm) and not ir.scheme_ids:
                return RouteDecision(Route.CLARIFY, reason="topic")
            return RouteDecision(Route.DEFER, reason="not a scheme question")

        sid = ir.scheme_ids[0] if ir.scheme_ids else None
        if not sid and (ir.refers_to_context or intent in _NEEDS_SCHEME):
            sid = active_sid or context_sid
        if not sid and ir.scheme_hints and intent in _NEEDS_SCHEME:
            sid = ir.scheme_hints[0]

        if ir.unverified_name and not ir.scheme_ids:
            return RouteDecision(Route.CLARIFY, reason="unverified")

        domain = DOMAIN_OF_INTENT.get(intent)
        if intent == Intent.WORKER_SUPPORT and _WAGE_RIGHTS_RE.search(text_norm) and not ir.discovery_cue:
            return RouteDecision(Route.DEFER, reason="wage rights (legacy rules)")

        if ir.complex_cue and (sid or ir.domains) and intent not in (Intent.SCHEME_DISCOVERY,):
            cands = [sid] if sid else []
            for d in ir.domains:
                cands += [c for c in self.repo.domain_primary(d) if c not in cands]
            return RouteDecision(Route.COMPLEX_SCENARIO, scheme_id=sid, candidates=cands, reason="complex cue")

        if intent in (Intent.GENERAL_SCHEME_INFORMATION, Intent.REQUIRED_DOCUMENTS,
                      Intent.APPLICATION_PROCEDURE, Intent.BENEFIT_INFORMATION):
            if sid:
                return RouteDecision(Route.DIRECT_INFORMATION, scheme_id=sid)
            if intent == Intent.BENEFIT_INFORMATION:
                return RouteDecision(Route.SCHEME_DISCOVERY, reason="benefits, no scheme")
            for d in ir.domains:
                prim = self.repo.domain_primary(d)
                if prim and d != "MIGRANT":
                    return RouteDecision(Route.DIRECT_INFORMATION, scheme_id=prim[0], reason=f"domain {d}")
            return RouteDecision(Route.CLARIFY, reason="which scheme")

        if intent == Intent.ELIGIBILITY_CHECK:
            if sid:
                return RouteDecision(Route.ELIGIBILITY, scheme_id=sid, candidates=[sid])
            return RouteDecision(Route.SCHEME_DISCOVERY, reason="eligibility, no scheme")

        if domain:
            if domain == "MIGRANT":
                return RouteDecision(Route.DIRECT_INFORMATION, scheme_id=sid, reason="migrant overview")
            if sid and domain in (self.repo.scheme(sid).get("domains") or []):
                return RouteDecision(Route.ELIGIBILITY, scheme_id=sid, candidates=[sid], reason=f"{domain} + scheme")
            prim = self.repo.domain_primary(domain)
            if prim:
                return RouteDecision(Route.ELIGIBILITY, scheme_id=prim[0], candidates=prim, reason=f"{domain} primary")
            return RouteDecision(Route.SCHEME_DISCOVERY, reason=f"{domain} discovery")

        if intent == Intent.SCHEME_DISCOVERY:
            return RouteDecision(Route.SCHEME_DISCOVERY)

        if intent in (Intent.APPLICATION_STATUS, Intent.GRIEVANCE, Intent.DOCUMENT_HELP):
            if not sid:
                for d in ir.domains:
                    prim = self.repo.domain_primary(d)
                    if prim and d != "MIGRANT":
                        sid = prim[0]
                        break
            return RouteDecision(Route.DIRECT_INFORMATION, scheme_id=sid, reason=intent.value)

        return RouteDecision(Route.DEFER, reason="unrouted")
