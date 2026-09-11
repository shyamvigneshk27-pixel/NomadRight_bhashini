"""
SessionState - the lightweight memory of one conversation.

Holds only what is needed to avoid asking twice: the profile built from the
person's own statements, the current intent, the scheme being discussed, the
last recommended schemes and the one pending question. No transcript is
stored. Everything is forgotten after SESSION_TTL_S of inactivity so the
next person at the kiosk starts clean.
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C
from pocketinfer.applications.nomad_right.scheme_intel.models import Intent, Profile

_BOOLEAN_SLOTS = {
    "ration_card", "aadhaar", "aadhaar_seeded_ration", "has_bank_account", "income_tax_payer",
    "epfo_esic_member", "is_bpl", "owns_land", "lpg_connection", "e_shram_registration",
    "nfsa_beneficiary", "has_daughter", "disability_status", "in_secc_2011",
}
# A bare answer to a pending question is rephrased into a first-person
# statement so ScenarioEngine can read it with its normal rules.
_REPLY_TEMPLATES = {
    "age": "I am {} years old",
    "occupation": "I work as a {}",
    "origin_state": "I am from {}",
    "current_state": "I am living in {}",
    "monthly_income": "My monthly income is {}",
    "family_size": "We are {} people in my family",
    "construction_days_12m": "I worked {} days in construction",
}
_YES_RE = re.compile(r"^(?:yes|yeah|yep|haan|han|ha|ji|ji haan|sure|correct|right|of course|i do|i have|we have|it is|yes i do|yes i have)\b")
_NO_RE = re.compile(r"^(?:no|nope|nahi|nahin|not|never|i don't|i do not|we don't|no i don't|no i do not|it is not)\b")


class SessionState:
    def __init__(self, ttl_s: float = C.SESSION_TTL_S, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_s
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self.profile = Profile()
        self.current_intent: Optional[Intent] = None
        self.active_scheme_id: Optional[str] = None
        self.last_candidates: List[str] = []
        self.pending_slot: Optional[str] = None
        self.asked_slots: List[str] = []
        self.missing_information: List[str] = []
        self.turns = 0
        self.last_ts = self._clock()

    def expire_if_idle(self) -> bool:
        if self.turns and self._clock() - self.last_ts > self._ttl:
            self.reset()
            return True
        return False

    def touch(self) -> None:
        self.turns += 1
        self.last_ts = self._clock()

    def absorb(self, facts: Profile) -> List[str]:
        changed = self.profile.merge(facts)
        if self.pending_slot and (facts.has(self.pending_slot) or self.pending_slot in facts.bounds):
            self.pending_slot = None
        return changed

    def pending_reply_as_statement(self, text_norm: str) -> Optional[str]:
        """Turns "yes" / "32" / "Bihar" into a statement for the pending slot, if it looks like an answer."""
        slot = self.pending_slot
        if not slot or not text_norm:
            return None
        if slot in _BOOLEAN_SLOTS:
            return None  # handled by boolean_reply()
        if slot in _REPLY_TEMPLATES and len(text_norm.split()) <= 6:
            value = re.sub(r"^(?:i am|i'm|im|my|it is|it's|about|around)\s+", "", text_norm).strip(" .")
            return _REPLY_TEMPLATES[slot].format(value) if value else None
        return None

    def boolean_reply(self, text_norm: str) -> Optional[bool]:
        if self.pending_slot not in _BOOLEAN_SLOTS or not text_norm or len(text_norm.split()) > 8:
            return None
        if _NO_RE.search(text_norm):
            return False
        if _YES_RE.search(text_norm):
            return True
        return None

    def remember(self, intent: Intent, scheme_id: Optional[str] = None, candidates: Optional[List[str]] = None,
                 asked_slot: Optional[str] = None, missing: Optional[List[str]] = None) -> None:
        self.current_intent = intent
        if scheme_id:
            self.active_scheme_id = scheme_id
        if candidates is not None:
            self.last_candidates = list(candidates)
        self.pending_slot = asked_slot
        if asked_slot and asked_slot not in self.asked_slots:
            self.asked_slots.append(asked_slot)
        if missing is not None:
            self.missing_information = list(missing)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.scenario_view(),
            "extra_facts": {k: v for k, v in self.profile.values.items() if k not in self.profile.scenario_view()},
            "current_intent": self.current_intent.value if self.current_intent else None,
            "active_scheme_id": self.active_scheme_id,
            "last_candidates": self.last_candidates,
            "pending_slot": self.pending_slot,
            "missing_information": self.missing_information,
            "turns": self.turns,
        }
