"""
Shared data model of the scheme-intelligence layer.

Nothing here imports numpy, torch or any model - these are plain value
objects passed between IntentEngine, ScenarioEngine, RulesEngine,
RAGEngine, QueryRouter and ResponseComposer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Intent(str, Enum):
    SCHEME_DISCOVERY = "SCHEME_DISCOVERY"
    ELIGIBILITY_CHECK = "ELIGIBILITY_CHECK"
    BENEFIT_INFORMATION = "BENEFIT_INFORMATION"
    APPLICATION_PROCEDURE = "APPLICATION_PROCEDURE"
    REQUIRED_DOCUMENTS = "REQUIRED_DOCUMENTS"
    RATION_ACCESS = "RATION_ACCESS"
    HEALTH_SUPPORT = "HEALTH_SUPPORT"
    WORKER_SUPPORT = "WORKER_SUPPORT"
    PENSION_SUPPORT = "PENSION_SUPPORT"
    MATERNITY_SUPPORT = "MATERNITY_SUPPORT"
    EDUCATION_SUPPORT = "EDUCATION_SUPPORT"
    HOUSING_SUPPORT = "HOUSING_SUPPORT"
    DISABILITY_SUPPORT = "DISABILITY_SUPPORT"
    FINANCIAL_SUPPORT = "FINANCIAL_SUPPORT"
    MIGRANT_SUPPORT = "MIGRANT_SUPPORT"
    APPLICATION_STATUS = "APPLICATION_STATUS"
    GRIEVANCE = "GRIEVANCE"
    DOCUMENT_HELP = "DOCUMENT_HELP"
    GENERAL_SCHEME_INFORMATION = "GENERAL_SCHEME_INFORMATION"
    OTHER = "OTHER"


# Support-area intents <-> the scheme "domains" tagged in the knowledge base.
DOMAIN_OF_INTENT: Dict[Intent, str] = {
    Intent.RATION_ACCESS: "RATION",
    Intent.HEALTH_SUPPORT: "HEALTH",
    Intent.WORKER_SUPPORT: "WORKER",
    Intent.PENSION_SUPPORT: "PENSION",
    Intent.MATERNITY_SUPPORT: "MATERNITY",
    Intent.EDUCATION_SUPPORT: "EDUCATION",
    Intent.HOUSING_SUPPORT: "HOUSING",
    Intent.DISABILITY_SUPPORT: "DISABILITY",
    Intent.FINANCIAL_SUPPORT: "FINANCIAL",
    Intent.MIGRANT_SUPPORT: "MIGRANT",
}
INTENT_OF_DOMAIN: Dict[str, Intent] = {v: k for k, v in DOMAIN_OF_INTENT.items()}


class QType(str, Enum):
    """What kind of answer the question asks for (independent of topic)."""
    OVERVIEW = "OVERVIEW"
    ELIGIBILITY = "ELIGIBILITY"
    BENEFITS = "BENEFITS"
    PROCEDURE = "PROCEDURE"
    DOCUMENTS = "DOCUMENTS"
    CONTACT = "CONTACT"
    FEES = "FEES"
    STATUS = "STATUS"
    GRIEVANCE = "GRIEVANCE"
    DOC_HELP = "DOC_HELP"
    DISCOVERY = "DISCOVERY"
    NONE = "NONE"


class Status(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    LIKELY_ELIGIBLE = "LIKELY_ELIGIBLE"
    POTENTIALLY_ELIGIBLE = "POTENTIALLY_ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class Route(str, Enum):
    DIRECT_INFORMATION = "DIRECT_INFORMATION"
    ELIGIBILITY = "ELIGIBILITY"
    SCHEME_DISCOVERY = "SCHEME_DISCOVERY"
    COMPLEX_SCENARIO = "COMPLEX_SCENARIO"
    VISION = "VISION"
    CLARIFY = "CLARIFY"
    DEFER = "DEFER"  # hand the query back to the legacy pipeline


# The scenario fields the product spec asks for, in spec order.
SCENARIO_FIELDS: Tuple[str, ...] = (
    "age", "gender", "occupation", "employment_status", "monthly_income",
    "annual_income", "family_size", "marital_status", "pregnancy_status",
    "disability_status", "current_state", "current_district", "current_city",
    "origin_state", "origin_district", "migration_status", "ration_card",
    "nfsa_beneficiary", "aadhaar", "e_shram_registration",
    "existing_scheme_membership",
)


class Profile:
    """
    What is known about the person being helped.

    A field that is absent is UNKNOWN - there are no defaults, so missing
    information can never silently satisfy a rule. Numeric fields may carry
    only a bound ("above 60") in `bounds` instead of an exact value.
    """

    __slots__ = ("values", "evidence", "bounds")

    def __init__(self) -> None:
        self.values: Dict[str, Any] = {}
        self.evidence: Dict[str, str] = {}
        self.bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # ── access ──────────────────────────────────────────────────────────
    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.values

    def set(self, name: str, value: Any, evidence: str = "") -> None:
        if value is None:
            return
        if name == "existing_scheme_membership":
            merged = list(self.values.get(name, []))
            for item in (value if isinstance(value, (list, tuple, set)) else [value]):
                if item not in merged:
                    merged.append(item)
            value = merged
        self.values[name] = value
        if evidence:
            self.evidence[name] = evidence[:60]
        self.bounds.pop(name, None)

    def set_bound(self, name: str, lo: Optional[float] = None, hi: Optional[float] = None,
                  evidence: str = "") -> None:
        if name in self.values:
            return
        self.bounds[name] = (lo, hi)
        if evidence:
            self.evidence[name] = evidence[:60]

    def unset(self, name: str) -> None:
        self.values.pop(name, None)
        self.bounds.pop(name, None)
        self.evidence.pop(name, None)

    # ── bulk ────────────────────────────────────────────────────────────
    def merge(self, newer: "Profile") -> List[str]:
        """Takes every fact from `newer` (a correction wins). Returns the changed field names."""
        changed: List[str] = []
        for name, value in newer.values.items():
            if name == "existing_scheme_membership":
                before = list(self.values.get(name, []))
                self.set(name, value, newer.evidence.get(name, ""))
                if self.values.get(name) != before:
                    changed.append(name)
                continue
            if self.values.get(name) != value:
                changed.append(name)
            self.values[name] = value
            self.bounds.pop(name, None)
            if name in newer.evidence:
                self.evidence[name] = newer.evidence[name]
        for name, bound in newer.bounds.items():
            if name not in self.values and self.bounds.get(name) != bound:
                self.bounds[name] = bound
                changed.append(name)
                if name in newer.evidence:
                    self.evidence[name] = newer.evidence[name]
        return changed

    def copy(self) -> "Profile":
        p = Profile()
        p.values = dict(self.values)
        p.evidence = dict(self.evidence)
        p.bounds = dict(self.bounds)
        return p

    def known_fields(self) -> List[str]:
        return list(self.values.keys()) + [b for b in self.bounds if b not in self.values]

    def scenario_view(self) -> Dict[str, Any]:
        """The spec's scenario fields, unknown ones as None."""
        return {f: self.values.get(f) for f in SCENARIO_FIELDS}

    def summary(self, limit: int = 12) -> str:
        items = [f"{k}={v}" for k, v in self.values.items()]
        items += [f"{k}~{b}" for k, b in self.bounds.items() if k not in self.values]
        return ", ".join(items[:limit])

    def __bool__(self) -> bool:
        return bool(self.values or self.bounds)

    def __repr__(self) -> str:
        return f"Profile({self.summary(40)})"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    source: str                      # "rules" | "classifier" | "none"
    qtype: QType = QType.NONE
    domains: List[str] = field(default_factory=list)
    scheme_ids: List[str] = field(default_factory=list)       # explicitly named this turn
    scheme_hints: List[str] = field(default_factory=list)     # weak references ("health card")
    refers_to_context: bool = False  # "this scheme", "it"
    discovery_cue: bool = False      # "which schemes can I get", "I don't know what scheme"
    portability_cue: bool = False    # "here", "another state", "moved"
    has_question: bool = False
    complex_cue: bool = False        # "what happens if", "why", comparisons
    unverified_name: Optional[str] = None  # a real scheme the KB has no verified data for
    matched: List[str] = field(default_factory=list)


@dataclass
class CondResult:
    field: str
    op: str
    expected: Any
    actual: Any
    value: Optional[bool]            # True / False / None (unknown)
    kind: str                        # "core" | "verify" | "exclusion"
    desc: str
    source: str = ""


@dataclass
class SchemeEvaluation:
    scheme_id: str
    status: Status
    passed: List[CondResult] = field(default_factory=list)
    failed: List[CondResult] = field(default_factory=list)
    unknown_core: List[CondResult] = field(default_factory=list)
    unknown_verify: List[CondResult] = field(default_factory=list)
    exclusions_hit: List[CondResult] = field(default_factory=list)
    exclusions_unknown: List[CondResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    relevance: float = 0.0
    rank_score: float = 0.0

    def missing_fields(self, include_verify: bool = True) -> List[str]:
        out: List[str] = []
        pools = [self.unknown_core] + ([self.unknown_verify] if include_verify else [])
        for pool in pools:
            for c in pool:
                if c.field not in out:
                    out.append(c.field)
        return out


@dataclass
class RouteDecision:
    route: Route
    scheme_id: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SIAnswer:
    voice: str
    top: str
    bottom: str
    route: Route
    intent: Intent
    status: str = ""                 # eligibility status or answer kind, for the QR/audit log
    scheme_id: Optional[str] = None
    legacy_code: Optional[str] = None
    scheme_ids: List[str] = field(default_factory=list)
    severity: str = "INFO"
    used_llm: bool = False
    asked_slot: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)
