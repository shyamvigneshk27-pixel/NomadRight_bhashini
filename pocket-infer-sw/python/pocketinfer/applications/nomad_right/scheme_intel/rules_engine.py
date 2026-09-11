"""
RulesEngine - deterministic, LLM-free eligibility evaluation.

Three-valued logic: every condition is True, False or UNKNOWN (the fact was
never stated). An unknown fact never satisfies a condition.

Per scheme:
  INELIGIBLE               a condition is known to fail, or an exclusion applies,
                           or the scheme has ended
  ELIGIBLE                 every condition is known to hold and every exclusion
                           is known not to apply
  LIKELY_ELIGIBLE          all core conditions hold; only verification items
                           (Aadhaar seeding, bank account, "not a PF member"...)
                           are unconfirmed
  POTENTIALLY_ELIGIBLE     nothing fails, but a core condition is unknown, and
                           at least one condition holds or the scheme targets
                           the person's occupation
  INSUFFICIENT_INFORMATION core conditions unknown and nothing relevant known
  UNKNOWN                  the knowledge base has no rules for this scheme
"""

from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

from pocketinfer.applications.nomad_right.scheme_intel.models import (
    CondResult, Profile, SchemeEvaluation, Status,
)

# Occupations the official texts list as unorganised work - PM-SYM
# eligibility (scheme compilation p.3 and p.111: street vendors, agricultural
# workers, construction workers, domestic workers, head loaders, brick kiln
# workers, cobblers, rag pickers, washer-men, rickshaw/auto pullers, landless
# labourers, beedi/handloom/leather workers, carpenters, fishermen) and
# e-Shram (p.3: migrant, construction, gig and platform workers).
UNORGANISED_OCCUPATIONS = frozenset({
    "CONSTRUCTION_WORKER", "DOMESTIC_WORKER", "STREET_VENDOR", "AGRICULTURAL_LABOURER",
    "TRANSPORT_WORKER", "HEAD_LOADER", "GIG_WORKER", "BRICK_KILN_WORKER", "RAG_PICKER",
    "WEAVER", "FISHERMAN", "BEEDI_WORKER", "LEATHER_WORKER", "DAILY_WAGE_WORKER",
})
UNORGANISED_TRADES = frozenset({"COBBLER", "WASHERMAN", "CARPENTER", "MASON"})
ORGANISED_OCCUPATIONS = frozenset({"GOVT_EMPLOYEE"})

# Facts that define WHO a scheme is for. While one of these is unknown, the
# scheme is not presented as a possible match (only asked about).
GATE_FIELDS = frozenset({
    "is_unorganised_worker", "occupation", "trade", "construction_days_12m", "disability_percentage",
    "marital_status", "owns_land", "small_marginal_farmer", "girl_child_under_10", "farmer_type",
    "housing_deprivation", "in_secc_2011", "residence_state", "gender", "is_bpl", "poor_household",
    "ration_card", "breadwinner_age", "area_type", "tribal_forest_dweller", "community_land",
    "insurable_interest", "has_electricity_connection",
})

_MEMBERSHIP_FIELDS = {
    "SCH_PMSYM": "pmsym_member", "SCH_BOCW": "bocw_member", "SCH_NSAP_OA": "ignoaps_member",
    "SCH_PMJAY": "pmjay_member", "SCH_ESHRAM": "eshram_member",
}
_ENDED_STATUSES = {"REPEALED", "CLOSED", "DISCONTINUED", "ENDED"}


def derive(profile: Profile) -> Profile:
    """Adds facts that follow directly (and only) from stated facts."""
    d = profile.copy()
    occ, trade = d.get("occupation"), d.get("trade")
    if not d.has("is_unorganised_worker"):
        if occ in UNORGANISED_OCCUPATIONS or trade in UNORGANISED_TRADES:
            d.set("is_unorganised_worker", True, f"occupation {occ or trade}")
        elif occ in ORGANISED_OCCUPATIONS:
            d.set("is_unorganised_worker", False, f"occupation {occ}")
    for sid in d.get("existing_scheme_membership") or []:
        if sid in _MEMBERSHIP_FIELDS:
            d.set(_MEMBERSHIP_FIELDS[sid], True, "already enrolled")
    if d.has("land_area_ha") and not d.has("small_marginal_farmer"):
        # PM-KMY's own definition: small and marginal = cultivable land up to 2 hectares.
        d.set("small_marginal_farmer", float(d.get("land_area_ha")) <= 2.0, "land area")
    if occ == "FARMER" and d.get("owns_land") is True and not d.has("farmer_type"):
        d.set("farmer_type", "OWNER_FARMER", "farmer with own land")
    # housing_status ("I have no house") is deliberately NOT turned into
    # housing_deprivation: PMAY-G selects from the SECC 2011 / Awaas+ lists,
    # which a spoken statement cannot confirm (it stays a verification item).
    if not d.has("poor_household"):
        if d.get("is_bpl") is True or d.get("ration_card_type") in ("AAY", "PHH", "BPL"):
            d.set("poor_household", True, "BPL / AAY / PHH card")
        elif d.get("is_bpl") is False:
            d.set("poor_household", False, "above poverty line")
    ages = d.get("girl_child_ages")
    if ages:
        d.set("girl_child_under_10", any(a < 10 for a in ages), "daughter's age")
    if d.get("govt_employee") is True or d.get("govt_employee_in_family") is True:
        d.set("govt_employee_or_family", True, "government employee")
    return d


def _num(x: Any) -> float:
    if isinstance(x, bool):
        raise TypeError("bool is not a number here")
    return float(x)


def _compare(actual: Any, op: str, expected: Any) -> Optional[bool]:
    try:
        if op in ("==", "="):
            if isinstance(expected, bool):
                return (actual is expected) if isinstance(actual, bool) else None
            if isinstance(expected, str):
                return str(actual).strip().upper() == expected.strip().upper()
            return _num(actual) == _num(expected)
        if op == "!=":
            if isinstance(expected, bool):
                return (actual is not expected) if isinstance(actual, bool) else None
            if isinstance(expected, str):
                return str(actual).strip().upper() != expected.strip().upper()
            return _num(actual) != _num(expected)
        if op in (">=", "<=", ">", "<"):
            a, e = _num(actual), _num(expected)
            return {">=": a >= e, "<=": a <= e, ">": a > e, "<": a < e}[op]
        if op in ("IN", "NOT_IN", "NOT IN"):
            pool = {str(v).upper() for v in expected}
            hit = str(actual).upper() in pool
            return hit if op == "IN" else not hit
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _compare_bounds(bounds: Tuple[Optional[float], Optional[float]], op: str, expected: Any) -> Optional[bool]:
    lo, hi = bounds
    try:
        e = _num(expected)
    except (TypeError, ValueError):
        return None
    if op == ">=":
        if lo is not None and lo >= e:
            return True
        if hi is not None and hi < e:
            return False
    elif op == ">":
        if lo is not None and lo > e:
            return True
        if hi is not None and hi <= e:
            return False
    elif op == "<=":
        if hi is not None and hi <= e:
            return True
        if lo is not None and lo > e:
            return False
    elif op == "<":
        if hi is not None and hi < e:
            return True
        if lo is not None and lo >= e:
            return False
    return None


class RulesEngine:
    def __init__(self, repo, verify_fields: Optional[List[str]] = None,
                 today: Callable[[], date] = date.today):
        self.repo = repo
        self.verify_fields = frozenset(verify_fields or repo.meta.get("verify_fields") or [])
        self._today = today

    # ── leaf / group evaluation ──────────────────────────────────────────
    def _leaf(self, node: Dict[str, Any], d: Profile, kind_default: str) -> CondResult:
        field, op, expected = node["field"], node["op"], node.get("value")
        kind = node.get("kind") or ("verify" if field in self.verify_fields else kind_default)
        soft = node.get("soft_if_occupation")
        if soft and d.get("occupation") in soft and kind == "core":
            kind = "verify"
        if field == "residence_state":
            origin, current = d.get("origin_state"), d.get("current_state")
            want = str(expected).upper()
            if (origin and origin.upper() == want) or (current and current.upper() == want):
                value, actual = True, origin or current
            elif origin and current:
                value, actual = False, f"{origin}/{current}"
            else:
                value, actual = None, origin or current
        elif d.has(field):
            actual = d.get(field)
            value = _compare(actual, op, expected)
        elif field in d.bounds:
            actual = d.bounds[field]
            value = _compare_bounds(d.bounds[field], op, expected)
        else:
            actual, value = None, None
        return CondResult(field=field, op=op, expected=expected, actual=actual, value=value,
                          kind=kind, desc=node.get("desc", field), source=node.get("source", ""))

    def _node(self, node: Dict[str, Any], d: Profile, kind_default: str) -> CondResult:
        if "items" not in node:
            return self._leaf(node, d, kind_default)
        children = [self._node(c, d, kind_default) for c in node["items"]]
        vals = [c.value for c in children]
        if node.get("op", "AND").upper() == "OR":
            value = True if any(v is True for v in vals) else (False if all(v is False for v in vals) else None)
        else:
            value = False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
        if node.get("kind"):
            kind = node["kind"]
        elif kind_default == "exclusion":
            kind = "exclusion"
        else:
            kind = "verify" if all(c.kind == "verify" for c in children) else "core"
        fields = "|".join(sorted({c.field for c in children}))
        desc = node.get("desc") or " or ".join(c.desc for c in children[:2])
        return CondResult(field=fields, op=node.get("op", "AND"), expected=None,
                          actual=[c.actual for c in children], value=value, kind=kind, desc=desc,
                          source=node.get("source", ""))

    def _valid_now(self, rule: Dict[str, Any], today_iso: str) -> bool:
        vf, vu = rule.get("valid_from"), rule.get("valid_until")
        return (not vf or vf <= today_iso) and (not vu or vu >= today_iso)

    # ── scheme evaluation ─────────────────────────────────────────────────
    def evaluate(self, sid: str, profile: Profile, derived: Optional[Profile] = None) -> SchemeEvaluation:
        d = derived if derived is not None else derive(profile)
        ev = SchemeEvaluation(scheme_id=sid, status=Status.UNKNOWN)
        scheme = self.repo.scheme(sid)
        today_iso = self._today().isoformat()
        status = (scheme.get("status") or "").upper()
        until = scheme.get("effective_until")
        if status in _ENDED_STATUSES or (until and until < today_iso and status != "ACTIVE"):
            ev.status = Status.INELIGIBLE
            ev.notes.append("ENDED")
            if scheme.get("superseded_by"):
                ev.notes.append(f"REPLACED_BY:{scheme['superseded_by']}")
            return ev
        if until and until < today_iso:
            ev.notes.append("VALIDITY_CHECK")

        rules = [r for r in self.repo.rules(sid) if self._valid_now(r, today_iso)]
        quals = [r for r in rules if r["rule_type"] == "QUALIFICATION"]
        excls = [r for r in rules if r["rule_type"] == "EXCLUSION"]
        if not quals and not excls:
            return ev

        for rule in quals:
            logic = rule["logic"]
            tops = logic["items"] if logic.get("op", "AND").upper() == "AND" and "items" in logic else [logic]
            for node in tops:
                cr = self._node(node, d, "core")
                if cr.value is True:
                    ev.passed.append(cr)
                elif cr.value is False:
                    ev.failed.append(cr)
                elif cr.kind == "core":
                    ev.unknown_core.append(cr)
                else:
                    ev.unknown_verify.append(cr)

        for rule in excls:
            unless = rule.get("unless_logic")
            if unless and self._node(unless, d, "core").value is True:
                continue
            logic = rule["logic"]
            tops = logic["items"] if logic.get("op", "OR").upper() == "OR" and "items" in logic else [logic]
            for node in tops:
                cr = self._node(node, d, "exclusion")
                cr.kind = "exclusion"
                if cr.value is True:
                    ev.exclusions_hit.append(cr)
                elif cr.value is None:
                    ev.exclusions_unknown.append(cr)

        targeted = self.targets_person(sid, d)
        if ev.failed or ev.exclusions_hit:
            ev.status = Status.INELIGIBLE
        elif ev.unknown_core:
            # Unknown "who is this scheme for" facts (disability, widowhood, land,
            # construction work ...) mean nothing relevant is known yet.
            gate_unknown = any(f in GATE_FIELDS for c in ev.unknown_core for f in c.field.split("|"))
            relevant = targeted or (bool(ev.passed) and not gate_unknown)
            ev.status = Status.POTENTIALLY_ELIGIBLE if relevant else Status.INSUFFICIENT_INFORMATION
        elif not ev.passed:
            # Nothing confirmed at all (e.g. only verification items exist) - never "likely".
            ev.status = Status.POTENTIALLY_ELIGIBLE if targeted else Status.INSUFFICIENT_INFORMATION
        elif ev.unknown_verify or ev.exclusions_unknown:
            ev.status = Status.LIKELY_ELIGIBLE
        else:
            ev.status = Status.ELIGIBLE
        return ev

    def evaluate_many(self, sids: List[str], profile: Profile) -> List[SchemeEvaluation]:
        d = derive(profile)
        return [self.evaluate(sid, profile, derived=d) for sid in sids]

    def targets_person(self, sid: str, d: Profile) -> bool:
        targets = self.repo.scheme(sid).get("occupation") or []
        if not targets:
            return False
        if d.get("occupation") in targets or d.get("trade") in targets:
            return True
        return "UNORGANISED" in targets and d.get("is_unorganised_worker") is True
