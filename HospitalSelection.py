"""
Emergency Hospital Selection System
Multi-Criteria Decision Evaluation with A* based cost function.

Cost function:  f(n) = g(n) + h(n)
 - g(n) = Travel time (ETA in minutes) from patient location to hospital.
 - h(n) = Capability deficiency penalty (soft-constraint scoring).

Hard constraints filter out hospitals that cannot satisfy mandatory
clinical requirements. Soft constraints apply weighted penalties for
missing optional capabilities, scaled by patient severity.

Architecture:
  - Pure data layer   → dataclasses & typed dictionaries
  - Pure logic layer  → functions that return result objects (no I/O)
  - Presentation layer → CLI display functions (all print() calls)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 1. ENUMERATIONS

class Severity(Enum):
    """Patient acuity level."""
    MODERATE = "Moderate"
    CRITICAL = "Critical"


class EDStatus(Enum):
    """Emergency Department operational status."""
    OPEN = "Open"
    FULL = "Full"


# ════════════════════════════════════════════════════════════════
# 2. DATA MODELS
# ════════════════════════════════════════════════════════════════

@dataclass
class HospitalStatus:
    """
    Dynamic, real-time operational status of a hospital.

    These values change over time and represent the *current* state
    at the moment of query — not static design-time capabilities.
    """
    ed_status: EDStatus = EDStatus.OPEN
    icu_beds_available: int = 0
    neurosurg_on_call: bool = False
    ventilators_available: int = 0
    or_available: bool = True       # Operating Room availability
    blood_units_available: int = 0


@dataclass
class Hospital:
    """
    Complete hospital record combining static capabilities
    with dynamic real-time status.
    """
    name: str
    # Static capabilities (design-time)
    capabilities: dict[str, bool]
    # Dynamic status (run-time)
    status: HospitalStatus
    # Distance & ETA from patient location
    distance_km: float = 0.0
    eta_minutes: float = 0.0


@dataclass
class RequirementSpec:
    """
    Clinical requirement specification for a given emergency profile.

    Hard constraints are absolute must-haves; a hospital lacking any
    hard requirement is disqualified entirely.

    Soft constraints are beneficial but not mandatory; missing ones
    incur a weighted penalty.
    """
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Evaluation outcome for a single hospital candidate."""
    hospital_name: str
    distance_km: float
    eta_minutes: float
    g_cost: float               # g(n) = weighted travel time
    h_cost: float               # h(n) = capability deficiency penalty
    f_cost: float               # f(n) = g(n) + h(n)
    matched: list[str]          # Requirements the hospital satisfies
    missing_soft: list[str]     # Missing optional capabilities
    disqualified: bool = False
    disqualification_reasons: list[str] = field(default_factory=list)
    icu_beds_available: int = 0
    ed_status: str = "Open"


@dataclass
class SelectionReport:
    """Complete output of the hospital selection process."""
    emergency_type: str
    severity: Severity
    requirements: RequirementSpec
    ranked_results: list[EvaluationResult]
    disqualified: list[EvaluationResult]


# ════════════════════════════════════════════════════════════════
# 3. HOSPITAL DATABASE (Dynamic Status Integrated)
# ════════════════════════════════════════════════════════════════

HOSPITAL_DATABASE: list[Hospital] = [
    Hospital(
        name="RS Panti Rapih",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": True, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=3,
            neurosurg_on_call=True,
            ventilators_available=5,
            or_available=True,
            blood_units_available=20,
        ),
        distance_km=4.2,
        eta_minutes=8,
    ),
    Hospital(
        name="RS JIH Yogyakarta",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": True, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": False,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=1,
            neurosurg_on_call=True,
            ventilators_available=3,
            or_available=True,
            blood_units_available=0,
        ),
        distance_km=9.5,
        eta_minutes=18,
    ),
    Hospital(
        name="RSUP Dr. Sardjito",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=5,
            neurosurg_on_call=False,
            ventilators_available=8,
            or_available=True,
            blood_units_available=40,
        ),
        distance_km=6.8,
        eta_minutes=13,
    ),
]


# 4. PENALTY WEIGHT TABLE (Base Weights)

BASE_PENALTY_WEIGHTS: dict[str, float] = {
    "ED":            10,
    "ICU":           20,
    "CT":            20,
    "Surgery":       20,
    "Neurosurgery":  25,
    "Trauma":        10,
    "Cardiac":       20,
    "Stroke":        20,
    "Ventilator":    25,
    "BloodBank":     15,
}

# Severity multiplier: Critical cases receive much heavier penalties
# for missing capabilities, reflecting the greater clinical risk.
SEVERITY_MULTIPLIER: dict[Severity, float] = {
    Severity.MODERATE: 1.0,
    Severity.CRITICAL: 2.5,
}

# ETA weight multiplier: Critical patients are more time-sensitive.
ETA_WEIGHT: dict[Severity, float] = {
    Severity.MODERATE: 1.0,
    Severity.CRITICAL: 2.0,
}


# 5. REQUIREMENT CONFIGURATION (Data-Driven Mapping Table)
# Replaces the long if/elif chain with a clean dictionary lookup.
# Each key is (emergency_type, severity) → RequirementSpec.
#
# Hard constraints: Hospital is DISQUALIFIED if any are missing.
# Soft constraints: Missing ones add a weighted penalty to f(n).

REQUIREMENT_TABLE: dict[tuple[str, Severity], RequirementSpec] = {
    # ── Major Trauma ─────────────────────────────────────────
    ("Major Trauma", Severity.MODERATE): RequirementSpec(
        hard=["ED", "Trauma"],
        soft=["CT", "Surgery"],
    ),
    ("Major Trauma", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "Surgery", "Trauma"],
        soft=["CT", "BloodBank"],
    ),

    # ── Severe Head Trauma ───────────────────────────────────
    ("Severe Head Trauma", Severity.MODERATE): RequirementSpec(
        hard=["ED", "CT"],
        soft=["Trauma"],
    ),
    ("Severe Head Trauma", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "CT", "Neurosurgery"],
        soft=["Surgery", "Trauma", "BloodBank"],
    ),

    # ── Cardiac Emergency ────────────────────────────────────
    ("Cardiac Emergency", Severity.MODERATE): RequirementSpec(
        hard=["ED", "Cardiac"],
        soft=[],
    ),
    ("Cardiac Emergency", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "Cardiac", "ICU"],
        soft=[],
    ),

    # ── Stroke ───────────────────────────────────────────────
    ("Stroke", Severity.MODERATE): RequirementSpec(
        hard=["ED", "CT", "Stroke"],
        soft=[],
    ),
    ("Stroke", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "CT", "Stroke"],
        soft=[],
    ),

    # ── Severe Respiratory Emergency ─────────────────────────
    ("Severe Respiratory Emergency", Severity.MODERATE): RequirementSpec(
        hard=["ED", "Ventilator"],
        soft=[],
    ),
    ("Severe Respiratory Emergency", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "Ventilator"],
        soft=[],
    ),
}


def get_requirements(emergency_type: str, severity: Severity) -> Optional[RequirementSpec]:
    """
    Look up clinical requirements from the configuration table.

    Returns None if no matching profile exists.
    """
    return REQUIREMENT_TABLE.get((emergency_type, severity))


# 6. DYNAMIC STATUS CHECKS
# These checks translate real-time hospital status into
# effective capability availability.  A hospital may *own*
# a capability but have it functionally unavailable right now.

def get_effective_capability(
    hospital: Hospital,
    requirement: str,
) -> bool:
    """
    Determine whether a requirement is *effectively* available,
    combining static capability with dynamic real-time status.

    Examples:
      - ICU capability exists, but 0 beds available → False
      - ED exists, but ED status is "Full" → False
      - Neurosurgery exists, but no neurosurgeon on call → False
    """
    # Static capability check first
    if not hospital.capabilities.get(requirement, False):
        return False

    # Dynamic availability overrides
    status = hospital.status

    if requirement == "ED" and status.ed_status == EDStatus.FULL:
        return False

    if requirement == "ICU" and status.icu_beds_available <= 0:
        return False

    if requirement == "Neurosurgery" and not status.neurosurg_on_call:
        return False

    if requirement == "Ventilator" and status.ventilators_available <= 0:
        return False

    if requirement == "Surgery" and not status.or_available:
        return False

    if requirement == "BloodBank" and status.blood_units_available <= 0:
        return False

    return True


# 7. CORE EVALUATION ENGINE (Pure Logic — No I/O)


def evaluate_hospital(
    hospital: Hospital,
    requirements: RequirementSpec,
    severity: Severity,
) -> EvaluationResult:
    """
    Evaluate a single hospital against patient requirements.

    Cost model:
      f(n) = g(n) + h(n)
      g(n) = ETA × severity_weight   (travel time cost)
      h(n) = Σ penalty(missing_soft)  (capability deficiency penalty)

    Hard constraints cause immediate disqualification.
    Soft constraints accumulate weighted penalties.

    Returns an EvaluationResult with all scoring details.
    """
    disqualification_reasons: list[str] = []
    matched: list[str] = []
    missing_soft: list[str] = []

    # ── Hard Constraint Check (absolute filter) ──────────────
    for req in requirements.hard:
        if get_effective_capability(hospital, req):
            matched.append(req)
        else:
            disqualification_reasons.append(req)

    if disqualification_reasons:
        return EvaluationResult(
            hospital_name=hospital.name,
            distance_km=hospital.distance_km,
            eta_minutes=hospital.eta_minutes,
            g_cost=math.inf,
            h_cost=math.inf,
            f_cost=math.inf,
            matched=matched,
            missing_soft=[],
            disqualified=True,
            disqualification_reasons=disqualification_reasons,
            icu_beds_available=hospital.status.icu_beds_available,
            ed_status=hospital.status.ed_status.value,
        )

    # ── Soft Constraint Check (penalty accumulation) 
    severity_mult = SEVERITY_MULTIPLIER[severity]

    h_cost = 0.0
    for req in requirements.soft:
        if get_effective_capability(hospital, req):
            matched.append(req)
        else:
            missing_soft.append(req)
            base_weight = BASE_PENALTY_WEIGHTS.get(req, 10)
            h_cost += base_weight * severity_mult

    # ── g(n): Weighted Travel Time
    eta_weight = ETA_WEIGHT[severity]
    g_cost = hospital.eta_minutes * eta_weight

    # ── f(n) = g(n) + h(n) 
    f_cost = g_cost + h_cost

    return EvaluationResult(
        hospital_name=hospital.name,
        distance_km=hospital.distance_km,
        eta_minutes=hospital.eta_minutes,
        g_cost=round(g_cost, 2),
        h_cost=round(h_cost, 2),
        f_cost=round(f_cost, 2),
        matched=matched,
        missing_soft=missing_soft,
        disqualified=False,
        icu_beds_available=hospital.status.icu_beds_available,
        ed_status=hospital.status.ed_status.value,
    )


def select_hospitals(
    emergency_type: str,
    severity: Severity,
    hospital_db: list[Hospital] | None = None,
    top_n: int = 3,
) -> SelectionReport:
    """
    Evaluate all hospitals and return a ranked Top-N selection report.

    Process:
      1. Look up requirements from the configuration table.
      2. Evaluate every hospital (hard filter + soft scoring).
      3. Rank qualified hospitals by f(n) ascending.
      4. Tie-breaking order:
         a) Lowest f(n)
         b) Shortest ETA (fastest arrival)
         c) Most ICU beds available
         d) Hospital name (alphabetical, last resort)

    Returns a SelectionReport with ranked and disqualified lists.
    """
    if hospital_db is None:
        hospital_db = HOSPITAL_DATABASE

    requirements = get_requirements(emergency_type, severity)
    if requirements is None:
        return SelectionReport(
            emergency_type=emergency_type,
            severity=severity,
            requirements=RequirementSpec(),
            ranked_results=[],
            disqualified=[],
        )

    qualified: list[EvaluationResult] = []
    disqualified: list[EvaluationResult] = []

    for hospital in hospital_db:
        result = evaluate_hospital(hospital, requirements, severity)
        if result.disqualified:
            disqualified.append(result)
        else:
            qualified.append(result)

    # ── Tie-breaking sort
    # Primary:   lowest f(n)
    # Secondary: shortest ETA (time is life in emergencies)
    # Tertiary:  most ICU beds (higher capacity = more resilient)
    # Quaternary: alphabetical name (deterministic fallback)
    qualified.sort(
        key=lambda r: (
            r.f_cost,
            r.eta_minutes,
            -r.icu_beds_available,
            r.hospital_name,
        )
    )

    return SelectionReport(
        emergency_type=emergency_type,
        severity=severity,
        requirements=requirements,
        ranked_results=qualified[:top_n],
        disqualified=disqualified,
    )


# 8. PRESENTATION LAYER (CLI Display — All I/O Here)

DIVIDER = "═" * 70
THIN_DIVIDER = "─" * 70


def display_hospitals(hospital_db: list[Hospital] | None = None) -> None:
    """Display a formatted summary of the hospital database."""
    if hospital_db is None:
        hospital_db = HOSPITAL_DATABASE

    print(f"\n{DIVIDER}")
    print("  HOSPITAL DATABASE — Capabilities & Real-Time Status")
    print(DIVIDER)

    for h in hospital_db:
        print(f"\n  ┌─ {h.name}")
        print(f"  │  Distance: {h.distance_km} km  |  ETA: {h.eta_minutes} min")
        print(f"  │  ED Status: {h.status.ed_status.value}  |  "
              f"ICU Beds: {h.status.icu_beds_available}  |  "
              f"Ventilators: {h.status.ventilators_available}")
        print(f"  │  Neurosurgeon On-Call: {'Yes' if h.status.neurosurg_on_call else 'No'}  |  "
              f"OR Available: {'Yes' if h.status.or_available else 'No'}  |  "
              f"Blood Units: {h.status.blood_units_available}")
        print(f"  │")
        print(f"  │  Capabilities:")
        for cap, available in h.capabilities.items():
            symbol = "✓" if available else "✗"
            effective = get_effective_capability(h, cap)
            live_note = ""
            if available and not effective:
                live_note = "  ⚠ currently unavailable"
            print(f"  │    {cap:<15} {symbol}{live_note}")
        print(f"  └{'─' * 50}")


def display_requirements(requirements: RequirementSpec) -> None:
    """Display the clinical requirement breakdown."""
    print(f"\n{DIVIDER}")
    print("  PATIENT REQUIREMENTS")
    print(DIVIDER)

    if requirements.hard:
        print("\n  Hard Constraints (MUST-HAVE — hospital disqualified if missing):")
        for i, req in enumerate(requirements.hard, 1):
            print(f"    {i}. {req}")

    if requirements.soft:
        print("\n  Soft Constraints (NICE-TO-HAVE — penalty if missing):")
        for i, req in enumerate(requirements.soft, 1):
            print(f"    {i}. {req}")


def display_evaluation_detail(result: EvaluationResult, rank: int) -> None:
    """Display detailed evaluation for a single hospital."""
    if result.disqualified:
        print(f"\n  ✗ {result.hospital_name}  [DISQUALIFIED]")
        print(f"    Missing mandatory: {', '.join(result.disqualification_reasons)}")
        return

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    print(f"\n  {medal}  Rank #{rank}: {result.hospital_name}")
    print(f"  {THIN_DIVIDER}")
    print(f"    Distance     : {result.distance_km} km")
    print(f"    ETA          : {result.eta_minutes} min")
    print(f"    ED Status    : {result.ed_status}")
    print(f"    ICU Beds     : {result.icu_beds_available}")
    print(f"    g(n) Travel  : {result.g_cost}")
    print(f"    h(n) Penalty : {result.h_cost}")
    print(f"    f(n) Total   : {result.f_cost}")
    print(f"    Matched      : {', '.join(result.matched) if result.matched else '—'}")
    if result.missing_soft:
        print(f"    Missing (soft): {', '.join(result.missing_soft)}")


def display_report(report: SelectionReport) -> None:
    """Display the complete selection report."""
    print(f"\n{'━' * 70}")
    print("  EMERGENCY HOSPITAL SELECTION — RESULTS")
    print(f"{'━' * 70}")
    print(f"\n  Emergency Type : {report.emergency_type}")
    print(f"  Severity       : {report.severity.value}")

    sev_mult = SEVERITY_MULTIPLIER[report.severity]
    eta_w = ETA_WEIGHT[report.severity]
    print(f"\n  Scoring Parameters:")
    print(f"    Severity Penalty Multiplier : ×{sev_mult}")
    print(f"    ETA Weight (time urgency)   : ×{eta_w}")
    print(f"    Cost Function               : f(n) = g(n) + h(n)")
    print(f"      g(n) = ETA × {eta_w}  (travel time cost)")
    print(f"      h(n) = Σ penalty(missing) × {sev_mult}  (deficiency penalty)")

    # Ranked candidates
    if report.ranked_results:
        print(f"\n{DIVIDER}")
        print(f"  TOP-{len(report.ranked_results)} RECOMMENDED HOSPITALS")
        print(DIVIDER)
        for rank, result in enumerate(report.ranked_results, 1):
            display_evaluation_detail(result, rank)

        best = report.ranked_results[0]
        print(f"\n{'━' * 70}")
        print(f"  >>> PRIMARY RECOMMENDATION: {best.hospital_name}")
        print(f"      ETA {best.eta_minutes} min  |  "
              f"f(n) = {best.f_cost}  |  "
              f"ED: {best.ed_status}  |  "
              f"ICU Beds: {best.icu_beds_available}")
        print(f"{'━' * 70}")
    else:
        print(f"\n  ⚠  NO SUITABLE HOSPITAL FOUND.")
        print(f"      All candidates were disqualified due to missing")
        print(f"      mandatory capabilities for this emergency profile.")

    # Disqualified hospitals
    if report.disqualified:
        print(f"\n{DIVIDER}")
        print("  DISQUALIFIED HOSPITALS")
        print(DIVIDER)
        for result in report.disqualified:
            display_evaluation_detail(result, rank=0)

# 9. USER INPUT (CLI Interface)

EMERGENCY_OPTIONS: dict[int, str] = {
    1: "Major Trauma",
    2: "Severe Head Trauma",
    3: "Cardiac Emergency",
    4: "Stroke",
    5: "Severe Respiratory Emergency",
}

SEVERITY_OPTIONS: dict[int, Severity] = {
    1: Severity.MODERATE,
    2: Severity.CRITICAL,
}


def get_user_input() -> tuple[str, Severity]:
    """Collect emergency type and severity from the user via CLI."""
    print(f"\n{DIVIDER}")
    print("  EMERGENCY HOSPITAL SELECTION SYSTEM")
    print("  Multi-Criteria Decision Evaluation")
    print(DIVIDER)

    # ── Emergency Type
    print("\n  Emergency Type:")
    for num, name in EMERGENCY_OPTIONS.items():
        print(f"    {num}. {name}")

    while True:
        try:
            choice = int(input("\n  Enter emergency type (1-5): "))
            if choice in EMERGENCY_OPTIONS:
                emergency_type = EMERGENCY_OPTIONS[choice]
                break
            print("  Please enter a number from 1 to 5.")
        except ValueError:
            print("  Please enter a valid number.")

    # ── Severity
    print("\n  Severity:")
    for num, sev in SEVERITY_OPTIONS.items():
        print(f"    {num}. {sev.value}")

    while True:
        try:
            choice = int(input("\n  Enter severity (1-2): "))
            if choice in SEVERITY_OPTIONS:
                severity = SEVERITY_OPTIONS[choice]
                break
            print("  Please enter 1 or 2.")
        except ValueError:
            print("  Please enter a valid number.")

    return emergency_type, severity



# 10. MAIN PROGRAM
def main() -> None:
    """Entry point: collect input → evaluate → display results."""
    # 1. Show hospital database overview
    display_hospitals()

    # 2. Get patient information
    emergency_type, severity = get_user_input()

    # 3. Look up requirements
    requirements = get_requirements(emergency_type, severity)
    if requirements is None:
        print("\n  ⚠  No requirement configuration found for this profile.")
        return

    # 4. Display requirements
    display_requirements(requirements)

    # 5. Run evaluation (pure logic — no side effects)
    report = select_hospitals(
        emergency_type=emergency_type,
        severity=severity,
        top_n=3,
    )

    # 6. Display results
    display_report(report)

if __name__ == "__main__":
    main()
