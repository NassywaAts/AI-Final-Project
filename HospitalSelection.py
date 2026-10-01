"""
Emergency Hospital Selection System
A* Search Algorithm for Clinical Requirement Matching

Cost function:  f(n) = g(n) + h(n)
  - g(n) = Accumulated penalty / mismatch cost of already evaluated requirements.
  - h(n) = Minimum remaining mismatch cost for unevaluated requirements (admissible heuristic).
  - f(n) = Total estimated mismatch cost used by A* to prioritize states with the lowest mismatch.

Hard constraints disqualify hospitals that lack mandatory clinical capabilities (g(n) = inf).
Soft constraints add weighted penalties for missing optional capabilities, scaled by patient severity.

Architecture:
  - Pure data layer    -> dataclasses & typed enumerations
  - A* Search engine   -> pure logic layer (heapq / state-based search, no I/O)
  - Presentation layer -> CLI display functions (all print() calls)
"""

from __future__ import annotations

import heapq
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


# 2. DATA MODELS

@dataclass
class HospitalStatus:
    """
    Dynamic, real-time operational status of a hospital.
    Represents the live operational state at query time.
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
    region: str
    capabilities: dict[str, bool]
    status: HospitalStatus


@dataclass
class RequirementSpec:
    """
    Clinical requirement specification for a given emergency profile.

    Hard constraints are absolute must-haves; a hospital lacking any
    hard requirement is disqualified entirely (g(n) = inf).

    Soft constraints are beneficial but not mandatory; missing ones
    incur a weighted penalty accumulated into g(n).
    """
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)

    @property
    def all_requirements(self) -> list[tuple[str, bool]]:
        """
        Returns an ordered list of all requirements:
        [(name, is_hard), ...] with hard constraints evaluated first.
        """
        return [(r, True) for r in self.hard] + [(r, False) for r in self.soft]


@dataclass
class EvaluationResult:
    """Evaluation outcome for a single hospital candidate produced by A* search."""
    hospital_name: str
    region: str
    g_cost: float               # g(n) = penalty already evaluated
    h_cost: float               # h(n) = remaining heuristic mismatch cost
    f_cost: float               # f(n) = g(n) + h(n)
    matched: list[str]          # Requirements the hospital satisfies
    missing_soft: list[str]     # Missing optional capabilities
    disqualified: bool = False
    disqualification_reasons: list[str] = field(default_factory=list)
    icu_beds_available: int = 0
    ventilators_available: int = 0
    ed_status: str = "Open"


@dataclass
class SelectionReport:
    """Complete output of the hospital selection process."""
    emergency_type: str
    severity: Severity
    requirements: RequirementSpec
    ranked_results: list[EvaluationResult]
    disqualified: list[EvaluationResult]


# 3. HOSPITAL DATABASE (8 Hospitals across D.I. Yogyakarta)

HOSPITAL_DATABASE: list[Hospital] = [

    # ── Kota Yogyakarta ──────────────────────────────────────
    Hospital(
        name="RSUD Kota Yogyakarta",
        region="Kota Yogyakarta",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=2,
            neurosurg_on_call=False,
            ventilators_available=4,
            or_available=True,
            blood_units_available=15,
        ),
    ),
    Hospital(
        name="RS Panti Rapih",
        region="Kota Yogyakarta",
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
    ),

    # ── Sleman ───────────────────────────────────────────────
    Hospital(
        name="RSUP Dr. Sardjito",
        region="Sleman",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": True, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=8,
            neurosurg_on_call=True,
            ventilators_available=12,
            or_available=True,
            blood_units_available=50,
        ),
    ),
    Hospital(
        name="RS JIH Yogyakarta",
        region="Sleman",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": True, "Trauma": True,
            "Cardiac": True, "Stroke": True,
            "Ventilator": True, "BloodBank": False,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=2,
            neurosurg_on_call=True,
            ventilators_available=4,
            or_available=True,
            blood_units_available=0,
        ),
    ),

    # ── Bantul ───────────────────────────────────────────────
    Hospital(
        name="RSUD Panembahan Senopati",
        region="Bantul",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": False, "Stroke": True,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=2,
            neurosurg_on_call=False,
            ventilators_available=3,
            or_available=True,
            blood_units_available=10,
        ),
    ),
    Hospital(
        name="RSPAU dr. S. Hardjolukito",
        region="Bantul",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": True, "Stroke": False,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=3,
            neurosurg_on_call=False,
            ventilators_available=4,
            or_available=True,
            blood_units_available=18,
        ),
    ),

    # ── Kulon Progo ──────────────────────────────────────────
    Hospital(
        name="RSUD Wates",
        region="Kulon Progo",
        capabilities={
            "ED": True, "ICU": True, "CT": True,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": False, "Stroke": False,
            "Ventilator": True, "BloodBank": True,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=1,
            neurosurg_on_call=False,
            ventilators_available=2,
            or_available=True,
            blood_units_available=8,
        ),
    ),

    # ── Gunungkidul ──────────────────────────────────────────
    Hospital(
        name="RSUD Wonosari",
        region="Gunungkidul",
        capabilities={
            "ED": True, "ICU": True, "CT": False,
            "Surgery": True, "Neurosurgery": False, "Trauma": True,
            "Cardiac": False, "Stroke": False,
            "Ventilator": True, "BloodBank": False,
        },
        status=HospitalStatus(
            ed_status=EDStatus.OPEN,
            icu_beds_available=1,
            neurosurg_on_call=False,
            ventilators_available=2,
            or_available=True,
            blood_units_available=0,
        ),
    ),
]


# ════════════════════════════════════════════════════════════════
# 4. PENALTY WEIGHT TABLE
# ════════════════════════════════════════════════════════════════

BASE_PENALTY_WEIGHTS: dict[str, float] = {
    "ED":            10.0,
    "ICU":           20.0,
    "CT":            20.0,
    "Surgery":       20.0,
    "Neurosurgery":  25.0,
    "Trauma":        10.0,
    "Cardiac":       20.0,
    "Stroke":        20.0,
    "Ventilator":    25.0,
    "BloodBank":     15.0,
}

# Severity multiplier: Critical cases receive heavier penalties for missing capabilities
SEVERITY_MULTIPLIER: dict[Severity, float] = {
    Severity.MODERATE: 1.0,
    Severity.CRITICAL: 2.5,
}


# 5. REQUIREMENT CONFIGURATION TABLE

REQUIREMENT_TABLE: dict[tuple[str, Severity], RequirementSpec] = {
    # ── Major Trauma
    ("Major Trauma", Severity.MODERATE): RequirementSpec(
        hard=["ED", "Trauma"],
        soft=["CT", "Surgery"],
    ),
    ("Major Trauma", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "Surgery", "Trauma"],
        soft=["CT", "BloodBank"],
    ),

    # ── Severe Head Trauma
    ("Severe Head Trauma", Severity.MODERATE): RequirementSpec(
        hard=["ED", "CT"],
        soft=["Trauma"],
    ),
    ("Severe Head Trauma", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "CT", "Neurosurgery"],
        soft=["Surgery", "Trauma", "BloodBank"],
    ),

    # ── Cardiac Emergency
    ("Cardiac Emergency", Severity.MODERATE): RequirementSpec(
        hard=["ED", "Cardiac"],
        soft=[],
    ),
    ("Cardiac Emergency", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "Cardiac", "ICU"],
        soft=[],
    ),

    # ── Stroke
    ("Stroke", Severity.MODERATE): RequirementSpec(
        hard=["ED", "CT", "Stroke"],
        soft=[],
    ),
    ("Stroke", Severity.CRITICAL): RequirementSpec(
        hard=["ED", "ICU", "CT", "Stroke"],
        soft=[],
    ),

    # ── Severe Respiratory Emergency
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
    """Look up clinical requirements from the configuration table."""
    return REQUIREMENT_TABLE.get((emergency_type, severity))


# 6. DYNAMIC STATUS CHECKS

def get_effective_capability(
    hospital: Hospital,
    requirement: str,
) -> bool:
    """
    Determine whether a requirement is effectively available,
    combining static capability with dynamic real-time status.
    """
    # Static capability check
    if not hospital.capabilities.get(requirement, False):
        return False

    # Dynamic status overrides
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

# 7. A* SEARCH EVALUATION ENGINE (Pure Logic — No I/O)

@dataclass(order=True)
class AStarNode:
    """
    Search node representing a state in the evaluation process.

    Ordering is based on (f_cost, -icu_beds, hospital_name) to allow
    heapq to prioritize states with the lowest estimated mismatch cost.
    """
    f_cost: float
    icu_beds: int
    step: int
    hospital: Hospital = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    matched: list[str] = field(default_factory=list, compare=False)
    missing_soft: list[str] = field(default_factory=list, compare=False)
    disqualified: bool = field(default=False, compare=False)
    disqualification_reasons: list[str] = field(default_factory=list, compare=False)


def calculate_heuristic(
    remaining_requirements: list[tuple[str, bool]],
    severity: Severity,
) -> float:
    """
    h(n): Minimum remaining mismatch cost for unevaluated requirements.

    Admissible Heuristic:
      Assumes in the best-case that all remaining unevaluated requirements
      might be satisfied by the hospital (penalty = 0).
      Hence, h(n) = 0.0, providing an admissible lower bound on remaining mismatch.
    """
    return 0.0


def a_star_evaluate_hospital(
    hospital: Hospital,
    requirements: RequirementSpec,
    severity: Severity,
) -> EvaluationResult:
    """
    Evaluates a single hospital using an A* state-space evaluation:

      State n = (hospital, step_index)
        - g(n) = Accumulated penalty of already evaluated requirements
        - h(n) = Minimum remaining mismatch cost for unevaluated requirements
        - f(n) = g(n) + h(n) (Estimated total mismatch cost)

    Transitions:
      - At each step i, requirement r_i is evaluated.
      - If r_i is a hard constraint and missing:
          g(n) = inf, hospital is immediately disqualified.
      - If r_i is a soft constraint and missing:
          g(n) += base_weight(r_i) * severity_multiplier
      - If r_i is satisfied:
          g(n) unchanged (cost added = 0)
      - Goal State: All requirements evaluated (step == total_requirements).
    """
    all_reqs = requirements.all_requirements
    total_steps = len(all_reqs)
    severity_mult = SEVERITY_MULTIPLIER[severity]

    # Initial state: step 0, g(0) = 0
    g_cost = 0.0
    matched: list[str] = []
    missing_soft: list[str] = []
    disqualification_reasons: list[str] = []

    for step in range(total_steps):
        req_name, is_hard = all_reqs[step]
        is_satisfied = get_effective_capability(hospital, req_name)

        if is_satisfied:
            matched.append(req_name)
        else:
            if is_hard:
                disqualification_reasons.append(req_name)
                # Hard constraint failure -> infinite penalty
                return EvaluationResult(
                    hospital_name=hospital.name,
                    region=hospital.region,
                    g_cost=math.inf,
                    h_cost=math.inf,
                    f_cost=math.inf,
                    matched=matched,
                    missing_soft=missing_soft,
                    disqualified=True,
                    disqualification_reasons=disqualification_reasons,
                    icu_beds_available=hospital.status.icu_beds_available,
                    ventilators_available=hospital.status.ventilators_available,
                    ed_status=hospital.status.ed_status.value,
                )
            else:
                missing_soft.append(req_name)
                penalty = BASE_PENALTY_WEIGHTS.get(req_name, 10.0) * severity_mult
                g_cost += penalty

    # Goal state reached: all requirements evaluated
    # Remaining unevaluated requirements = 0 -> h(goal) = 0
    h_cost = 0.0
    f_cost = g_cost + h_cost

    return EvaluationResult(
        hospital_name=hospital.name,
        region=hospital.region,
        g_cost=round(g_cost, 2),
        h_cost=round(h_cost, 2),
        f_cost=round(f_cost, 2),
        matched=matched,
        missing_soft=missing_soft,
        disqualified=False,
        icu_beds_available=hospital.status.icu_beds_available,
        ventilators_available=hospital.status.ventilators_available,
        ed_status=hospital.status.ed_status.value,
    )


def select_hospitals(
    emergency_type: str,
    severity: Severity,
    hospital_db: list[Hospital] | None = None,
    top_n: int = 3,
) -> SelectionReport:
    """
    Evaluates all hospitals via A* mismatch scoring and ranks the Top-N candidates.

    Tie-breaking hierarchy:
      1. Lowest f(n) (minimal total mismatch penalty)
      2. Most ICU beds available (highest clinical capacity)
      3. Most ventilators available (respiratory backup capacity)
      4. Alphabetical name (deterministic fallback)
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
        result = a_star_evaluate_hospital(hospital, requirements, severity)
        if result.disqualified:
            disqualified.append(result)
        else:
            qualified.append(result)

    # Rank qualified candidates: lowest f_cost, highest ICU beds, highest ventilators, alphabetical
    qualified.sort(
        key=lambda r: (
            r.f_cost,
            -r.icu_beds_available,
            -r.ventilators_available,
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
        print(f"\n  ┌─ {h.name}  [{h.region}]")
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
    """Display detailed A* evaluation for a single hospital."""
    if result.disqualified:
        print(f"\n  ✗ {result.hospital_name}  [{result.region}]  [DISQUALIFIED]")
        print(f"    Missing mandatory: {', '.join(result.disqualification_reasons)}")
        return

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    print(f"\n  {medal}  Rank #{rank}: {result.hospital_name}  [{result.region}]")
    print(f"  {THIN_DIVIDER}")
    print(f"    ED Status               : {result.ed_status}")
    print(f"    ICU Beds Available      : {result.icu_beds_available}")
    print(f"    Ventilators Available   : {result.ventilators_available}")
    print(f"    g(n) Evaluated Penalty  : {result.g_cost}")
    print(f"    h(n) Remaining Heuristic: {result.h_cost}")
    print(f"    f(n) Total Mismatch Cost: {result.f_cost}")
    print(f"    Matched Capabilities    : {', '.join(result.matched) if result.matched else '—'}")
    if result.missing_soft:
        print(f"    Missing Optional (Soft) : {', '.join(result.missing_soft)}")


def display_report(report: SelectionReport) -> None:
    """Display the complete A* selection report."""
    print(f"\n{'━' * 70}")
    print("  EMERGENCY HOSPITAL SELECTION — A* RESULTS")
    print(f"{'━' * 70}")
    print(f"\n  Emergency Type : {report.emergency_type}")
    print(f"  Severity       : {report.severity.value}")

    sev_mult = SEVERITY_MULTIPLIER[report.severity]
    print(f"\n  A* Cost Formulation:")
    print(f"    Severity Penalty Multiplier : ×{sev_mult}")
    print(f"    Cost Function               : f(n) = g(n) + h(n)")
    print(f"      g(n) = Penalty of already evaluated requirements")
    print(f"      h(n) = Minimum remaining mismatch cost for unevaluated requirements")
    print(f"      f(n) = Total estimated mismatch cost")

    # Ranked candidates
    if report.ranked_results:
        print(f"\n{DIVIDER}")
        print(f"  TOP-{len(report.ranked_results)} RECOMMENDED HOSPITALS (Lowest Mismatch Cost)")
        print(DIVIDER)
        for rank, result in enumerate(report.ranked_results, 1):
            display_evaluation_detail(result, rank)

        best = report.ranked_results[0]
        print(f"\n{'━' * 70}")
        print(f"  >>> PRIMARY RECOMMENDATION: {best.hospital_name}  [{best.region}]")
        print(f"      f(n) Mismatch Cost = {best.f_cost}  |  "
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
        print("  DISQUALIFIED HOSPITALS (Hard Constraint Violations)")
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
    print("  A* Clinical Requirement Mismatch Evaluation")
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
    """Entry point: show database -> get input -> evaluate via A* -> display report."""
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

    # 5. Run A* evaluation
    report = select_hospitals(
        emergency_type=emergency_type,
        severity=severity,
        top_n=3,
    )

    # 6. Display results
    display_report(report)


if __name__ == "__main__":
    main()
