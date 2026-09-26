import heapq


# ============================================================
# EMERGENCY HOSPITAL SELECTION USING A*
# ============================================================


# ============================================================
# 1. HOSPITAL DATABASE
# ============================================================
# True  = capability available
# False = capability unavailable
#
# NOTE:
# These are DEMO/SIMULATION values.
# Replace them with verified hospital data later.
# ============================================================

hospitals = {

    "RS Panti Rapih": {
        "ED": True,
        "ICU": True,
        "CT": True,
        "Surgery": True,
        "Neurosurgery": True,
        "Trauma": True,
        "Cardiac": True,
        "Stroke": True,
        "Ventilator": True,
        "BloodBank": True
    },

    "RS JIH Yogyakarta": {
        "ED": True,
        "ICU": True,
        "CT": True,
        "Surgery": True,
        "Neurosurgery": True,
        "Trauma": True,
        "Cardiac": True,
        "Stroke": True,
        "Ventilator": True,
        "BloodBank": False
    },

    "RSUP Dr. Sardjito": {
        "ED": True,
        "ICU": True,
        "CT": True,
        "Surgery": True,
        "Neurosurgery": False,
        "Trauma": True,
        "Cardiac": True,
        "Stroke": True,
        "Ventilator": True,
        "BloodBank": True
    }
}


# ============================================================
# 2. PENALTY WEIGHTS
# ============================================================
# Example values for the prototype.
#
# IMPORTANT:
# These weights must be justified in the final project.
# ============================================================

penalty = {

    "ED": 10,
    "ICU": 20,
    "CT": 20,
    "Surgery": 20,
    "Neurosurgery": 25,
    "Trauma": 10,
    "Cardiac": 20,
    "Stroke": 20,
    "Ventilator": 25,
    "BloodBank": 15
}


# ============================================================
# 3. REQUIREMENT GENERATOR
# ============================================================
# Determines medical requirements based on:
# Emergency Type + Severity
#
# This is a SIMPLIFIED PROJECT MODEL,
# not a clinical decision-making protocol.
# ============================================================

def get_requirements(emergency_type, severity):

    # --------------------------------------------------------
    # 1. MAJOR TRAUMA
    # --------------------------------------------------------

    if emergency_type == "Major Trauma":

        if severity == "Moderate":

            return [
                "ED",
                "CT",
                "Surgery",
                "Trauma"
            ]

        elif severity == "Critical":

            return [
                "ED",
                "ICU",
                "CT",
                "Surgery",
                "Trauma",
                "BloodBank"
            ]


    # --------------------------------------------------------
    # 2. SEVERE HEAD TRAUMA
    # --------------------------------------------------------

    elif emergency_type == "Severe Head Trauma":

        if severity == "Moderate":

            return [
                "ED",
                "CT",
                "Trauma"
            ]

        elif severity == "Critical":

            return [
                "ED",
                "ICU",
                "CT",
                "Surgery",
                "Neurosurgery",
                "Trauma",
                "BloodBank"
            ]


    # --------------------------------------------------------
    # 3. CARDIAC EMERGENCY
    # --------------------------------------------------------

    elif emergency_type == "Cardiac Emergency":

        if severity == "Moderate":

            return [
                "ED",
                "Cardiac"
            ]

        elif severity == "Critical":

            return [
                "ED",
                "Cardiac",
                "ICU"
            ]


    # --------------------------------------------------------
    # 4. STROKE
    # --------------------------------------------------------

    elif emergency_type == "Stroke":

        if severity == "Moderate":

            return [
                "ED",
                "CT",
                "Stroke"
            ]

        elif severity == "Critical":

            return [
                "ED",
                "ICU",
                "CT",
                "Stroke"
            ]


    # --------------------------------------------------------
    # 5. SEVERE RESPIRATORY EMERGENCY
    # --------------------------------------------------------

    elif emergency_type == "Severe Respiratory Emergency":

        if severity == "Moderate":

            return [
                "ED",
                "Ventilator"
            ]

        elif severity == "Critical":

            return [
                "ED",
                "ICU",
                "Ventilator"
            ]


    return []


# ============================================================
# 4. HEURISTIC FUNCTION
# ============================================================

def heuristic(hospital_name, remaining_requirements):

    hospital = hospitals[hospital_name]

    h = 0

    for requirement in remaining_requirements:

        # If the hospital cannot satisfy this requirement,
        # the penalty is unavoidable.

        if not hospital[requirement]:

            h += penalty[requirement]

    return h


# ============================================================
# 5. DISPLAY HOSPITAL DATABASE
# ============================================================

def display_hospitals():

    print("\n")
    print("=" * 70)
    print("HOSPITAL CAPABILITY DATABASE")
    print("=" * 70)

    for hospital_name, capabilities in hospitals.items():

        print(f"\n{hospital_name}")

        for capability, available in capabilities.items():

            symbol = "✓" if available else "X"

            print(
                f"   {capability:<15} : {symbol}"
            )


# ============================================================
# 6. DISPLAY REQUIREMENTS
# ============================================================

def display_requirements(requirements):

    print("\n")
    print("=" * 70)
    print("PATIENT REQUIREMENTS")
    print("=" * 70)

    for i, requirement in enumerate(requirements, start=1):

        print(
            f"{i}. {requirement}"
        )


# ============================================================
# 7. A* SEARCH
# ============================================================

def a_star_hospital_selection(requirements):

    # --------------------------------------------------------
    # Priority Queue
    #
    # Each item:
    #
    # (
    #   f,
    #   g,
    #   hospital,
    #   checked_requirements,
    #   remaining_requirements
    # )
    # --------------------------------------------------------

    open_list = []

    closed = set()


    # --------------------------------------------------------
    # INITIAL STATES
    # --------------------------------------------------------
    # One branch for each candidate hospital.
    # --------------------------------------------------------

    for hospital_name in hospitals:

        remaining = tuple(requirements)

        g = 0

        h = heuristic(
            hospital_name,
            remaining
        )

        f = g + h

        heapq.heappush(
            open_list,
            (
                f,
                g,
                hospital_name,
                tuple(),
                remaining
            )
        )


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    step = 1

    while open_list:

        (
            f,
            g,
            hospital_name,
            checked,
            remaining
        ) = heapq.heappop(open_list)


        # ----------------------------------------------------
        # STATE IDENTIFIER
        # ----------------------------------------------------

        state = (
            hospital_name,
            checked,
            remaining
        )


        # ----------------------------------------------------
        # SKIP VISITED STATE
        # ----------------------------------------------------

        if state in closed:

            continue

        closed.add(state)


        # ----------------------------------------------------
        # DISPLAY CURRENT NODE
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)

        print(
            f"STEP {step}"
        )

        print(
            f"Hospital : {hospital_name}"
        )

        print(
            f"g(n)     : {g}"
        )

        print(
            f"h(n)     : {f - g}"
        )

        print(
            f"f(n)     : {f}"
        )

        step += 1


        # ----------------------------------------------------
        # GOAL TEST
        # ----------------------------------------------------

        if len(remaining) == 0:

            return {
                "hospital": hospital_name,
                "g": g,
                "h": 0,
                "f": f
            }


        # ----------------------------------------------------
        # SELECT NEXT REQUIREMENT
        # ----------------------------------------------------

        current_requirement = remaining[0]

        new_remaining = remaining[1:]

        new_checked = (
            checked +
            (current_requirement,)
        )


        # ----------------------------------------------------
        # CHECK HOSPITAL CAPABILITY
        # ----------------------------------------------------

        capability_available = hospitals[
            hospital_name
        ][current_requirement]


        # ----------------------------------------------------
        # CALCULATE TRANSITION COST
        # ----------------------------------------------------

        if capability_available:

            cost = 0

            print(
                f"Checking {current_requirement}: ✓ AVAILABLE"
            )

            print(
                "Transition cost = 0"
            )

        else:

            cost = penalty[
                current_requirement
            ]

            print(
                f"Checking {current_requirement}: X NOT AVAILABLE"
            )

            print(
                f"Mismatch penalty = {cost}"
            )


        # ----------------------------------------------------
        # NEW G COST
        # ----------------------------------------------------

        new_g = g + cost


        # ----------------------------------------------------
        # NEW H COST
        # ----------------------------------------------------

        new_h = heuristic(
            hospital_name,
            new_remaining
        )


        # ----------------------------------------------------
        # NEW F COST
        # ----------------------------------------------------

        new_f = new_g + new_h


        print(
            f"New state → g={new_g}, "
            f"h={new_h}, "
            f"f={new_f}"
        )


        # ----------------------------------------------------
        # PUSH CHILD NODE
        # ----------------------------------------------------

        heapq.heappush(
            open_list,
            (
                new_f,
                new_g,
                hospital_name,
                new_checked,
                new_remaining
            )
        )


    # --------------------------------------------------------
    # NO SOLUTION
    # --------------------------------------------------------

    return None


# ============================================================
# 8. DISPLAY FINAL RESULT
# ============================================================

def display_result(
    result,
    emergency_type,
    severity,
    requirements
):

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if result is None:

        print(
            "No suitable hospital candidate found."
        )

        return


    hospital_name = result["hospital"]


    print(
        f"\nEmergency Type : {emergency_type}"
    )

    print(
        f"Severity       : {severity}"
    )

    print(
        f"\nSelected Hospital:"
    )

    print(
        f">>> {hospital_name}"
    )


    print("\n")
    print("A* Evaluation:")

    print(
        f"g(n) = {result['g']}"
    )

    print(
        f"h(n) = {result['h']}"
    )

    print(
        f"f(n) = {result['f']}"
    )


    # --------------------------------------------------------
    # CAPABILITY MATCH
    # --------------------------------------------------------

    hospital = hospitals[
        hospital_name
    ]

    matched = []
    mismatched = []


    for requirement in requirements:

        if hospital[requirement]:

            matched.append(
                requirement
            )

        else:

            mismatched.append(
                requirement
            )


    print("\n")
    print("Capability Evaluation:")

    print(
        f"Matched    : {len(matched)}/{len(requirements)}"
    )

    print(
        f"Mismatched : {len(mismatched)}/{len(requirements)}"
    )


    print("\nAvailable:")

    for item in matched:

        print(
            f"  ✓ {item}"
        )


    if mismatched:

        print("\nUnavailable:")

        for item in mismatched:

            print(
                f"  X {item}"
            )


# ============================================================
# 9. USER INPUT
# ============================================================

def get_user_input():

    print("\n")
    print("=" * 70)
    print("EMERGENCY HOSPITAL SELECTION SYSTEM")
    print("A* SEARCH")
    print("=" * 70)


    # --------------------------------------------------------
    # EMERGENCY TYPE
    # --------------------------------------------------------

    emergency_options = {

        1: "Major Trauma",
        2: "Severe Head Trauma",
        3: "Cardiac Emergency",
        4: "Stroke",
        5: "Severe Respiratory Emergency"
    }


    print("\nEmergency Type:")

    for number, emergency in emergency_options.items():

        print(
            f"{number}. {emergency}"
        )


    while True:

        try:

            choice = int(
                input(
                    "\nEnter emergency type (1-5): "
                )
            )

            if choice in emergency_options:

                emergency_type = emergency_options[
                    choice
                ]

                break

            print(
                "Please enter a number from 1 to 5."
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    severity_options = {

        1: "Moderate",
        2: "Critical"
    }


    print("\nSeverity:")

    for number, severity in severity_options.items():

        print(
            f"{number}. {severity}"
        )


    while True:

        try:

            choice = int(
                input(
                    "\nEnter severity (1-2): "
                )
            )

            if choice in severity_options:

                severity = severity_options[
                    choice
                ]

                break

            print(
                "Please enter 1 or 2."
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )


    return emergency_type, severity


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    # Get patient information
    emergency_type, severity = get_user_input()


    # Generate requirements
    requirements = get_requirements(
        emergency_type,
        severity
    )


    # Check whether requirements exist
    if not requirements:

        print(
            "\nNo requirement configuration found."
        )

        return


    # Display requirements
    display_requirements(
        requirements
    )


    # Run A*
    print("\n")
    print("=" * 70)
    print("STARTING A* SEARCH")
    print("=" * 70)


    result = a_star_hospital_selection(
        requirements
    )


    # Display result
    display_result(
        result,
        emergency_type,
        severity,
        requirements
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()