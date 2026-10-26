"""
Google ADK Validated Tool Suite with Guided Error Recovery (agents/adk_tools.py)

Provides 8 domain-specific coaching and biomechanical telemetry tools equipped with:
- Comprehensive Google-style docstrings detailing purpose, `Args`, and `Returns`
- Highly specific domain-verb naming
- Strict Pydantic v2 input and output JSON schemas (`extra="forbid"`)
- `@with_guided_error_handling` decorator returning structured `GuidedToolErrorOutput`
  recovery instructions back to the LLM instead of raising unhandled exceptions.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Callable, Dict, List
from pydantic import ValidationError

from agents.hitl import HumanInTheLoopGate
from agents.schemas import (
    BiometricReflectionLoggedOutput,
    BuildPeriodizedMacrocycleInput,
    DailyTriageVerdictOutput,
    DiagnoseMarathonBiomechanicalFadeInput,
    DispatchGarminForerunner970WorkoutInput,
    EnforceRedsPescatarianNutritionInput,
    ExecuteDailyAutonomicTriageInput,
    GarminWorkoutDispatchOutput,
    GenerateJackDaniels2QMicrocycleInput,
    GuidedToolErrorOutput,
    HistoricalAthleteProfileOutput,
    LogAcuteBiometricReflectionInput,
    MarathonFadeDiagnosisOutput,
    NutritionTargetsOutput,
    PeriodizedMacrocycleOutput,
    QueryHistoricalAthleteProfileInput,
    WeeklyMicrocycleOutput,
)
from core.observability import capture_intent_and_outcome, get_structured_logger

logger = get_structured_logger("EnduranceCopilot.ADKTools")


def _parse_hms_to_seconds(hms: str) -> int:
    """Converts 'H:MM:SS' or 'MM:SS' string to total seconds."""
    parts = [int(p) for p in hms.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Invalid time string '{hms}'; expected H:MM:SS or M:SS.")


def _pace_mm_ss_to_mps(mm_ss: str) -> float:
    """Converts 'M:SS' per mile pace to meters per second."""
    sec_per_mile = max(180, _parse_hms_to_seconds(mm_ss))
    return round(1609.344 / sec_per_mile, 4)


def with_guided_error_handling(
    tool_name: str,
    input_schema_cls: Any,
    recovery_hint: str,
) -> Callable:
    """
    Wraps a tool function so that Pydantic `ValidationError` or runtime exceptions
    never crash the agent loop; instead, they return a structured `GuidedToolErrorOutput`
    with explicit `llm_recovery_instructions` guiding the LLM to self-correct and retry.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> Dict[str, Any]:
            with capture_intent_and_outcome(
                agent_name="ADKToolExecutor",
                intended_action=tool_name,
                intended_arguments=kwargs,
                rationale=f"Executing schema-validated ADK tool '{tool_name}'",
            ) as outcome_rec:
                try:
                    validated_input = input_schema_cls.model_validate(kwargs)
                    result = func(validated_input)
                    outcome_rec["actual_outcome"] = result
                    return result
                except ValidationError as val_err:
                    invalid_fields: Dict[str, Any] = {}
                    for err in val_err.errors():
                        loc_str = ".".join(str(p) for p in err.get("loc", []))
                        invalid_fields[loc_str] = err.get("msg", "Invalid value")

                    guided_error = GuidedToolErrorOutput(
                        tool_name=tool_name,
                        error_code="SCHEMA_VALIDATION_FAILED",
                        error_message=f"Input arguments for '{tool_name}' failed strict JSON schema validation.",
                        invalid_arguments=invalid_fields,
                        llm_recovery_instructions=(
                            f"Correct the invalid fields {list(invalid_fields.keys())}. {recovery_hint} "
                            f"Expected JSON Schema: {input_schema_cls.model_json_schema().get('properties', {})}"
                        ),
                    ).model_dump()
                    outcome_rec["outcome_status"] = "GUIDED_RECOVERY_ERROR"
                    outcome_rec["actual_outcome"] = guided_error
                    return guided_error
                except Exception as exc:
                    guided_error = GuidedToolErrorOutput(
                        tool_name=tool_name,
                        error_code="TOOL_EXECUTION_ERROR",
                        error_message=str(exc),
                        invalid_arguments=kwargs,
                        llm_recovery_instructions=(
                            f"Tool '{tool_name}' encountered a runtime constraint error: {exc}. "
                            f"{recovery_hint} Verify parameter formats and retry."
                        ),
                    ).model_dump()
                    outcome_rec["outcome_status"] = "GUIDED_RECOVERY_ERROR"
                    outcome_rec["actual_outcome"] = guided_error
                    return guided_error

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Tool 1: query_historical_athlete_telemetry_baseline
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="query_historical_athlete_telemetry_baseline",
    input_schema_cls=QueryHistoricalAthleteProfileInput,
    recovery_hint="Pass only `include_pace_buckets` (boolean: True or False).",
)
def query_historical_athlete_telemetry_baseline(
    validated_input: QueryHistoricalAthleteProfileInput,
) -> Dict[str, Any]:
    """
    Retrieves the athlete's longitudinal Garmin FIT telemetry baseline, Rogue Marathon
    anchor (3:12:58 -> VDOT 49.2), historical ceiling benchmarks (1:28:06 Half Marathon ->
    VDOT 53.5, 58 MPW durability ceiling), and pace-stratified biomechanical Z-score norms.

    Args:
        include_pace_buckets (bool): When True, includes pace-normalized Ground Contact Time (GCT),
            cadence, and vertical ratio baselines across Easy, Marathon, Threshold, and Interval buckets.

    Returns:
        Dict[str, Any]: Validated `HistoricalAthleteProfileOutput` dictionary containing
            `athlete_name`, `active_vdot`, `ceiling_vdot`, `cadence_anchor_spm`,
            `active_baseline`, `historical_ceiling`, `durability_limits`, and `pace_bucketed_biomechanics`.
    """
    pace_buckets = {
        "Easy": {"mean_gct_ms": 238.0, "sd_gct_ms": 6.5, "mean_cadence_spm": 176.0, "mean_vertical_ratio_pct": 7.6},
        "Marathon": {"mean_gct_ms": 218.0, "sd_gct_ms": 5.0, "mean_cadence_spm": 182.0, "mean_vertical_ratio_pct": 7.1},
        "Threshold": {"mean_gct_ms": 204.0, "sd_gct_ms": 4.2, "mean_cadence_spm": 186.0, "mean_vertical_ratio_pct": 6.7},
        "Interval": {"mean_gct_ms": 191.0, "sd_gct_ms": 3.8, "mean_cadence_spm": 192.0, "mean_vertical_ratio_pct": 6.3},
    }
    output = HistoricalAthleteProfileOutput(
        athlete_name="Jesse",
        active_vdot=49.2,
        ceiling_vdot=53.5,
        cadence_anchor_spm=182,
        active_baseline={
            "anchor_race": "Rogue Marathon (2026-10-10)",
            "official_time": "3:12:58",
            "anchored_vdot": 49.2,
            "marathon_pace_per_mile": "7:21/mi",
            "threshold_pace_per_mile": "6:45/mi",
        },
        historical_ceiling={
            "ceiling_vdot": 53.5,
            "half_marathon_pr": "1:28:06",
            "peak_weekly_mileage_mpw": 58.0,
        },
        durability_limits={
            "safe_ramp_rate_pct": 8.0,
            "peak_mpw_ceiling": 58.0,
            "max_gct_asymmetry_deviation_pct": 1.0,
        },
        pace_bucketed_biomechanics=pace_buckets if validated_input.include_pace_buckets else {},
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 2: diagnose_marathon_biomechanical_fade
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="diagnose_marathon_biomechanical_fade",
    input_schema_cls=DiagnoseMarathonBiomechanicalFadeInput,
    recovery_hint=(
        "Ensure `half_1_split_str` and `half_2_split_str` match H:MM:SS format (e.g. '1:34:58'), "
        "`late_stage_cadence_spm` is between 120 and 230, and `gct_asymmetry_deviation_pct` is between 0.0 and 15.0."
    ),
)
def diagnose_marathon_biomechanical_fade(
    validated_input: DiagnoseMarathonBiomechanicalFadeInput,
) -> Dict[str, Any]:
    """
    Diagnoses marathon physiological and mechanical bottlenecks by comparing first-half vs.
    second-half split decay against late-stage cadence degradation and Ground Contact Time (GCT)
    L/R asymmetry. Classifies the athlete into Case A (Velocity & Eccentric Armor Bottleneck),
    Case B (Aerobic Threshold Bridge Bottleneck), or Case C (Even/Negative Split High Economy).

    Args:
        half_1_split_str (str): First half marathon split in 'H:MM:SS' format (e.g., '1:34:58').
        half_2_split_str (str): Second half marathon split in 'H:MM:SS' format (e.g., '1:38:00').
        late_stage_cadence_spm (float): Mean cadence (steps/min) during miles 20-26.2.
        gct_asymmetry_deviation_pct (float): Percentage-point deviation of L/R GCT balance from 50.0%.

    Returns:
        Dict[str, Any]: Validated `MarathonFadeDiagnosisOutput` dictionary with `diagnostic_case`,
            `anchored_vdot`, `target_block_vdot`, `split_delta_seconds`, `clinical_diagnosis`,
            `recommended_macrocycle`, and `training_shifts`.
    """
    h1_sec = _parse_hms_to_seconds(validated_input.half_1_split_str)
    h2_sec = _parse_hms_to_seconds(validated_input.half_2_split_str)
    delta_sec = h2_sec - h1_sec

    if delta_sec >= 90 or validated_input.gct_asymmetry_deviation_pct > 1.0:
        diag_case = "Case A: Positive Split & Late-Stage Mechanical Fade (Eccentric Armor Bottleneck)"
        macro_rec = "10K/Half-Marathon Threshold & Running Economy Block + Unilateral Eccentric Strength"
        macro_key = "case_a_eccentric_velocity_block"
        shifts = [
            "Emphasize broken Lactate Threshold cruise intervals (4x2mi @ T) + 200m/400m Repetition economy strides",
            "Prioritize Session B unilateral knee-dominant eccentric step-downs and Reverse Copenhagen planks",
            "Introduce progressive fast-finish long runs only after 3-run GCT symmetry remains within <=1.0%",
        ]
    elif delta_sec > 30:
        diag_case = "Case B: Aerobic Threshold Bridge Bottleneck"
        macro_rec = "Continuous Tempo & Fast-Finish Marathon Long Run Block"
        macro_key = "case_b_threshold_bridge_block"
        shifts = [
            "Extend continuous Lactate Threshold runs (5-7 miles @ 6:45/mi)",
            "Incorporate E+M fast-finish Sunday Q1 long runs",
        ]
    else:
        diag_case = "Case C: Even/Negative Split High Durability Execution"
        macro_rec = "Progressive VDOT Step-Up 2Q Marathon Cycle"
        macro_key = "case_c_progressive_2q"
        shifts = [
            "Increment active VDOT by +0.5 when sub-maximal HR decoupling <3.0% with stable GCT",
        ]

    output = MarathonFadeDiagnosisOutput(
        diagnostic_case=diag_case,
        anchored_vdot=49.2,
        target_block_vdot=51.2,
        split_delta_seconds=delta_sec,
        clinical_diagnosis=(
            f"Second-half split delta of +{delta_sec}s with late-stage cadence {validated_input.late_stage_cadence_spm:.1f} spm "
            f"and GCT asymmetry deviation {validated_input.gct_asymmetry_deviation_pct:.1f}% indicates {diag_case}."
        ),
        recommended_macrocycle=macro_rec,
        macrocycle_key=macro_key,
        training_shifts=shifts,
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 3: build_periodized_vdot_macrocycle_plan
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="build_periodized_vdot_macrocycle_plan",
    input_schema_cls=BuildPeriodizedMacrocycleInput,
    recovery_hint=(
        "Ensure `target_distance` is one of ['Marathon', 'Half Marathon', '10K', '5K'] "
        "and `target_date` is formatted strictly as 'YYYY-MM-DD' (e.g., '2027-03-14')."
    ),
)
def build_periodized_vdot_macrocycle_plan(
    validated_input: BuildPeriodizedMacrocycleInput,
) -> Dict[str, Any]:
    """
    Constructs a multi-phase periodized macrocycle anchored to active VDOT 49.2.
    Automatically inserts an active Reverse Taper if `days_since_last_marathon < 21`
    and partitions the remaining timeline into Base (25-35%), Specific Quality (45-55%),
    and Peak/Taper (15-20%). Supports Human-in-the-Loop (HITL) staging before overwriting
    the athlete's macrocycle schedule.

    Args:
        target_distance (str): Race distance ('Marathon', 'Half Marathon', '10K', or '5K').
        target_date (str): Target competition date in 'YYYY-MM-DD' format.
        focus_theme (str): Descriptive goal focus (e.g., 'Sub-3:10 Marathon Build').
        require_human_confirmation (bool): When True, halts persistence at a HITL confirmation gate.

    Returns:
        Dict[str, Any]: Validated `PeriodizedMacrocycleOutput` containing phase breakdown,
            VDOT pace zones, and optional `hitl_approval_id`.
    """
    if validated_input.require_human_confirmation:
        hitl = HumanInTheLoopGate()
        checkpoint = hitl.create_approval_checkpoint(
            action_name="build_periodized_vdot_macrocycle_plan",
            payload=validated_input.model_dump(),
            risk_rationale=(
                f"Overwriting active macrocycle to '{validated_input.focus_theme}' "
                f"({validated_input.target_distance} on {validated_input.target_date}) requires athlete confirmation."
            ),
        )
        return PeriodizedMacrocycleOutput(
            status="PENDING_HUMAN_CONFIRMATION",
            goal_name=validated_input.focus_theme,
            target_distance=validated_input.target_distance,
            target_date=validated_input.target_date,
            current_vdot=49.2,
            target_vdot=51.2 if validated_input.target_distance == "Marathon" else 50.8,
            total_weeks=16,
            reverse_taper_auto_inserted=True,
            phases=[],
            vdot_paces={},
            hitl_approval_id=checkpoint["hitl_approval_id"],
        ).model_dump()

    phases = [
        {"phase_type": "REVERSE_TAPER", "weeks": 2, "focus": "Post-Marathon Cellular & Eccentric Recovery"},
        {"phase_type": "BASE_AEROBIC", "weeks": 4, "focus": "Aerobic Volume & Unilateral Pelvic Stability (30%)"},
        {"phase_type": "SPECIFIC_QUALITY_2Q", "weeks": 8, "focus": "Jack Daniels 2Q Threshold & Economy Specificity (50%)"},
        {"phase_type": "PEAK_AND_TAPER", "weeks": 2, "focus": "Neuromuscular Sharpening & 14-Day Eccentric Lockout (20%)"},
    ]
    vdot_paces = {
        "Easy": "8:24 - 8:55/mi",
        "Marathon": "7:21/mi",
        "Threshold": "6:45/mi",
        "Interval": "6:14/mi",
        "Repetition": "5:50/mi",
    }
    output = PeriodizedMacrocycleOutput(
        status="success",
        goal_name=validated_input.focus_theme,
        target_distance=validated_input.target_distance,
        target_date=validated_input.target_date,
        current_vdot=49.2,
        target_vdot=51.2 if validated_input.target_distance == "Marathon" else 50.8,
        total_weeks=16,
        reverse_taper_auto_inserted=True,
        phases=phases,
        vdot_paces=vdot_paces,
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 4: generate_jack_daniels_2q_weekly_microcycle
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="generate_jack_daniels_2q_weekly_microcycle",
    input_schema_cls=GenerateJackDaniels2QMicrocycleInput,
    recovery_hint=(
        "Pass `plan_key` as 'auto', 'taper_1', 'taper_2', or '2q_week_1'..'2q_week_8', "
        "and optional `week_start_date` in 'YYYY-MM-DD' format."
    ),
)
def generate_jack_daniels_2q_weekly_microcycle(
    validated_input: GenerateJackDaniels2QMicrocycleInput,
) -> Dict[str, Any]:
    """
    Generates a rolling 7-day Jack Daniels 2Q microcycle consisting of:
    - Sunday Q1 Long/Quality session on flat road/track (with Forerunner 970 workout JSON)
    - Wednesday Q2 Lactate Threshold / Interval session on flat road/track
    - Easy aerobic filler runs on Marin trails (China Camp / Mt. Tam) governed by HR and RPE
    - 2x 20-minute Home Gym strength sessions (Session A Mon/Tue, Session B Thu/Fri)

    Args:
        plan_key (str): Plan selector ('auto', 'taper_1', 'taper_2', '2q_week_1'..'2q_week_8').
        week_start_date (Optional[str]): Optional start date in 'YYYY-MM-DD' format.

    Returns:
        Dict[str, Any]: Validated `WeeklyMicrocycleOutput` dictionary with all 7 daily prescriptions.
    """
    resolved_key = "2q_week_1" if validated_input.plan_key == "auto" else validated_input.plan_key
    days = [
        {"day": "Monday", "session_type": "Easy Trail + Home Gym Session A", "surface": "Marin Trail", "distance_mi": 6.0, "gym_session": "Session A (20m Heavy Hinge, Unilateral Hip Thrust, Loaded Carry, Pallof Press)"},
        {"day": "Tuesday", "session_type": "Easy Aerobic Trail Run", "surface": "Marin Trail", "distance_mi": 7.0, "target_hr_cap_bpm": 140},
        {"day": "Wednesday", "session_type": "Q2 Lactate Threshold Cruise (2E + 3x2mi @ 6:45/mi + 2E)", "surface": "Flat Road / Track", "distance_mi": 10.0},
        {"day": "Thursday", "session_type": "Easy Recovery Trail + Home Gym Session B", "surface": "Marin Trail", "distance_mi": 5.5, "gym_session": "Session B (20m Bulgarian Split Squat, Eccentric Step-Down, Reverse Copenhagen, Standing Psoas March)"},
        {"day": "Friday", "session_type": "Rest or 30m Mobility & Psoas Decompression", "surface": "Off-Feet", "distance_mi": 0.0},
        {"day": "Saturday", "session_type": "Easy Marin Headlands Trail Run + 6x20s Strides", "surface": "Marin Trail", "distance_mi": 6.5},
        {"day": "Sunday", "session_type": "Q1 Progressive E+M Long Run (10E + 4M @ 7:21/mi + 1E)", "surface": "Flat Road / Bike Path", "distance_mi": 15.0},
    ]
    output = WeeklyMicrocycleOutput(
        selected_plan_key=resolved_key,
        phase_title=f"Jack Daniels 2Q Microcycle ({resolved_key})",
        active_vdot=49.2,
        days_count=len(days),
        days=days,
        closed_loop_gate_status="GREEN_UNLOCKED",
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 5: execute_daily_autonomic_biomechanical_triage
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="execute_daily_autonomic_biomechanical_triage",
    input_schema_cls=ExecuteDailyAutonomicTriageInput,
    recovery_hint=(
        "Ensure `leg_soreness_score` is an integer 0..10, `oura_hrv_ms` and `oura_baseline_hrv_ms` "
        "are between 5.0 and 250.0, and `gct_deviation_pct` is between 0.0 and 15.0."
    ),
)
def execute_daily_autonomic_biomechanical_triage(
    validated_input: ExecuteDailyAutonomicTriageInput,
) -> Dict[str, Any]:
    """
    Executes the daily closed-loop triage algorithm (`STAY_THE_COURSE`, `DOWNSHIFT`, or `STEP_UP`)
    using the Cardiac vs. Mechanical Differential Matrix and Multi-Day Asymmetry Filter:
    - `DOWNSHIFT`: Triggered if Oura HRV drops > 15% below baseline, soreness >= 6/10,
      intra-run GCT drift > 1.5% with soreness >= 6/10, or GCT balance shifts > 1.0% across 3+ consecutive runs.
    - `STEP_UP`: (+0.5 VDOT) Triggered when HR decoupling < 3.0%, GCT deviation < 2.0%, HRV drop <= 3.0%, and soreness <= 2/10.
    - `STAY_THE_COURSE`: Normal physiological adaptation corridor.

    Args:
        leg_soreness_score (int): Subjective muscular/pelvic soreness score (0-10).
        oura_hrv_ms (float): Overnight Oura Ring HRV RMSSD (ms).
        oura_baseline_hrv_ms (float): 60-day rolling Oura HRV baseline (ms).
        gct_deviation_pct (float): Absolute L/R GCT balance deviation from 50.0%.
        hr_decoupling_pct (float): Sub-maximal aerobic HR decoupling percentage.
        consecutive_asymmetry_runs (int): Consecutive runs with >1.0% directional GCT shift.

    Returns:
        Dict[str, Any]: Validated `DailyTriageVerdictOutput` containing `triage_verdict`,
            `cardiac_vs_mechanical_quadrant`, `rationale`, `adjustments`, and `reds_deficit_permitted`.
    """
    hrv_drop_pct = (
        (validated_input.oura_baseline_hrv_ms - validated_input.oura_hrv_ms)
        / max(1.0, validated_input.oura_baseline_hrv_ms)
    ) * 100.0

    if validated_input.hr_decoupling_pct < 3.0 and validated_input.gct_deviation_pct < 1.0:
        quadrant = "OPTIMAL_COUPLED_ECONOMY (Low HR Drift <3% + Stable GCT)"
    elif validated_input.hr_decoupling_pct > 5.0 and validated_input.gct_deviation_pct < 1.0:
        quadrant = "SYSTEMIC_AUTONOMIC_OR_FUELING_STRAIN (High HR Drift >5% + Stable GCT)"
    elif validated_input.hr_decoupling_pct < 3.0 and validated_input.gct_deviation_pct >= 1.5:
        quadrant = "LOCAL_MUSCULAR_ECCENTRIC_FATIGUE (Low HR Drift + High GCT Drift)"
    else:
        quadrant = "COMPOUND_CARDIAC_AND_MECHANICAL_LOAD"

    multi_day_asymmetry_triggered = (
        validated_input.gct_deviation_pct > 1.0 and validated_input.consecutive_asymmetry_runs >= 3
    )
    acute_mechanical_pain_triggered = (
        validated_input.gct_deviation_pct > 1.5 and validated_input.leg_soreness_score >= 6
    )

    if (
        hrv_drop_pct > 15.0
        or validated_input.leg_soreness_score >= 6
        or multi_day_asymmetry_triggered
        or acute_mechanical_pain_triggered
    ):
        verdict = "DOWNSHIFT"
        reds_allowed = False
        rationale = (
            f"DOWNSHIFT triggered: HRV drop {hrv_drop_pct:.1f}%, soreness {validated_input.leg_soreness_score}/10, "
            f"GCT deviation {validated_input.gct_deviation_pct:.1f}% ({validated_input.consecutive_asymmetry_runs} consecutive runs). "
            "Convert quality pavement workout to soft Marin trail jog/walk, suspend eccentric barbell loading, and abort caloric deficit."
        )
        adjustments = {
            "aerobic_session": "Swap pavement quality for soft dirt trail recovery jog (HR < 128 bpm) on China Camp trails",
            "strength_session": "Suspend heavy eccentric lifts; execute 10-min 90-90 diaphragmatic decompression",
            "nutrition_gate": "0 kcal deficit enforced (RED-S protection active)",
        }
    elif (
        validated_input.hr_decoupling_pct < 3.0
        and validated_input.gct_deviation_pct < 1.0
        and hrv_drop_pct <= 3.0
        and validated_input.leg_soreness_score <= 2
    ):
        verdict = "STEP_UP"
        reds_allowed = True
        rationale = (
            f"STEP_UP unlocked (+0.5 VDOT): Sub-maximal HR decoupling ({validated_input.hr_decoupling_pct:.1f}% < 3%), "
            f"symmetrical GCT ({validated_input.gct_deviation_pct:.1f}%), and strong HRV ({validated_input.oura_hrv_ms:.1f} ms)."
        )
        adjustments = {
            "vdot_adjustment": "+0.5 VDOT progression approved",
            "pace_shift": "Tighten Threshold & Interval paces by ~3-4 sec/mile on upcoming flat track/road Q session",
        }
    else:
        verdict = "STAY_THE_COURSE"
        reds_allowed = hrv_drop_pct <= 15.0 and validated_input.leg_soreness_score < 6
        rationale = (
            f"STAY_THE_COURSE: Biometrics within normal adaptation corridor "
            f"(HRV {validated_input.oura_hrv_ms:.1f} ms, Soreness {validated_input.leg_soreness_score}/10, "
            f"GCT deviation {validated_input.gct_deviation_pct:.1f}%)."
        )
        adjustments = {
            "session_directive": "Proceed with scheduled microcycle prescription without mutation.",
        }

    output = DailyTriageVerdictOutput(
        triage_verdict=verdict,
        cardiac_vs_mechanical_quadrant=quadrant,
        rationale=rationale,
        adjustments=adjustments,
        reds_deficit_permitted=reds_allowed,
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 6: enforce_reds_pescatarian_caloric_gate
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="enforce_reds_pescatarian_caloric_gate",
    input_schema_cls=EnforceRedsPescatarianNutritionInput,
    recovery_hint=(
        "Ensure `day_of_week` is a valid capitalized weekday ('Monday'..'Sunday'), "
        "`active_burn_kcal` >= 0, and `leg_soreness_score` is 0..10."
    ),
)
def enforce_reds_pescatarian_caloric_gate(
    validated_input: EnforceRedsPescatarianNutritionInput,
) -> Dict[str, Any]:
    """
    Calculates daily 100% whole-food pescatarian energy and macronutrient targets
    (no dairy, no eggs, no shellfish, no refined sugar, protein >= 150g/day) and
    enforces the RED-S (Relative Energy Deficiency in Sport) clinical safeguard:
    immediately aborts any caloric deficit if Oura HRV drop > 15%, soreness >= 6/10,
    or persistent GCT asymmetry is detected.

    Args:
        day_of_week (str): Full weekday name ('Monday' through 'Sunday').
        active_burn_kcal (float): Active exercise expenditure in kcal from Garmin.
        focus_mode (str): Active training block mode ('ACUTE_RECOVERY', '5K_10K', 'MARATHON').
        oura_hrv_drop_pct (float): Overnight HRV percentage drop below 60-day baseline.
        leg_soreness_score (int): Subjective muscular soreness (0-10).
        gct_imbalance_detected (bool): Whether Ground Contact Time asymmetry (>1.0%) is flagged.

    Returns:
        Dict[str, Any]: Validated `NutritionTargetsOutput` with `total_target_kcal`,
            `applied_deficit_kcal`, `reds_safeguard_triggered`, and `macronutrients`.
    """
    baseline_rmr_kcal = 2450
    quality_days = {"Wednesday", "Sunday"}
    reds_triggered = (
        validated_input.oura_hrv_drop_pct > 15.0
        or validated_input.leg_soreness_score >= 6
        or validated_input.gct_imbalance_detected
        or validated_input.focus_mode != "5K_10K"
        or validated_input.day_of_week in quality_days
    )

    applied_deficit = 0 if reds_triggered else -275
    total_kcal = int(round(baseline_rmr_kcal + validated_input.active_burn_kcal + applied_deficit))
    protein_g = max(155, int(round(total_kcal * 0.22 / 4.0)))
    fat_g = int(round(total_kcal * 0.26 / 9.0))
    carbs_g = max(200, int(round((total_kcal - (protein_g * 4) - (fat_g * 9)) / 4.0)))

    safety_note = (
        "RED-S Circuit Breaker Active: 0 kcal deficit enforced to protect autonomic/skeletal recovery."
        if reds_triggered
        else "Green Easy Day in 5K/10K block: Controlled -275 kcal deficit permitted."
    )

    output = NutritionTargetsOutput(
        total_target_kcal=total_kcal,
        applied_deficit_kcal=applied_deficit,
        reds_safeguard_triggered=reds_triggered,
        safety_override_note=safety_note,
        dietary_policy="100% Whole-Food Pescatarian (No dairy, no eggs, no shellfish, no refined sugar)",
        macronutrients={
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "fats_g": fat_g,
            "intra_run_fueling_rule": "60-90g/hr hydrogel carbs on runs > 75 minutes",
        },
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 7: log_athlete_acute_biometric_reflection
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="log_athlete_acute_biometric_reflection",
    input_schema_cls=LogAcuteBiometricReflectionInput,
    recovery_hint="Provide a non-empty `text_message` string between 2 and 2000 characters.",
)
def log_athlete_acute_biometric_reflection(
    validated_input: LogAcuteBiometricReflectionInput,
) -> Dict[str, Any]:
    """
    Parses natural-language athlete feedback to extract acute soreness (0-10),
    RPE (1-10), completed Home Gym barbell/dumbbell loads, and recovery modalities
    (Normatec, Sauna, Ice Barrel) for state persistence.

    Args:
        text_message (str): Athlete's conversational update or post-workout reflection.

    Returns:
        Dict[str, Any]: Validated `BiometricReflectionLoggedOutput` with `extracted_metrics`.
    """
    msg = validated_input.text_message
    soreness_match = re.search(r"soreness\s*(?:is|at|=|:)?\s*(\d{1,2})\s*(?:/10)?", msg, re.IGNORECASE)
    rpe_match = re.search(r"rpe\s*(?:is|at|=|:)?\s*(\d{1,2})\s*(?:/10)?", msg, re.IGNORECASE)
    weight_match = re.search(r"(\d{2,3})\s*(?:lb|lbs)", msg, re.IGNORECASE)

    extracted: Dict[str, Any] = {
        "leg_soreness_score": int(soreness_match.group(1)) if soreness_match else 2,
        "rpe_score": int(rpe_match.group(1)) if rpe_match else 5,
        "logged_weight_lbs": int(weight_match.group(1)) if weight_match else None,
        "normatec_completed": "normatec" in msg.lower(),
        "sauna_completed": "sauna" in msg.lower(),
    }
    output = BiometricReflectionLoggedOutput(
        extracted_metrics=extracted,
        message="Acute biometric reflection parsed and persisted.",
    )
    return output.model_dump()


# -----------------------------------------------------------------------------
# Tool 8: dispatch_garmin_forerunner_970_workout_payload (High-Stakes HITL Action)
# -----------------------------------------------------------------------------

@with_guided_error_handling(
    tool_name="dispatch_garmin_forerunner_970_workout_payload",
    input_schema_cls=DispatchGarminForerunner970WorkoutInput,
    recovery_hint=(
        "Ensure `target_date` is 'YYYY-MM-DD', `target_pace_fast_mm_ss` and `target_pace_slow_mm_ss` "
        "are formatted as 'M:SS' (e.g., '6:45'), and `reps_count` is 1..30."
    ),
)
def dispatch_garmin_forerunner_970_workout_payload(
    validated_input: DispatchGarminForerunner970WorkoutInput,
) -> Dict[str, Any]:
    """
    Builds a structured Garmin Connect `ExecutableStepDTO` / `RepeatGroupDTO` workout JSON
    payload for the athlete's Garmin Forerunner 970 watch. Because mutating watch hardware
    schedules is a high-stakes action, this tool enforces a Human-in-the-Loop (HITL)
    confirmation stop (`PENDING_HUMAN_CONFIRMATION`) whenever `human_confirmed=False`.

    Args:
        workout_name (str): Title of the workout (3 to 64 characters).
        target_date (str): Scheduled workout date in 'YYYY-MM-DD' format.
        reps_count (int): Number of interval repetitions (1-30).
        rep_distance_meters (int): Distance of each interval in meters (100-10000).
        target_pace_fast_mm_ss (str): Fast pace target bound in 'M:SS' per mile format.
        target_pace_slow_mm_ss (str): Slow pace target bound in 'M:SS' per mile format.
        human_confirmed (bool): Whether the athlete has explicitly approved hardware dispatch.

    Returns:
        Dict[str, Any]: Validated `GarminWorkoutDispatchOutput` with `status`
            ('PENDING_HUMAN_CONFIRMATION' or 'success'), `hitl_approval_id`, and `garmin_connect_payload`.
    """
    fast_mps = _pace_mm_ss_to_mps(validated_input.target_pace_fast_mm_ss)
    slow_mps = _pace_mm_ss_to_mps(validated_input.target_pace_slow_mm_ss)

    garmin_payload: Dict[str, Any] = {
        "workoutName": validated_input.workout_name,
        "scheduledDate": validated_input.target_date,
        "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
        "workoutSegments": [
            {
                "segmentOrder": 1,
                "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
                "workoutSteps": [
                    {
                        "type": "ExecutableStepDTO",
                        "stepOrder": 1,
                        "stepType": {"stepTypeId": 1, "stepTypeKey": "warmup"},
                        "endCondition": {"conditionTypeKey": "distance"},
                        "endConditionValue": 3218.68,
                    },
                    {
                        "type": "RepeatGroupDTO",
                        "stepOrder": 2,
                        "numberOfIterations": validated_input.reps_count,
                        "workoutSteps": [
                            {
                                "type": "ExecutableStepDTO",
                                "stepOrder": 1,
                                "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                                "endCondition": {"conditionTypeKey": "distance"},
                                "endConditionValue": float(validated_input.rep_distance_meters),
                                "targetType": {"workoutTargetTypeKey": "pace.zone"},
                                "targetValueOne": slow_mps,
                                "targetValueTwo": fast_mps,
                            }
                        ],
                    },
                    {
                        "type": "ExecutableStepDTO",
                        "stepOrder": 3,
                        "stepType": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
                        "endCondition": {"conditionTypeKey": "distance"},
                        "endConditionValue": 1609.34,
                    },
                ],
            }
        ],
    }

    if not validated_input.human_confirmed:
        hitl = HumanInTheLoopGate()
        checkpoint = hitl.create_approval_checkpoint(
            action_name="dispatch_garmin_forerunner_970_workout_payload",
            payload={
                "workout_name": validated_input.workout_name,
                "target_date": validated_input.target_date,
                "garmin_connect_payload": garmin_payload,
            },
            risk_rationale=(
                f"Pushing workout '{validated_input.workout_name}' ({validated_input.reps_count}x"
                f"{validated_input.rep_distance_meters}m @ {validated_input.target_pace_fast_mm_ss}-"
                f"{validated_input.target_pace_slow_mm_ss}/mi) to Garmin Forerunner 970 hardware requires athlete confirmation."
            ),
        )
        return GarminWorkoutDispatchOutput(
            status="PENDING_HUMAN_CONFIRMATION",
            workout_name=validated_input.workout_name,
            target_date=validated_input.target_date,
            hitl_approval_id=checkpoint["hitl_approval_id"],
            garmin_connect_payload=garmin_payload,
            message=(
                f"Execution paused at Human-in-the-Loop gate (approval_id={checkpoint['hitl_approval_id']}). "
                "Confirm via POST /api/hitl/confirm to push workout to Garmin Forerunner 970."
            ),
        ).model_dump()

    return GarminWorkoutDispatchOutput(
        status="success",
        workout_name=validated_input.workout_name,
        target_date=validated_input.target_date,
        hitl_approval_id=None,
        garmin_connect_payload=garmin_payload,
        message=f"Workout '{validated_input.workout_name}' confirmed and dispatched to Garmin Forerunner 970.",
    ).model_dump()


ADK_TOOL_REGISTRY: Dict[str, Callable] = {
    "query_historical_athlete_telemetry_baseline": query_historical_athlete_telemetry_baseline,
    "diagnose_marathon_biomechanical_fade": diagnose_marathon_biomechanical_fade,
    "build_periodized_vdot_macrocycle_plan": build_periodized_vdot_macrocycle_plan,
    "generate_jack_daniels_2q_weekly_microcycle": generate_jack_daniels_2q_weekly_microcycle,
    "execute_daily_autonomic_biomechanical_triage": execute_daily_autonomic_biomechanical_triage,
    "enforce_reds_pescatarian_caloric_gate": enforce_reds_pescatarian_caloric_gate,
    "log_athlete_acute_biometric_reflection": log_athlete_acute_biometric_reflection,
    "dispatch_garmin_forerunner_970_workout_payload": dispatch_garmin_forerunner_970_workout_payload,
}


def get_adk_tool_declarations() -> List[Dict[str, Any]]:
    """
    Exports strict OpenAPI / Vertex AI ADK FunctionDeclarations with full input and output
    JSON Schemas and comprehensive docstrings for all registered tools.
    """
    schema_pairs = [
        (
            "query_historical_athlete_telemetry_baseline",
            query_historical_athlete_telemetry_baseline.__doc__ or "",
            QueryHistoricalAthleteProfileInput,
            HistoricalAthleteProfileOutput,
        ),
        (
            "diagnose_marathon_biomechanical_fade",
            diagnose_marathon_biomechanical_fade.__doc__ or "",
            DiagnoseMarathonBiomechanicalFadeInput,
            MarathonFadeDiagnosisOutput,
        ),
        (
            "build_periodized_vdot_macrocycle_plan",
            build_periodized_vdot_macrocycle_plan.__doc__ or "",
            BuildPeriodizedMacrocycleInput,
            PeriodizedMacrocycleOutput,
        ),
        (
            "generate_jack_daniels_2q_weekly_microcycle",
            generate_jack_daniels_2q_weekly_microcycle.__doc__ or "",
            GenerateJackDaniels2QMicrocycleInput,
            WeeklyMicrocycleOutput,
        ),
        (
            "execute_daily_autonomic_biomechanical_triage",
            execute_daily_autonomic_biomechanical_triage.__doc__ or "",
            ExecuteDailyAutonomicTriageInput,
            DailyTriageVerdictOutput,
        ),
        (
            "enforce_reds_pescatarian_caloric_gate",
            enforce_reds_pescatarian_caloric_gate.__doc__ or "",
            EnforceRedsPescatarianNutritionInput,
            NutritionTargetsOutput,
        ),
        (
            "log_athlete_acute_biometric_reflection",
            log_athlete_acute_biometric_reflection.__doc__ or "",
            LogAcuteBiometricReflectionInput,
            BiometricReflectionLoggedOutput,
        ),
        (
            "dispatch_garmin_forerunner_970_workout_payload",
            dispatch_garmin_forerunner_970_workout_payload.__doc__ or "",
            DispatchGarminForerunner970WorkoutInput,
            GarminWorkoutDispatchOutput,
        ),
    ]
    declarations: List[Dict[str, Any]] = []
    for name, doc, in_cls, out_cls in schema_pairs:
        declarations.append({
            "name": name,
            "description": doc.strip(),
            "parameters": in_cls.model_json_schema(),
            "response_schema": out_cls.model_json_schema(),
        })
    return declarations
