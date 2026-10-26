"""
Strict Pydantic v2 Input & Output JSON Schemas (agents/schemas.py)
Explicit JSON Schemas
Guided Error Handling Schema
Validates all tool arguments and constrains LLM inputs/outputs using strict JSON schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchemaBase(BaseModel):
    """Base schema enforcing strict field validation and forbidding undeclared fields."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# -----------------------------------------------------------------------------
# Guided Error Handling Output Schema
# -----------------------------------------------------------------------------

class GuidedToolErrorOutput(StrictSchemaBase):
    """
    Structured error response returned to the LLM when a tool call fails validation
    or encounters a domain constraint violation, providing actionable self-correction instructions.
    """
    status: Literal["error"] = "error"
    tool_name: str = Field(..., description="Name of the tool that encountered the error.")
    error_code: str = Field(..., description="Machine-readable error classification code.")
    error_message: str = Field(..., description="Human-readable description of what went wrong.")
    invalid_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of input arguments that failed validation.",
    )
    llm_recovery_instructions: str = Field(
        ...,
        description="Explicit step-by-step instructions telling the LLM how to fix its arguments and retry.",
    )


# -----------------------------------------------------------------------------
# Tool 1: query_historical_athlete_telemetry_baseline
# -----------------------------------------------------------------------------

class QueryHistoricalAthleteProfileInput(StrictSchemaBase):
    """Input schema for querying the athlete's multi-year Garmin/Oura baseline profile."""
    include_pace_buckets: bool = Field(
        default=True,
        description="Whether to include pace-stratified biomechanical Z-score baselines (Easy, Marathon, Threshold, Interval).",
    )


class HistoricalAthleteProfileOutput(StrictSchemaBase):
    """Output schema for historical athlete profile and biomechanical ground truth."""
    status: Literal["success"] = "success"
    athlete_name: str
    active_vdot: float = Field(..., ge=30.0, le=85.0)
    ceiling_vdot: float = Field(..., ge=30.0, le=85.0)
    cadence_anchor_spm: int = Field(..., ge=150, le=220)
    active_baseline: Dict[str, Any]
    historical_ceiling: Dict[str, Any]
    durability_limits: Dict[str, Any]
    pace_bucketed_biomechanics: Dict[str, Any]


# -----------------------------------------------------------------------------
# Tool 2: diagnose_marathon_biomechanical_fade
# -----------------------------------------------------------------------------

class DiagnoseMarathonBiomechanicalFadeInput(StrictSchemaBase):
    """Input schema for diagnosing race split bottlenecks and late-stage mechanical drift."""
    half_1_split_str: str = Field(
        default="1:34:58",
        pattern=r"^\d{1,2}:\d{2}(:\d{2})?$",
        description="First half marathon split in H:MM:SS format (e.g. '1:34:58').",
    )
    half_2_split_str: str = Field(
        default="1:38:00",
        pattern=r"^\d{1,2}:\d{2}(:\d{2})?$",
        description="Second half marathon split in H:MM:SS format (e.g. '1:38:00').",
    )
    late_stage_cadence_spm: float = Field(
        default=180.0,
        ge=120.0,
        le=230.0,
        description="Average cadence (spm) during the final 10K of the marathon.",
    )
    gct_asymmetry_deviation_pct: float = Field(
        default=1.8,
        ge=0.0,
        le=15.0,
        description="Ground Contact Time L/R deviation from 50.0% in percentage points (e.g., 1.8 means 51.8% L / 48.2% R).",
    )


class MarathonFadeDiagnosisOutput(StrictSchemaBase):
    """Output schema for marathon split & biomechanical bottleneck diagnosis."""
    status: Literal["success"] = "success"
    diagnostic_case: str
    anchored_vdot: float
    target_block_vdot: float
    split_delta_seconds: int
    clinical_diagnosis: str
    recommended_macrocycle: str
    macrocycle_key: str
    training_shifts: List[str]


# -----------------------------------------------------------------------------
# Tool 3: build_periodized_vdot_macrocycle_plan
# -----------------------------------------------------------------------------

class BuildPeriodizedMacrocycleInput(StrictSchemaBase):
    """Input schema for initializing a multi-phase periodized macrocycle."""
    target_distance: Literal["Marathon", "Half Marathon", "10K", "5K"] = Field(
        ...,
        description="Target race distance ('Marathon', 'Half Marathon', '10K', or '5K').",
    )
    target_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Target race date in strict ISO YYYY-MM-DD format.",
    )
    focus_theme: str = Field(
        default="Sub-3:10 Marathon Build",
        min_length=3,
        max_length=120,
        description="Primary physiological focus theme for the training block.",
    )
    require_human_confirmation: bool = Field(
        default=False,
        description="If True, stages the macrocycle mutation behind a Human-in-the-Loop confirmation gate.",
    )


class PeriodizedMacrocycleOutput(StrictSchemaBase):
    """Output schema for a constructed periodized macrocycle plan."""
    status: Literal["success", "PENDING_HUMAN_CONFIRMATION"] = "success"
    goal_name: str
    target_distance: str
    target_date: str
    current_vdot: float
    target_vdot: float
    total_weeks: int
    reverse_taper_auto_inserted: bool
    phases: List[Dict[str, Any]]
    vdot_paces: Dict[str, Any]
    hitl_approval_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Tool 4: generate_jack_daniels_2q_weekly_microcycle
# -----------------------------------------------------------------------------

class GenerateJackDaniels2QMicrocycleInput(StrictSchemaBase):
    """Input schema for generating a 7-day rolling microcycle."""
    plan_key: str = Field(
        default="auto",
        description="Microcycle key ('auto', 'taper_1', 'taper_2', '2q_week_1' .. '2q_week_8').",
    )
    week_start_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Optional ISO YYYY-MM-DD start date for the 7-day microcycle.",
    )


class WeeklyMicrocycleOutput(StrictSchemaBase):
    """Output schema for the 7-day rolling microcycle."""
    status: Literal["success"] = "success"
    selected_plan_key: str
    phase_title: str
    active_vdot: float
    days_count: int = Field(..., ge=1, le=14)
    days: List[Dict[str, Any]]
    closed_loop_gate_status: str


# -----------------------------------------------------------------------------
# Tool 5: execute_daily_autonomic_biomechanical_triage
# -----------------------------------------------------------------------------

class ExecuteDailyAutonomicTriageInput(StrictSchemaBase):
    """Input schema for executing the daily closed-loop triage decision engine."""
    leg_soreness_score: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Subjective muscular/pelvic soreness rating on a 0-10 scale.",
    )
    oura_hrv_ms: float = Field(
        default=42.0,
        ge=5.0,
        le=250.0,
        description="Overnight Oura Ring HRV RMSSD in milliseconds.",
    )
    oura_baseline_hrv_ms: float = Field(
        default=42.0,
        ge=5.0,
        le=250.0,
        description="Athlete's 60-day rolling Oura HRV RMSSD baseline in milliseconds.",
    )
    gct_deviation_pct: float = Field(
        default=0.4,
        ge=0.0,
        le=15.0,
        description="Absolute L/R Ground Contact Time asymmetry deviation from 50.0%.",
    )
    hr_decoupling_pct: float = Field(
        default=2.1,
        ge=-10.0,
        le=30.0,
        description="Aerobic Pace:HR cardiac decoupling percentage across sub-maximal segment.",
    )
    consecutive_asymmetry_runs: int = Field(
        default=0,
        ge=0,
        le=30,
        description="Number of consecutive runs with >1.0% GCT shift in the same direction.",
    )


class DailyTriageVerdictOutput(StrictSchemaBase):
    """Output schema for the daily triage decision (STAY_THE_COURSE, DOWNSHIFT, or STEP_UP)."""
    status: Literal["success"] = "success"
    triage_verdict: Literal["STAY_THE_COURSE", "DOWNSHIFT", "STEP_UP"]
    cardiac_vs_mechanical_quadrant: str
    rationale: str
    adjustments: Dict[str, Any]
    reds_deficit_permitted: bool


# -----------------------------------------------------------------------------
# Tool 6: enforce_reds_pescatarian_caloric_gate
# -----------------------------------------------------------------------------

class EnforceRedsPescatarianNutritionInput(StrictSchemaBase):
    """Input schema for computing whole-food pescatarian macros with RED-S safety gating."""
    day_of_week: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ] = Field(default="Monday", description="Day of the week.")
    active_burn_kcal: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Active exercise kilocalories burned from Garmin telemetry.",
    )
    focus_mode: str = Field(
        default="5K_10K",
        description="Current macrocycle focus ('ACUTE_RECOVERY', '5K_10K', 'MARATHON').",
    )
    oura_hrv_drop_pct: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Percentage drop in overnight Oura HRV relative to 60-day baseline.",
    )
    leg_soreness_score: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Muscular soreness score (0-10); >=6 triggers RED-S deficit abort.",
    )
    gct_imbalance_detected: bool = Field(
        default=False,
        description="True if persistent Ground Contact Time asymmetry (>1.0%) is flagged.",
    )


class NutritionTargetsOutput(StrictSchemaBase):
    """Output schema for RED-S gated whole-food pescatarian daily nutrition targets."""
    status: Literal["success"] = "success"
    total_target_kcal: int = Field(..., ge=1800, le=12000)
    applied_deficit_kcal: int = Field(..., ge=0, le=500)
    reds_safeguard_triggered: bool
    safety_override_note: Optional[str] = None
    dietary_policy: str
    macronutrients: Dict[str, Any]


# -----------------------------------------------------------------------------
# Tool 7: log_athlete_acute_biometric_reflection
# -----------------------------------------------------------------------------

class LogAcuteBiometricReflectionInput(StrictSchemaBase):
    """Input schema for logging natural language athlete reflections and acute metrics."""
    text_message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Athlete's natural language feedback (RPE, soreness, gym weights, recovery modalities).",
    )

    @field_validator("text_message")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text_message cannot be blank or whitespace-only.")
        return v.strip()


class BiometricReflectionLoggedOutput(StrictSchemaBase):
    """Output schema confirming reflection extraction and state persistence."""
    status: Literal["success"] = "success"
    extracted_metrics: Dict[str, Any]
    message: str


# -----------------------------------------------------------------------------
# Tool 8: dispatch_garmin_forerunner_970_workout_payload (High-Stakes HITL Action)
# -----------------------------------------------------------------------------

class DispatchGarminForerunner970WorkoutInput(StrictSchemaBase):
    """Input schema for dispatching a structured workout to Garmin Connect (Forerunner 970)."""
    workout_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Workout title displayed on the Garmin Forerunner 970 watch face.",
    )
    target_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Scheduled date in YYYY-MM-DD format.",
    )
    reps_count: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of interval repetitions.",
    )
    rep_distance_meters: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Distance of each interval repetition in meters.",
    )
    target_pace_fast_mm_ss: str = Field(
        default="6:45",
        pattern=r"^\d{1,2}:\d{2}$",
        description="Fastest target pace bound in M:SS/mi format.",
    )
    target_pace_slow_mm_ss: str = Field(
        default="6:52",
        pattern=r"^\d{1,2}:\d{2}$",
        description="Slowest target pace bound in M:SS/mi format.",
    )
    human_confirmed: bool = Field(
        default=False,
        description="Must be True or approved via HITL token before hardware dispatch.",
    )


class GarminWorkoutDispatchOutput(StrictSchemaBase):
    """Output schema for Garmin Forerunner 970 workout dispatch or HITL pause."""
    status: Literal["success", "PENDING_HUMAN_CONFIRMATION"]
    workout_name: str
    target_date: str
    hitl_approval_id: Optional[str] = None
    garmin_connect_payload: Dict[str, Any]
    message: str
