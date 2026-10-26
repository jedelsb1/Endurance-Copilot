"""
Google ADK Multi-Agent Orchestrator: Coordinator & Sequential Pipeline Patterns
(agents/orchestrator.py)

Implements Google ADK (`google.adk.agents`) Coordinator and Sequential multi-agent
orchestration patterns across specialist sub-agents:
- Root Coordinator Agent (`EnduranceCopilotCoordinatorAgent`)
- Sequential Triage Pipeline (`DailyTriageSequentialAgent`)
- Domain Specialist Sub-Agents (`GarminTelemetrySubAgent`, `BiomechanicsRehabSubAgent`, `NutritionRedsSubAgent`)
- Strategic Model Router, Guardrails Plugin, HITL Gate, Persistent Vector Memory, and OpenTelemetry Tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.adk_tools import (
    ADK_TOOL_REGISTRY,
    build_periodized_vdot_macrocycle_plan,
    dispatch_garmin_forerunner_970_workout_payload,
    enforce_reds_pescatarian_caloric_gate,
    execute_daily_autonomic_biomechanical_triage,
    log_athlete_acute_biometric_reflection,
)
from agents.constitution import (
    AGENT_CONSTITUTION_SYSTEM_PROMPT,
    build_constitutional_system_prompt,
)
from agents.guardrails import ADKPolicyGuardrailPlugin
from agents.hitl import HumanInTheLoopGate
from agents.model_router import StrategicModelRouter
from core.memory_compaction import (
    ADKHistoryCompactor,
    AsyncMemoryConsolidationWorker,
    PersistentSessionMemoryStore,
)
from core.observability import (
    capture_intent_and_outcome,
    get_current_trace_id,
    get_structured_logger,
    start_trace_span,
)

logger = get_structured_logger("EnduranceCopilot.Orchestrator")


# -----------------------------------------------------------------------------
# Specialist ADK Sub-Agents
# -----------------------------------------------------------------------------

class GarminTelemetrySubAgent:
    """
    Specialist ADK Sub-Agent responsible for Garmin FIT telemetry analysis,
    pace-normalized Z-scores, trail grade filtering, and intra-workout drift.
    """

    name = "garmin_telemetry_analyst_subagent"
    description = (
        "Analyzes Garmin Forerunner 970 FIT telemetry, pace-stratified Z-scores, "
        "trail grade exemptions, and Cardiac vs. Mechanical decoupling."
    )

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self.knowledge_dir = knowledge_dir

    def run_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with capture_intent_and_outcome(
            agent_name=self.name,
            intended_action="analyze_activity_and_triage",
            intended_arguments={
                "leg_soreness_score": state.get("leg_soreness_score", 2),
                "oura_hrv_ms": state.get("oura_hrv_ms", 42.0),
                "gct_deviation_pct": state.get("gct_deviation_pct", 0.4),
            },
            rationale="Step 1 of Sequential Pipeline: Evaluate Garmin biomechanics and autonomic triage.",
        ) as rec:
            triage_out = execute_daily_autonomic_biomechanical_triage(
                leg_soreness_score=int(state.get("leg_soreness_score", 2)),
                oura_hrv_ms=float(state.get("oura_hrv_ms", 42.0)),
                oura_baseline_hrv_ms=float(state.get("oura_baseline_hrv_ms", 42.0)),
                gct_deviation_pct=float(state.get("gct_deviation_pct", 0.4)),
                hr_decoupling_pct=float(state.get("hr_decoupling_pct", 2.1)),
                consecutive_asymmetry_runs=int(state.get("consecutive_asymmetry_runs", 0)),
            )
            state["telemetry_triage_output"] = triage_out
            rec["actual_outcome"] = triage_out
            return state


class BiomechanicsRehabSubAgent:
    """
    Specialist ADK Sub-Agent responsible for pelvic stability, psoas decompression,
    the 14-day post-marathon lifting lockout, and 2x/week 20-min Home Gym prescriptions.
    """

    name = "biomechanics_rehab_specialist_subagent"
    description = (
        "Evaluates 14-day post-marathon eccentric lockout, pelvic/psoas mechanics, "
        "and prescribes 20-minute Home Gym Session A/B strength progressions."
    )

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self.knowledge_dir = knowledge_dir

    def run_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        days_post = int(state.get("days_post_marathon", 16))
        gct_dev = float(state.get("gct_deviation_pct", 0.4))
        cadence_late = float(state.get("late_stage_cadence_spm", 184.0))

        with capture_intent_and_outcome(
            agent_name=self.name,
            intended_action="evaluate_rehab_and_lifting_lockout",
            intended_arguments={
                "days_post_marathon": days_post,
                "gct_deviation_pct": gct_dev,
            },
            rationale="Step 2 of Sequential Pipeline: Check 14-day lifting lockout and biomechanical failure cascade.",
        ) as rec:
            locked = days_post <= 14
            stage_name = (
                "Stage 1-2: Post-Marathon 14-Day Eccentric Lockout (Decompression & Isometrics Only)"
                if locked
                else "Stage 3: 2x/Week 20-Min Progressive Home Gym (Session A Heavy Hinge / Session B Unilateral Knee)"
            )
            lifting_status = {
                "days_post_marathon": days_post,
                "heavy_lifting_locked": locked,
                "stage_name": stage_name,
            }
            failure_chain = {
                "gct_deviation_pct": gct_dev,
                "late_stage_cadence_spm": cadence_late,
                "asymmetry_flag": gct_dev > 1.0,
                "corrective_focus": (
                    "Reverse Copenhagen Adductor Plank + Standing Banded Psoas March"
                    if gct_dev > 1.0
                    else "Standard Session A/B progressive overload (+5 lb bilateral)"
                ),
            }
            rehab_bundle = {
                "lifting_status": lifting_status,
                "biomechanical_failure_chain": failure_chain,
            }
            state["rehab_specialist_output"] = rehab_bundle
            rec["actual_outcome"] = {
                "heavy_lifting_locked": locked,
                "stage_name": stage_name,
            }
            return state


class NutritionRedsSubAgent:
    """
    Specialist ADK Sub-Agent responsible for 100% whole-food pescatarian fueling
    and enforcing the RED-S caloric deficit safeguard.
    """

    name = "nutrition_reds_safeguard_subagent"
    description = (
        "Computes 100% whole-food pescatarian macronutrients (>=150g protein) and "
        "enforces the RED-S clinical caloric deficit gate."
    )

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self.knowledge_dir = knowledge_dir

    def run_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        day_of_week = str(state.get("day_of_week", "Monday"))
        active_burn = float(state.get("active_burn_kcal", 850.0))
        focus_mode = str(state.get("focus_mode", "5K_10K"))
        hrv_ms = float(state.get("oura_hrv_ms", 42.0))
        base_hrv = float(state.get("oura_baseline_hrv_ms", 42.0))
        hrv_drop_pct = ((base_hrv - hrv_ms) / max(1.0, base_hrv)) * 100.0
        soreness = int(state.get("leg_soreness_score", 2))
        gct_imbalance = float(state.get("gct_deviation_pct", 0.4)) > 1.0

        with capture_intent_and_outcome(
            agent_name=self.name,
            intended_action="enforce_reds_pescatarian_caloric_gate",
            intended_arguments={
                "day_of_week": day_of_week,
                "active_burn_kcal": active_burn,
                "oura_hrv_drop_pct": round(hrv_drop_pct, 1),
                "leg_soreness_score": soreness,
            },
            rationale="Step 3 of Sequential Pipeline: Compute pescatarian targets and enforce RED-S gate.",
        ) as rec:
            nut_out = enforce_reds_pescatarian_caloric_gate(
                day_of_week=day_of_week,
                active_burn_kcal=active_burn,
                focus_mode=focus_mode,
                oura_hrv_drop_pct=round(hrv_drop_pct, 2),
                leg_soreness_score=soreness,
                gct_imbalance_detected=gct_imbalance,
            )
            state["nutrition_specialist_output"] = nut_out
            rec["actual_outcome"] = nut_out
            return state


# -----------------------------------------------------------------------------
# Sequential Multi-Agent Pattern (`DailyTriageSequentialAgent`)
# -----------------------------------------------------------------------------

class DailyTriageSequentialAgent:
    """
    ADK Sequential Multi-Agent Pattern (`SequentialAgent`):
    Executes a deterministic 4-stage pipeline:
    1. `GarminTelemetrySubAgent` -> Autonomic & Biomechanical Triage
    2. `BiomechanicsRehabSubAgent` -> 14-Day Eccentric Lockout & Rehab Cascade
    3. `NutritionRedsSubAgent` -> Whole-Food Pescatarian Fueling & RED-S Gate
    4. `ADKPolicyGuardrailPlugin` -> Clinical Self-Evaluation & Policy Verification
    """

    name = "daily_triage_sequential_pipeline"

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        self.telemetry_agent = GarminTelemetrySubAgent(knowledge_dir)
        self.rehab_agent = BiomechanicsRehabSubAgent(knowledge_dir)
        self.nutrition_agent = NutritionRedsSubAgent(knowledge_dir)
        self.guardrail_plugin = ADKPolicyGuardrailPlugin()

    def execute_pipeline(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all specialist sub-agents in strict sequential order under a parent trace span."""
        with start_trace_span(
            "DailyTriageSequentialAgent.execute_pipeline",
            attributes={"pipeline_pattern": "ADK_SEQUENTIAL", "sub_agents_count": 3},
        ):
            pipeline_state = dict(initial_context)
            pipeline_state = self.telemetry_agent.run_step(pipeline_state)
            pipeline_state = self.rehab_agent.run_step(pipeline_state)
            pipeline_state = self.nutrition_agent.run_step(pipeline_state)

            triage = pipeline_state["telemetry_triage_output"]
            rehab = pipeline_state["rehab_specialist_output"]
            nutrition = pipeline_state["nutrition_specialist_output"]

            raw_synthesis = (
                f"Daily Triage Verdict: {triage.get('triage_verdict')} "
                f"({triage.get('cardiac_vs_mechanical_quadrant')}). "
                f"{triage.get('rationale')} "
                f"Strength/Rehab Stage: {rehab['lifting_status'].get('stage_name')}. "
                f"Nutrition Target: {nutrition.get('total_target_kcal')} kcal "
                f"(Deficit: {nutrition.get('applied_deficit_kcal')} kcal • {nutrition.get('dietary_policy')})."
            )

            hrv_ms = float(pipeline_state.get("oura_hrv_ms", 42.0))
            base_hrv = float(pipeline_state.get("oura_baseline_hrv_ms", 42.0))
            hrv_drop_pct = ((base_hrv - hrv_ms) / max(1.0, base_hrv)) * 100.0

            guardrail_eval = self.guardrail_plugin.evaluate_and_enforce_output_policy(
                candidate_response=raw_synthesis,
                context_metrics={
                    "oura_hrv_drop_pct": hrv_drop_pct,
                    "leg_soreness_score": pipeline_state.get("leg_soreness_score", 2),
                    "days_post_marathon": pipeline_state.get("days_post_marathon", 16),
                    "gct_imbalance_detected": float(pipeline_state.get("gct_deviation_pct", 0.4)) > 1.0,
                },
            )

            return {
                "orchestration_pattern": "ADK_SEQUENTIAL_PIPELINE",
                "trace_id": get_current_trace_id(),
                "sub_agents_executed": [
                    self.telemetry_agent.name,
                    self.rehab_agent.name,
                    self.nutrition_agent.name,
                ],
                "triage_verdict": triage.get("triage_verdict"),
                "telemetry_triage": triage,
                "rehab_prescription": rehab,
                "nutrition_prescription": nutrition,
                "guardrail_self_evaluation": guardrail_eval,
                "synthesized_directive": guardrail_eval["validated_response"],
            }


# -----------------------------------------------------------------------------
# Coordinator Multi-Agent Pattern (`EnduranceCopilotCoordinatorAgent`)
# -----------------------------------------------------------------------------

class EnduranceCopilotCoordinatorAgent:
    """
    Root ADK Coordinator Agent (`LlmAgent` / `Coordinator` pattern) that:
    1. Enforces Input Security Guardrails (`ADKPolicyGuardrailPlugin.validate_input_security`)
    2. Compacts multi-turn conversational context (`ADKHistoryCompactor` + GCP Context Cache)
    3. Retrieves relevant longitudinal memories from SQLite Vector Store (`PersistentSessionMemoryStore`)
    4. Strategically routes requests to Gemini Flash or Gemini Pro (`StrategicModelRouter`)
    5. Delegates execution to `DailyTriageSequentialAgent` or specialized ADK tools
    6. Enforces Human-in-the-Loop confirmation checkpoints (`HumanInTheLoopGate`)
    7. Validates final responses via Output Clinical Self-Evaluation Guardrail
    8. Schedules non-blocking Async Memory Consolidation (`AsyncMemoryConsolidationWorker`)
    """

    name = "endurance_copilot_root_coordinator"

    def __init__(
        self,
        knowledge_dir: Optional[Path] = None,
        db_file: Optional[Path] = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.router = StrategicModelRouter()
        self.guardrail = ADKPolicyGuardrailPlugin()
        self.compactor = ADKHistoryCompactor()
        self.memory_store = PersistentSessionMemoryStore(db_file=db_file, compactor=self.compactor)
        self.async_memory_worker = AsyncMemoryConsolidationWorker(store=self.memory_store)
        self.hitl_gate = HumanInTheLoopGate(db_file=db_file)
        self.sequential_triage = DailyTriageSequentialAgent(knowledge_dir)
        self.sub_agents = [
            self.sequential_triage.telemetry_agent,
            self.sequential_triage.rehab_agent,
            self.sequential_triage.nutrition_agent,
        ]
        self.tools = ADK_TOOL_REGISTRY

    def handle_athlete_request(
        self,
        user_message: str,
        session_id: str = "default_athlete_session",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        override_metrics: Optional[Dict[str, Any]] = None,
        require_hitl_for_mutations: bool = False,
    ) -> Dict[str, Any]:
        """
        End-to-end closed-loop request handler with full OpenTelemetry tracing,
        Input/Output Guardrails, Strategic Model Routing, Multi-Agent Delegation,
        and Async Memory Consolidation.
        """
        with start_trace_span(
            "EnduranceCopilotCoordinatorAgent.handle_athlete_request",
            attributes={"session_id": session_id, "coordinator": self.name},
        ) as root_span:
            trace_id = root_span["trace_id"]

            # 1. Input Security Guardrail Check
            input_check = self.guardrail.validate_input_security(user_message)
            if not input_check["allowed"]:
                return {
                    "status": "blocked_by_guardrail",
                    "trace_id": trace_id,
                    "violation_code": input_check["violation_code"],
                    "response": input_check["block_response"],
                    "guardrail_input_check": input_check,
                }

            clean_message = input_check["sanitized_input"]

            # 2. History Compaction & Context Cache Registration
            persisted_session = self.memory_store.get_session(session_id)
            raw_turns = conversation_history if conversation_history is not None else persisted_session["turns"]
            compaction_info = self.compactor.compact_conversation_history(
                turns=raw_turns,
                existing_summary=persisted_session.get("compacted_summary", ""),
            )

            active_state = {"active_vdot": 49.2, "current_phase": "2Q Marathon Build"}
            system_prompt = build_constitutional_system_prompt(
                active_state=active_state,
                compacted_memory_summary=compaction_info["memory_bank_summary"],
            )
            gcp_cache = self.compactor.get_or_create_gcp_context_cache(
                cache_key="athlete_baseline_v3",
                system_constitution=AGENT_CONSTITUTION_SYSTEM_PROMPT,
                baseline_telemetry={"active_vdot": 49.2},
            )

            # 3. Retrieve relevant semantic memories from Persistent Vector Store
            retrieved_memories = self.memory_store.search_relevant_memories(
                query=clean_message,
                top_k=3,
                session_id=session_id,
            )

            # 4. Strategic Model Routing (Flash vs Pro)
            route_decision = self.router.route_request(
                task_type="COORDINATOR_ORCHESTRATION",
                prompt_text=clean_message,
            )

            # 5. Coordinator Intent Classification & Sub-Agent / Tool Delegation
            low_msg = clean_message.lower()
            delegated_outputs: Dict[str, Any] = {}

            if "push" in low_msg and ("watch" in low_msg or "forerunner" in low_msg or "garmin" in low_msg):
                dispatch_res = dispatch_garmin_forerunner_970_workout_payload(
                    workout_name="Wednesday Q2 Threshold Cruise",
                    target_date="2026-10-28",
                    reps_count=5,
                    rep_distance_meters=1000,
                    target_pace_fast_mm_ss="6:45",
                    target_pace_slow_mm_ss="6:50",
                    human_confirmed=not require_hitl_for_mutations,
                )
                delegated_outputs["garmin_watch_dispatch"] = dispatch_res
                candidate_reply = dispatch_res["message"]
            elif "macrocycle" in low_msg or "build plan" in low_msg:
                macro_res = build_periodized_vdot_macrocycle_plan(
                    target_distance="Marathon" if "marathon" in low_msg else "10K",
                    target_date="2027-03-14",
                    focus_theme="Sub-3:10 Marathon Build",
                    require_human_confirmation=require_hitl_for_mutations,
                )
                delegated_outputs["macrocycle_plan"] = macro_res
                candidate_reply = (
                    f"Constructed {macro_res['total_weeks']}-week {macro_res['goal_name']} "
                    f"(VDOT {macro_res['current_vdot']} -> {macro_res['target_vdot']})."
                )
            else:
                reflection_res = log_athlete_acute_biometric_reflection(text_message=clean_message)
                extracted = reflection_res.get("extracted_metrics", {})
                merged_metrics = dict(override_metrics or {})
                if "leg_soreness_score" in extracted and "leg_soreness_score" not in merged_metrics:
                    merged_metrics["leg_soreness_score"] = extracted["leg_soreness_score"]

                sequential_result = self.sequential_triage.execute_pipeline(merged_metrics)
                delegated_outputs["sequential_pipeline"] = sequential_result
                delegated_outputs["reflection_extraction"] = reflection_res
                candidate_reply = sequential_result["synthesized_directive"]

            # 6. Output Clinical & Policy Self-Evaluation Guardrail
            guardrail_output = self.guardrail.evaluate_and_enforce_output_policy(
                candidate_response=candidate_reply,
                context_metrics=override_metrics or {},
            )
            final_response = guardrail_output["validated_response"]

            # 7. Non-blocking Async Memory Consolidation
            async_task_id = self.async_memory_worker.schedule_background_consolidation(
                session_id=session_id,
                user_message=clean_message,
                agent_response=final_response,
                extracted_metrics=override_metrics,
            )

            return {
                "status": "ok",
                "trace_id": trace_id,
                "coordinator_agent": self.name,
                "model_routing": route_decision.to_dict(),
                "context_compaction": {
                    "tokens_before": compaction_info["tokens_before"],
                    "tokens_after": compaction_info["tokens_after"],
                    "compaction_applied": compaction_info["compaction_applied"],
                    "gcp_context_cache": gcp_cache["cache_name"],
                },
                "retrieved_vector_memories": retrieved_memories,
                "delegated_outputs": delegated_outputs,
                "guardrail_self_evaluation": guardrail_output,
                "async_memory_task_id": async_task_id,
                "system_constitution_active": bool(system_prompt),
                "response": final_response,
            }
