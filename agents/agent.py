"""
Google Agent Development Kit (`adk-python` / `google-adk`) Canonical Agent Definition
(agents/agent.py)

Defines the official `adk-python` agent hierarchy (`root_agent`) for Endurance Copilot
using `google.adk.agents` (`Agent`, `LlmAgent`, `SequentialAgent`), `google.adk.tools`
(`FunctionTool`), `google.adk.runners` (`Runner`), and `google.adk.sessions`
(`DatabaseSessionService`, `InMemorySessionService`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from agents.adk_tools import (
    build_periodized_vdot_macrocycle_plan,
    diagnose_marathon_biomechanical_fade,
    dispatch_garmin_forerunner_970_workout_payload,
    enforce_reds_pescatarian_caloric_gate,
    execute_daily_autonomic_biomechanical_triage,
    generate_jack_daniels_2q_weekly_microcycle,
    log_athlete_acute_biometric_reflection,
    query_historical_athlete_telemetry_baseline,
)
from agents.constitution import AGENT_CONSTITUTION_SYSTEM_PROMPT
from agents.guardrails import ADKPolicyGuardrailPlugin
from agents.model_router import MODEL_FLASH, MODEL_PRO, StrategicModelRouter
from core.observability import capture_intent_and_outcome, get_structured_logger, start_trace_span

logger = get_structured_logger("EnduranceCopilot.ADKAgent")
_guardrail_plugin = ADKPolicyGuardrailPlugin()
_model_router = StrategicModelRouter()


# -----------------------------------------------------------------------------
# ADK Lifecycle Guardrail & Tracing Callbacks
# -----------------------------------------------------------------------------

def adk_before_model_guardrail_callback(
    callback_context: Any,
    llm_request: Any,
) -> Optional[Any]:
    """
    ADK `before_model_callback` hook:
    Screens incoming athlete prompts for prompt-injection and applies dynamic
    model routing (`gemini-2.5-flash` vs `gemini-2.5-pro`).
    """
    prompt_text = str(getattr(llm_request, "contents", "") or "")
    check = _guardrail_plugin.validate_input_security(prompt_text)
    if not check["allowed"]:
        logger.warning(
            "ADK before_model_callback blocked unsafe prompt",
            extra={
                "event_type": "ADK_BEFORE_MODEL_GUARDRAIL_BLOCK",
                "violation_code": check["violation_code"],
            },
        )
        return {"blocked": True, "response": check["block_response"]}
    return None


def adk_after_model_guardrail_callback(
    callback_context: Any,
    llm_response: Any,
) -> Optional[Any]:
    """
    ADK `after_model_callback` hook:
    Evaluates model outputs against the clinical constitution (RED-S caloric gate,
    100% whole-food pescatarian policy, and 14-day post-marathon lifting lockout).
    """
    text = str(getattr(llm_response, "text", "") or "")
    if text:
        eval_res = _guardrail_plugin.evaluate_and_enforce_output_policy(
            candidate_response=text,
            context_metrics={},
        )
        if not eval_res["passed_self_evaluation"] and hasattr(llm_response, "text"):
            llm_response.text = eval_res["validated_response"]
    return None


# -----------------------------------------------------------------------------
# ADK FunctionTool Wrapping (Explicit Schemas + Comprehensive Docstrings)
# -----------------------------------------------------------------------------

adk_function_tools: List[FunctionTool] = [
    FunctionTool(func=query_historical_athlete_telemetry_baseline),
    FunctionTool(func=diagnose_marathon_biomechanical_fade),
    FunctionTool(func=build_periodized_vdot_macrocycle_plan),
    FunctionTool(func=generate_jack_daniels_2q_weekly_microcycle),
    FunctionTool(func=execute_daily_autonomic_biomechanical_triage),
    FunctionTool(func=enforce_reds_pescatarian_caloric_gate),
    FunctionTool(func=log_athlete_acute_biometric_reflection),
    FunctionTool(func=dispatch_garmin_forerunner_970_workout_payload),
]


# -----------------------------------------------------------------------------
# Specialist ADK LlmAgents (Sub-Agents)
# -----------------------------------------------------------------------------

garmin_telemetry_analyst_agent = LlmAgent(
    name="garmin_telemetry_analyst_agent",
    model=MODEL_FLASH,
    description=(
        "Specialist ADK sub-agent that analyzes Garmin Forerunner 970 FIT telemetry, "
        "pace-normalized Z-scores, trail grade exemptions, and Cardiac vs. Mechanical decoupling."
    ),
    instruction=(
        f"{AGENT_CONSTITUTION_SYSTEM_PROMPT}\n\n"
        "ROLE: You are the Garmin Telemetry & Biomechanics Analyst sub-agent. "
        "Always invoke `execute_daily_autonomic_biomechanical_triage` and `query_historical_athlete_telemetry_baseline` "
        "to evaluate Ground Contact Time (GCT) drift, L/R balance asymmetry, and HR decoupling."
    ),
    tools=[
        FunctionTool(func=query_historical_athlete_telemetry_baseline),
        FunctionTool(func=execute_daily_autonomic_biomechanical_triage),
        FunctionTool(func=diagnose_marathon_biomechanical_fade),
    ],
    before_model_callback=adk_before_model_guardrail_callback,
    after_model_callback=adk_after_model_guardrail_callback,
    output_key="telemetry_triage_output",
)

biomechanics_rehab_specialist_agent = LlmAgent(
    name="biomechanics_rehab_specialist_agent",
    model=MODEL_FLASH,
    description=(
        "Specialist ADK sub-agent for pelvic stability, psoas decompression, "
        "the 14-day post-marathon eccentric lockout, and 2x/week 20-minute Home Gym strength."
    ),
    instruction=(
        f"{AGENT_CONSTITUTION_SYSTEM_PROMPT}\n\n"
        "ROLE: You are the Biomechanics & Strength Rehab Specialist sub-agent. "
        "Enforce the 3-consecutive-run GCT asymmetry filter (>1.0% shift), the 14-day post-marathon "
        "eccentric barbell lockout, and Session A / Session B 20-minute Home Gym progression (+5 lb bilateral)."
    ),
    tools=[
        FunctionTool(func=generate_jack_daniels_2q_weekly_microcycle),
        FunctionTool(func=log_athlete_acute_biometric_reflection),
    ],
    before_model_callback=adk_before_model_guardrail_callback,
    after_model_callback=adk_after_model_guardrail_callback,
    output_key="rehab_specialist_output",
)

nutrition_reds_safeguard_agent = LlmAgent(
    name="nutrition_reds_safeguard_agent",
    model=MODEL_FLASH,
    description=(
        "Specialist ADK sub-agent that computes 100% whole-food pescatarian macros (>=150g protein) "
        "and enforces the RED-S clinical caloric deficit gate."
    ),
    instruction=(
        f"{AGENT_CONSTITUTION_SYSTEM_PROMPT}\n\n"
        "ROLE: You are the Pescatarian Sports Nutrition & RED-S Safeguard sub-agent. "
        "Always call `enforce_reds_pescatarian_caloric_gate` to compute daily kcal and macros. "
        "Never output meal recipes or smoothie scripts; output only numeric macro/calorie targets and clinical rules."
    ),
    tools=[
        FunctionTool(func=enforce_reds_pescatarian_caloric_gate),
    ],
    before_model_callback=adk_before_model_guardrail_callback,
    after_model_callback=adk_after_model_guardrail_callback,
    output_key="nutrition_specialist_output",
)


# -----------------------------------------------------------------------------
# ADK SequentialAgent Pipeline (`daily_triage_sequential_agent`)
# -----------------------------------------------------------------------------

daily_triage_sequential_agent = SequentialAgent(
    name="daily_triage_sequential_agent",
    description=(
        "Deterministic ADK SequentialAgent pipeline executing Garmin Telemetry Triage -> "
        "Biomechanics & 14-Day Lockout Verification -> RED-S Pescatarian Fueling Gate."
    ),
    sub_agents=[
        garmin_telemetry_analyst_agent,
        biomechanics_rehab_specialist_agent,
        nutrition_reds_safeguard_agent,
    ],
)


# -----------------------------------------------------------------------------
# Canonical ADK Root Coordinator Agent (`root_agent`)
# -----------------------------------------------------------------------------

root_agent = Agent(
    name="endurance_copilot_root_coordinator",
    model=MODEL_PRO,
    description=(
        "Endurance Copilot Head Coach & Root ADK Coordinator Agent. Orchestrates macrocycle "
        "periodization (VDOT 49.2 -> 53.5 ceiling), Jack Daniels 2Q microcycles, Garmin Forerunner 970 "
        "workout dispatches (with HITL confirmation), and delegates daily triage to specialist sub-agents."
    ),
    instruction=AGENT_CONSTITUTION_SYSTEM_PROMPT,
    tools=adk_function_tools,
    sub_agents=[
        daily_triage_sequential_agent,
    ],
    before_model_callback=adk_before_model_guardrail_callback,
    after_model_callback=adk_after_model_guardrail_callback,
)


def create_adk_runner(db_url: str = "sqlite:///knowledge/copilot.db") -> Runner:
    """
    Creates a configured Google ADK `Runner` backed by `DatabaseSessionService`
    for persistent multi-turn execution of `root_agent`.
    """
    with start_trace_span("create_adk_runner", attributes={"db_url": db_url}):
        with capture_intent_and_outcome(
            agent_name="ADKRuntimeInitializer",
            intended_action="initialize_adk_runner",
            intended_arguments={"agent_name": root_agent.name, "db_url": db_url},
            rationale="Initialize Google ADK Runner with persistent SQLite DatabaseSessionService.",
        ) as outcome:
            session_service = DatabaseSessionService(db_url=db_url)
            runner = Runner(
                agent=root_agent,
                app_name="endurance_copilot_adk",
                session_service=session_service,
            )
            outcome["actual_outcome"] = {
                "runner_initialized": True,
                "root_agent": root_agent.name,
                "sub_agents": [sa.name for sa in root_agent.sub_agents],
            }
            return runner
