"""
Root ADK-Python Package Entrypoint (__init__.py)
Allows the repository root itself to be imported directly by `google-adk`
(`from . import agent` / `root_agent`).
"""

from . import agent
from .agent import (
    adk_function_tools,
    biomechanics_rehab_specialist_agent,
    create_adk_runner,
    daily_triage_sequential_agent,
    garmin_telemetry_analyst_agent,
    nutrition_reds_safeguard_agent,
    root_agent,
)

__all__ = [
    "agent",
    "root_agent",
    "daily_triage_sequential_agent",
    "garmin_telemetry_analyst_agent",
    "biomechanics_rehab_specialist_agent",
    "nutrition_reds_safeguard_agent",
    "adk_function_tools",
    "create_adk_runner",
]
