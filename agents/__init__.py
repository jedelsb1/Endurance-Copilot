"""
Endurance Copilot ADK Agent Package (`agents/__init__.py`)
Exports `root_agent` and `agent` module as required by Google's `adk-python` (`google-adk`) CLI:
- `adk run agents`
- `adk web`
- `adk eval agents tests/golden_dataset.json`
"""

from agents import agent
from agents.agent import (
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
    "create_adk_runner",
]
