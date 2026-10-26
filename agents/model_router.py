"""
Strategic Model Router (agents/model_router.py)
Strategic Model Routing
Routes specific requests to the most appropriate Gemini / Vertex AI model tier:
- Fast / Low-Latency Tier (`gemini-2.5-flash` / `gemini-3.8-flash`) for acute telemetry extraction,
  voice transcription, history summarization, and routine check-ins.
- Deep Reasoning / Planning Tier (`gemini-2.5-pro` / `gemini-3.8-pro`) for multi-month macrocycle
  periodization, race-split biomechanical bottleneck forensics, and compound cardiac-vs-mechanical
  differential diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.observability import get_structured_logger, start_trace_span

logger = get_structured_logger("EnduranceCopilot.ModelRouter")

MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro"


class TaskComplexityTier(str, Enum):
    FAST_FLASH = "FAST_FLASH"
    DEEP_PLANNING_PRO = "DEEP_PLANNING_PRO"


@dataclass(frozen=True)
class ModelRouteDecision:
    task_type: str
    complexity_tier: TaskComplexityTier
    primary_model: str
    fallback_models: List[str]
    temperature: float
    max_output_tokens: int
    routing_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "complexity_tier": self.complexity_tier.value,
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "routing_rationale": self.routing_rationale,
        }


class StrategicModelRouter:
    """
    Strategic Model Routing
    Classifies incoming agent tasks by reasoning depth and latency constraints,
    routing fast operational tasks to Gemini Flash and complex multi-constraint
    periodization / diagnostic tasks to Gemini Pro.
    """

    FLASH_PRIMARY = "gemini-2.5-flash"
    FLASH_FALLBACKS = ["gemini-3.8-flash", "gemini-2.5-flash-lite"]

    PRO_PRIMARY = "gemini-2.5-pro"
    PRO_FALLBACKS = ["gemini-3.8-pro", "gemini-2.5-flash"]

    PLANNING_KEYWORDS = {
        "macrocycle",
        "periodize",
        "periodization",
        "16-week",
        "12-week",
        "build plan",
        "bottleneck diagnosis",
        "positive split",
        "differential diagnosis",
        "race strategy",
        "sub-3:10",
        "vdot progression",
        "biomechanical failure chain",
    }

    FAST_KEYWORDS = {
        "log",
        "soreness",
        "rpe",
        "transcribe",
        "sauna",
        "normatec",
        "ice barrel",
        "status",
        "hrv today",
        "body battery",
        "compact",
        "summarize",
    }

    def route_request(
        self,
        task_type: str,
        prompt_text: str = "",
        require_deep_planning: bool = False,
    ) -> ModelRouteDecision:
        """
        Determines the optimal Gemini model and generation parameters for a given task.
        """
        with start_trace_span(
            "StrategicModelRouter.route_request",
            attributes={"task_type": task_type, "require_deep_planning": require_deep_planning},
        ) as span:
            low_prompt = (prompt_text or "").lower()
            task_upper = (task_type or "").upper()

            is_pro_task = (
                require_deep_planning
                or task_upper
                in {
                    "MACROCYCLE_PERIODIZATION",
                    "BIOMECHANICAL_BOTTLENECK_DIAGNOSIS",
                    "MULTI_AGENT_SYNTHESIS_PLANNING",
                    "GUARDRAIL_POLICY_SELF_EVAL",
                }
                or any(kw in low_prompt for kw in self.PLANNING_KEYWORDS)
            )

            if is_pro_task:
                decision = ModelRouteDecision(
                    task_type=task_type,
                    complexity_tier=TaskComplexityTier.DEEP_PLANNING_PRO,
                    primary_model=self.PRO_PRIMARY,
                    fallback_models=list(self.PRO_FALLBACKS),
                    temperature=0.2,
                    max_output_tokens=4096,
                    routing_rationale=(
                        "Routed to Gemini Pro tier for multi-constraint physiological planning, "
                        "macrocycle periodization, or biomechanical differential diagnosis."
                    ),
                )
            else:
                decision = ModelRouteDecision(
                    task_type=task_type,
                    complexity_tier=TaskComplexityTier.FAST_FLASH,
                    primary_model=self.FLASH_PRIMARY,
                    fallback_models=list(self.FLASH_FALLBACKS),
                    temperature=0.3,
                    max_output_tokens=1536,
                    routing_rationale=(
                        "Routed to Gemini Flash tier for low-latency biometric extraction, "
                        "daily triage check-in, voice transcription, or memory compaction."
                    ),
                )

            span["attributes"]["model_routed"] = decision.primary_model
            span["attributes"]["complexity_tier"] = decision.complexity_tier.value

            logger.info(
                f"Routed task '{task_type}' -> {decision.primary_model} ({decision.complexity_tier.value})",
                extra={
                    "event_type": "MODEL_ROUTING_DECISION",
                    "model_routed": decision.primary_model,
                    "metadata": decision.to_dict(),
                },
            )
            return decision
