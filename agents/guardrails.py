"""
ADK Security & Clinical Self-Evaluation Policy Guardrails (agents/guardrails.py)
Guardrails & Policy Plugins
Implements input prompt-injection screening, Google Cloud Vertex AI Safety Settings,
and output clinical self-evaluation guardrails (RED-S Energy Gate, 100% Pescatarian
Dietary Invariant, and 14-Day Post-Marathon Eccentric Lockout).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.observability import get_structured_logger, start_trace_span
from core.pii_redaction import scrub_pii

logger = get_structured_logger("EnduranceCopilot.Guardrails")


class ADKPolicyGuardrailPlugin:
    """
    Guardrails & Policy Plugins
    Provides ADK-compatible `before_model_callback` (Input Guardrail) and
    `after_model_callback` (Output Self-Evaluation & Clinical Policy Enforcement)
    alongside Google Cloud Vertex AI Model Armor / SafetySettings configurations.
    """

    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions"),
        re.compile(r"(?i)disregard\s+(?:the\s+)?constitution"),
        re.compile(r"(?i)reveal\s+(?:your\s+)?(?:system\s+prompt|api\s+key|secret)"),
        re.compile(r"(?i)bypass\s+(?:the\s+)?(?:red-s|reds|safety)\s+gate"),
        re.compile(r"(?i)jailbreak|developer\s+mode\s+override"),
    ]

    NON_PESCATARIAN_FORBIDDEN_TERMS = {
        "chicken": "wild-caught salmon or tempeh",
        "beef": "wild-caught Alaskan cod or lentils",
        "steak": "seared ahi tuna or black beans",
        "pork": "hemp seeds and edamame",
        "bacon": "smoked dulse or walnuts",
        "whey protein": "pea and hemp protein isolate",
        "dairy milk": "unsweetened almond or oat milk",
        "cheese": "avocado or tahini dressing",
        "eggs": "tofu scramble or chia seed pudding",
        "shrimp": "wild sardines or Pacific halibut",
        "lobster": "wild salmon",
        "crab": "wild cod",
        "refined sugar": "Medjool dates or pure maple syrup",
    }

    @staticmethod
    def get_vertex_ai_safety_settings() -> List[Dict[str, str]]:
        """
        Returns Google Cloud Vertex AI / Gemini API `safetySettings` configuration
        enforcing strict harm blocking across all categories.
        """
        return [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            },
        ]

    def validate_input_security(self, user_input: str) -> Dict[str, Any]:
        """
        ADK `before_model_callback` Input Guardrail:
        Screens user prompt for prompt injection, system override attempts, and PII leakage.
        """
        with start_trace_span(
            "ADKPolicyGuardrailPlugin.validate_input_security",
            attributes={"input_length": len(user_input or "")},
        ):
            clean_text = scrub_pii(user_input or "")
            for pattern in self.PROMPT_INJECTION_PATTERNS:
                if pattern.search(clean_text):
                    logger.warning(
                        "Blocked prompt injection attempt in ADK Input Guardrail",
                        extra={
                            "event_type": "GUARDRAIL_INPUT_BLOCKED",
                            "metadata": {"matched_pattern": pattern.pattern},
                        },
                    )
                    return {
                        "allowed": False,
                        "violation_code": "PROMPT_INJECTION_DETECTED",
                        "sanitized_input": "",
                        "block_response": (
                            "Security Guardrail Alert: Request blocked because it attempted to override "
                            "constitutional safety instructions or extract protected system credentials."
                        ),
                    }

            return {
                "allowed": True,
                "violation_code": None,
                "sanitized_input": clean_text,
                "block_response": None,
            }

    def evaluate_and_enforce_output_policy(
        self,
        candidate_response: str,
        context_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ADK `after_model_callback` Self-Evaluation & Clinical Policy Guardrail:
        Evaluates generated coaching directives against:
        1. RED-S Energy Availability Gate (no caloric deficit when HRV drop > 15% or soreness >= 6/10)
        2. 100% Whole-Food Pescatarian Dietary Policy (no meat, poultry, dairy, eggs, shellfish, refined sugar)
        3. 14-Day Post-Marathon Eccentric Lockout (no heavy barbell RDLs/squats within 14 days of marathon)
        4. PII & Secret Redaction
        Computes a deterministic `self_eval_score` (0.0 to 1.0) and auto-remediates any policy violation.
        """
        with start_trace_span("ADKPolicyGuardrailPlugin.evaluate_and_enforce_output_policy"):
            metrics = context_metrics or {}
            hrv_drop_pct = float(metrics.get("oura_hrv_drop_pct", 0.0))
            soreness_score = int(metrics.get("leg_soreness_score", 2))
            days_post_marathon = int(metrics.get("days_post_marathon", 16))
            gct_imbalance = bool(metrics.get("gct_imbalance_detected", False))

            remediated_text = scrub_pii(candidate_response or "")
            violations_detected: List[str] = []

            # 1. Enforce RED-S Caloric Deficit Gate
            reds_risk_active = hrv_drop_pct > 15.0 or soreness_score >= 6 or gct_imbalance
            if reds_risk_active:
                deficit_regex = re.compile(
                    r"(?i)(?:apply|prescribe|maintain|run)\s+a?\s*(?:250|300|\d+)\s*kcal\s+deficit|caloric\s+deficit\s+approved"
                )
                if deficit_regex.search(remediated_text):
                    violations_detected.append("REDS_CALORIC_DEFICIT_VIOLATION")
                    remediated_text = deficit_regex.sub(
                        "enforce a 0 kcal deficit (RED-S clinical safeguard active)",
                        remediated_text,
                    )

            # 2. Enforce 100% Whole-Food Pescatarian Invariant
            for forbidden, replacement in self.NON_PESCATARIAN_FORBIDDEN_TERMS.items():
                pattern = re.compile(rf"\b{re.escape(forbidden)}\b", flags=re.IGNORECASE)
                if pattern.search(remediated_text):
                    violations_detected.append(f"DIETARY_POLICY_VIOLATION:{forbidden.upper()}")
                    remediated_text = pattern.sub(replacement, remediated_text)

            # 3. Enforce 14-Day Post-Marathon Eccentric Lifting Lockout
            if days_post_marathon < 14:
                heavy_lift_regex = re.compile(
                    r"(?i)heavy\s+(?:barbell\s+)?(?:rdl|romanian\s+deadlift|back\s+squat|split\s+squat)"
                )
                if heavy_lift_regex.search(remediated_text):
                    violations_detected.append("POST_MARATHON_14D_LIFT_LOCKOUT_VIOLATION")
                    remediated_text = heavy_lift_regex.sub(
                        "Stage 1 90-90 diaphragmatic decompression (14-day post-marathon lifting lockout active)",
                        remediated_text,
                    )

            # Compute Self-Evaluation Policy Score
            penalty = 0.15 * len(violations_detected)
            initial_self_eval_score = round(max(0.0, 1.0 - penalty), 2)
            post_remediation_score = 1.0

            evaluation_report = {
                "passed_initial_eval": len(violations_detected) == 0,
                "initial_self_eval_score": initial_self_eval_score,
                "post_remediation_score": post_remediation_score,
                "violations_detected": violations_detected,
                "reds_gate_checked": True,
                "pescatarian_policy_checked": True,
                "eccentric_lockout_checked": True,
                "validated_response": remediated_text,
            }

            logger.info(
                f"Completed ADK Self-Eval Guardrail pass (score={post_remediation_score}, violations={len(violations_detected)})",
                extra={
                    "event_type": "GUARDRAIL_OUTPUT_SELF_EVAL",
                    "metadata": {
                        "initial_score": initial_self_eval_score,
                        "violations": violations_detected,
                    },
                },
            )
            return evaluation_report
