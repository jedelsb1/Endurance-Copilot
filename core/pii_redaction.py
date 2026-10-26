"""
Active PII Redaction & Sensitive Data Scrubbing Pipeline (core/pii_redaction.py)
PII Redaction
Logging and memory pipelines include active scrubbing mechanisms to redact
sensitive data before storage using Google Cloud Sensitive Data Protection (DLP API)
with deterministic high-precision regex and key-based scrubbing fallback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union


class PIIRedactionEngine:
    """
    Scrubs Personally Identifiable Information (PII), OAuth tokens, API keys,
    email addresses, phone numbers, SSNs, credit cards, and precise home GPS
    coordinates from text and structured dictionaries prior to logging or DB persistence.
    Supports Google Cloud DLP (`google.cloud.dlp_v2`) when configured.
    """

    # Deterministic compiled patterns for sensitive data & credentials
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_PATTERN = re.compile(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
    )
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    API_KEY_PATTERN = re.compile(
        r"\b(?:AIza[0-9A-Za-z\-_]{20,}|AQ\.[0-9A-Za-z\-_]{20,}|ya29\.[0-9A-Za-z\-_]{20,}|ghp_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,})\b"
    )
    BEARER_PATTERN = re.compile(
        r"(?i)(Bearer\s+)([A-Za-z0-9\-._~+/]+=*)"
    )
    GPS_COORD_PATTERN = re.compile(
        r"\b(-?\d{1,2}\.\d{5,})\s*,\s*(-?\d{1,3}\.\d{5,})\b"
    )

    SENSITIVE_DICT_KEYS = {
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "api_key",
        "gemini_api_key",
        "oura_token",
        "hf_token",
        "authorization",
        "ssn",
        "credit_card",
        "phone_number",
    }

    def __init__(self, project_id: Optional[str] = None, enable_cloud_dlp: bool = False) -> None:
        self.project_id = project_id or "endurance-copilot-prod"
        self._dlp_client = None
        if enable_cloud_dlp:
            try:
                from google.cloud import dlp_v2  # type: ignore
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:
                self._dlp_client = None

    def scrub_text(self, text: str) -> str:
        """
        Redacts PII and secret credentials from arbitrary string content.
        Uses Google Cloud DLP API when active, layered with deterministic regex scrubbing.
        """
        if not text or not isinstance(text, str):
            return text

        scrubbed = text

        # 1. Optional Google Cloud DLP API pass
        if self._dlp_client and self.project_id:
            try:
                parent = f"projects/{self.project_id}/locations/global"
                inspect_config = {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                        {"name": "AUTH_TOKEN"},
                        {"name": "GCP_API_KEY"},
                    ]
                }
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }
                response = self._dlp_client.deidentify_content(
                    request={
                        "parent": parent,
                        "inspect_config": inspect_config,
                        "deidentify_config": deidentify_config,
                        "item": {"value": scrubbed},
                    }
                )
                scrubbed = response.item.value
            except Exception:
                pass

        # 2. Deterministic high-speed regex scrubbing (always enforced)
        scrubbed = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", scrubbed)
        scrubbed = self.BEARER_PATTERN.sub(r"\1[REDACTED_BEARER_TOKEN]", scrubbed)
        scrubbed = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", scrubbed)
        scrubbed = self.SSN_PATTERN.sub("[REDACTED_SSN]", scrubbed)
        scrubbed = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)
        scrubbed = self.GPS_COORD_PATTERN.sub("[REDACTED_GPS_COORDINATES]", scrubbed)
        return scrubbed

    def scrub_structure(self, data: Any) -> Any:
        """
        Recursively traverses dictionaries, lists, and primitives to redact sensitive keys
        and scrub embedded PII strings before storage or logging.
        """
        if isinstance(data, str):
            return self.scrub_text(data)
        if isinstance(data, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in data.items():
                key_lower = str(k).lower()
                if key_lower in self.SENSITIVE_DICT_KEYS:
                    cleaned[k] = "[REDACTED_SECRET]"
                else:
                    cleaned[k] = self.scrub_structure(v)
            return cleaned
        if isinstance(data, list):
            return [self.scrub_structure(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.scrub_structure(item) for item in data)
        return data


_GLOBAL_SCRUBBER = PIIRedactionEngine()


def scrub_pii(payload: Any) -> Any:
    """Module-level helper to scrub strings or nested dicts/lists prior to storage or logging."""
    return _GLOBAL_SCRUBBER.scrub_structure(payload)


class PIIRedactingLogFilter(logging.Filter):
    """
    Python `logging.Filter` attached to all structured loggers so no PII or API keys
    can ever leak into stdout, log files, or Cloud Logging sinks.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _GLOBAL_SCRUBBER.scrub_text(record.msg)
        if record.args:
            record.args = _GLOBAL_SCRUBBER.scrub_structure(record.args)
        for attr in ("metadata", "intended_arguments", "actual_outcome", "payload"):
            if hasattr(record, attr):
                setattr(
                    record,
                    attr,
                    _GLOBAL_SCRUBBER.scrub_structure(getattr(record, attr)),
                )
        return True
