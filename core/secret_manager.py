"""
Secure Secret Management Layer (core/secret_manager.py)
Secure Secret Management
No hardcoded API keys; all tools and clients leverage a secure injection method
such as Google Cloud Secret Manager with environment variable injection fallback.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("SecretManager")

BASE_DIR = Path(__file__).resolve().parent.parent


class SecretManagerClient:
    """
    Unified client for retrieving sensitive API keys and OAuth credentials via
    Google Cloud Secret Manager (`google.cloud.secretmanager`) or runtime
    environment injection without ever hardcoding credentials in source code.
    """

    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = (
            project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
        )
        self._gcp_client = None
        self._init_gcp_client()

    def _init_gcp_client(self) -> None:
        """ Lazily initializes the Google Cloud Secret Manager client if installed and configured. """
        if not self.project_id:
            return
        try:
            from google.cloud import secretmanager  # type: ignore
            self._gcp_client = secretmanager.SecretManagerServiceClient()
        except Exception:
            self._gcp_client = None

    def get_secret(
        self,
        secret_id: str,
        version_id: str = "latest",
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves a secret by ID using a strict priority chain:
        1. Google Cloud Secret Manager (`projects/{project_id}/secrets/{secret_id}/versions/{version_id}`)
        2. Runtime Environment Variable (`os.environ[secret_id]`)
        3. Untracked local `.env` / `oura.env` files (for local development only)

        Args:
            secret_id: Canonical identifier of the secret (e.g., 'GEMINI_API_KEY', 'OURA_PERSONAL_ACCESS_TOKEN').
            version_id: Secret version in GCP Secret Manager (defaults to 'latest').
            default: Default value returned if the secret is not configured.

        Returns:
            Resolved secret string or `default` if unavailable.
        """
        # 1. Try Google Cloud Secret Manager first when configured
        if self._gcp_client and self.project_id:
            gcp_secret_name = secret_id.lower().replace("_", "-")
            resource_name = f"projects/{self.project_id}/secrets/{gcp_secret_name}/versions/{version_id}"
            try:
                response = self._gcp_client.access_secret_version(request={"name": resource_name})
                payload = response.payload.data.decode("UTF-8").strip()
                if payload:
                    return payload
            except Exception as exc:
                logger.debug(
                    "GCP Secret Manager lookup fallback",
                    extra={"secret_id": secret_id, "reason": str(exc)},
                )

        # 2. Check injected runtime environment variables
        env_val = os.environ.get(secret_id, "").strip()
        if env_val:
            return env_val

        # Common alias fallback for Gemini keys
        if secret_id == "GEMINI_API_KEY":
            for alias in ("GOOGLE_API_KEY", "VERTEX_API_KEY"):
                alias_val = os.environ.get(alias, "").strip()
                if alias_val:
                    return alias_val

        # 3. Check untracked local .env / oura.env files (never committed to Git)
        local_secrets = self._load_untracked_env_files()
        if secret_id in local_secrets and local_secrets[secret_id]:
            return local_secrets[secret_id]

        return default

    @staticmethod
    def _load_untracked_env_files() -> Dict[str, str]:
        """Reads untracked `.env` and `oura.env` files from local disk if present."""
        values: Dict[str, str] = {}
        for filename in (".env", "oura.env"):
            env_path = BASE_DIR / filename
            if not env_path.exists():
                continue
            try:
                with open(env_path, "r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        values[key.strip()] = val.strip().strip('"').strip("'")
            except Exception:
                continue
        return values


@lru_cache(maxsize=1)
def get_secret_manager() -> SecretManagerClient:
    """Returns singleton SecretManagerClient instance."""
    return SecretManagerClient()


def resolve_secret(secret_id: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience helper to resolve a secret securely without hardcoding."""
    return get_secret_manager().get_secret(secret_id, default=default)
