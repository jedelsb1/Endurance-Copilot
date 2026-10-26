"""
Human-in-the-Loop (HITL) Confirmation Gate (agents/hitl.py)
Human-in-the-Loop Hooks
High-stakes actions include explicit code stops requiring human confirmation before execution.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.memory_compaction import get_db_connection
from core.observability import get_structured_logger, start_trace_span
from core.pii_redaction import scrub_pii

logger = get_structured_logger("EnduranceCopilot.HITL")

HIGH_STAKES_ACTIONS = {
    "dispatch_garmin_forerunner_970_workout_payload",
    "build_periodized_vdot_macrocycle_plan",
    "execute_emergency_quality_workout_cancellation",
}


def _init_hitl_table(db_file: Optional[Path] = None) -> None:
    """Ensures the `hitl_approvals` table exists in SQLite WAL."""
    conn = get_db_connection(db_file)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hitl_approvals (
                    approval_id TEXT PRIMARY KEY,
                    action_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    risk_rationale TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_HUMAN_CONFIRMATION',
                    decided_by TEXT DEFAULT NULL,
                    decision_note TEXT DEFAULT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT DEFAULT NULL
                );
            """)
    finally:
        conn.close()


class HumanInTheLoopGate:
    """
    Human-in-the-Loop Hooks
    Enforces an explicit code execution stop (`PENDING_HUMAN_CONFIRMATION`) before
    executing high-stakes mutations (e.g., pushing workouts to Garmin Forerunner 970
    watch hardware, overwriting a multi-month macrocycle, or cancelling a primary Q1/Q2 workout).
    """

    def __init__(self, db_file: Optional[Path] = None) -> None:
        self.db_file = db_file
        _init_hitl_table(self.db_file)

    def create_approval_checkpoint(
        self,
        action_name: str,
        payload: Dict[str, Any],
        risk_rationale: str,
    ) -> Dict[str, Any]:
        """
        Halts automatic execution, persists the staged action to SQLite WAL,
        and returns a `PENDING_HUMAN_CONFIRMATION` checkpoint object.
        """
        with start_trace_span(
            "HumanInTheLoopGate.create_approval_checkpoint",
            attributes={"action_name": action_name},
        ):
            _init_hitl_table(self.db_file)
            approval_id = f"hitl_{uuid.uuid4().hex[:12]}"
            now_iso = datetime.now(timezone.utc).isoformat()
            clean_payload = scrub_pii(payload)

            conn = get_db_connection(self.db_file)
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO hitl_approvals (
                            approval_id, action_name, payload_json, risk_rationale, status, created_at
                        ) VALUES (?, ?, ?, ?, 'PENDING_HUMAN_CONFIRMATION', ?)
                        """,
                        (
                            approval_id,
                            action_name,
                            json.dumps(clean_payload, ensure_ascii=False),
                            risk_rationale,
                            now_iso,
                        ),
                    )
            finally:
                conn.close()

            logger.info(
                f"HITL code stop triggered for high-stakes action '{action_name}' (approval_id={approval_id})",
                extra={
                    "event_type": "HITL_CONFIRMATION_REQUIRED",
                    "metadata": {
                        "approval_id": approval_id,
                        "action_name": action_name,
                        "risk_rationale": risk_rationale,
                    },
                },
            )

            return {
                "status": "PENDING_HUMAN_CONFIRMATION",
                "hitl_approval_id": approval_id,
                "action_name": action_name,
                "staged_payload": clean_payload,
                "risk_rationale": risk_rationale,
                "confirmation_endpoint": f"/api/hitl/confirm/{approval_id}",
                "created_at": now_iso,
            }

    def confirm_action(
        self,
        approval_id: str,
        approved_by: str = "athlete_jesse",
        note: str = "Confirmed by athlete",
        executor_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Approves a pending HITL checkpoint and optionally executes the staged callback.
        """
        _init_hitl_table(self.db_file)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_file)
        try:
            cur = conn.execute(
                "SELECT approval_id, action_name, payload_json, risk_rationale, status "
                "FROM hitl_approvals WHERE approval_id = ?",
                (approval_id,),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "status": "error",
                    "error_code": "HITL_APPROVAL_NOT_FOUND",
                    "message": f"Approval ID '{approval_id}' was not found.",
                }

            with conn:
                conn.execute(
                    """
                    UPDATE hitl_approvals
                    SET status = 'APPROVED_AND_EXECUTED',
                        decided_by = ?,
                        decision_note = ?,
                        decided_at = ?
                    WHERE approval_id = ?
                    """,
                    (approved_by, note, now_iso, approval_id),
                )
        finally:
            conn.close()

        staged_payload = json.loads(row["payload_json"] or "{}")
        execution_result = None
        if executor_callback is not None:
            execution_result = executor_callback(staged_payload)

        logger.info(
            f"HITL checkpoint '{approval_id}' confirmed by {approved_by}",
            extra={
                "event_type": "HITL_ACTION_CONFIRMED",
                "metadata": {
                    "approval_id": approval_id,
                    "action_name": row["action_name"],
                    "approved_by": approved_by,
                },
            },
        )

        return {
            "status": "APPROVED_AND_EXECUTED",
            "hitl_approval_id": approval_id,
            "action_name": row["action_name"],
            "approved_by": approved_by,
            "decided_at": now_iso,
            "staged_payload": staged_payload,
            "execution_result": execution_result,
        }

    def reject_action(
        self,
        approval_id: str,
        rejected_by: str = "athlete_jesse",
        reason: str = "Declined by athlete",
    ) -> Dict[str, Any]:
        """Rejects and cancels a staged high-stakes action."""
        _init_hitl_table(self.db_file)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_file)
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE hitl_approvals
                    SET status = 'REJECTED_BY_HUMAN',
                        decided_by = ?,
                        decision_note = ?,
                        decided_at = ?
                    WHERE approval_id = ?
                    """,
                    (rejected_by, reason, now_iso, approval_id),
                )
        finally:
            conn.close()

        return {
            "status": "REJECTED_BY_HUMAN",
            "hitl_approval_id": approval_id,
            "rejected_by": rejected_by,
            "reason": reason,
            "decided_at": now_iso,
        }

    def list_pending_approvals(self) -> List[Dict[str, Any]]:
        """Lists all pending high-stakes actions awaiting human confirmation."""
        _init_hitl_table(self.db_file)
        conn = get_db_connection(self.db_file)
        try:
            cur = conn.execute(
                "SELECT approval_id, action_name, payload_json, risk_rationale, status, created_at "
                "FROM hitl_approvals WHERE status = 'PENDING_HUMAN_CONFIRMATION' ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
            return [
                {
                    "hitl_approval_id": r["approval_id"],
                    "action_name": r["action_name"],
                    "staged_payload": json.loads(r["payload_json"] or "{}"),
                    "risk_rationale": r["risk_rationale"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()
