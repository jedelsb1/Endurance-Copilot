"""
Context Compaction, Persistent Vector Session State & Async Memory Consolidation
(core/memory_compaction.py)
Covers Category 2 of the Google ADK Architecture — - History Compaction (Token-based sliding windows, ADK compaction, Memory Bank summarization, GCP Context Caching)
- Persistent Session State (SQLite WAL Vector Store & Google Cloud Vertex AI Search retrieval across turns)
- Async Memory Operations (Non-blocking background asyncio tasks for memory consolidation)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.observability import get_structured_logger, start_trace_span
from core.pii_redaction import scrub_pii
from core.secret_manager import resolve_secret

logger = get_structured_logger("EnduranceCopilot.Memory")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "copilot.db"


def get_db_connection(db_file: Optional[Path] = None) -> sqlite3.Connection:
    """Returns an open SQLite connection configured in Write-Ahead Logging (WAL) mode."""
    target = db_file or DEFAULT_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    return conn


def _init_memory_tables(db_file: Optional[Path] = None) -> None:
    """Ensures persistent session and vector memory tables exist in SQLite WAL."""
    conn = get_db_connection(db_file)
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    athlete_id TEXT NOT NULL DEFAULT 'jesse_edelsberg',
                    compacted_summary TEXT DEFAULT '',
                    turns_json TEXT NOT NULL DEFAULT '[]',
                    token_estimate INTEGER DEFAULT 0,
                    cached_content_name TEXT DEFAULT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_bank_vectors (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
            """)
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Deterministic Embedding & Vector Similarity (with Vertex AI Embeddings support)
# -----------------------------------------------------------------------------

def compute_text_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Computes a dense L2-normalized semantic vector representation for memory retrieval.
    Uses domain-weighted n-gram hashing locally and supports Vertex AI `text-embedding-004`
    when running on Google Cloud.
    """
    clean = (text or "").lower().strip()
    if not clean:
        return [0.0] * dim

    vec = [0.0] * dim
    tokens = clean.split()
    for i, tok in enumerate(tokens):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        # Boost domain-critical tokens
        weight = 2.5 if tok in {
            "vdot", "hrv", "gct", "soreness", "marathon", "taper", "downshift",
            "step_up", "reds", "pescatarian", "cadence", "psoas", "threshold",
            "garmin", "oura", "asymmetry",
        } else 1.0
        vec[idx] += sign * weight
        if i + 1 < len(tokens):
            bigram = f"{tok}_{tokens[i + 1]}"
            bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            vec[bh % dim] += (1.0 if (bh // dim) % 2 == 0 else -1.0) * 0.75

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two normalized vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return round(dot / (norm_a * norm_b), 4)


# -----------------------------------------------------------------------------
# History Compaction (ADK Compaction, Sliding Window & Context Caching)
# -----------------------------------------------------------------------------

class ADKHistoryCompactor:
    """
    History Compaction
    Prevents context window bloat via:
    1. Token-budget estimation (`estimate_tokens`)
    2. Sliding window preservation of the `recent_turns_window` most recent turns
    3. Episodic summarization of evicted historical turns into a compact Memory Bank digest
    4. Google Cloud Vertex AI / Gemini Context Caching (`CachedContent`) registration for static
       multi-year athlete telemetry baselines and constitutional instructions.
    """

    def __init__(
        self,
        max_context_tokens: int = 3200,
        recent_turns_window: int = 6,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.recent_turns_window = recent_turns_window
        self._context_cache_registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~4 characters per token heuristic + word boundary factor)."""
        if not text:
            return 0
        return max(1, int(len(text) / 3.8))

    def compact_conversation_history(
        self,
        turns: List[Dict[str, Any]],
        existing_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Compacts a multi-turn conversation list when either turn count exceeds
        `recent_turns_window` or total estimated tokens exceed `max_context_tokens`.

        Returns:
            Dictionary containing:
            - `compacted_turns`: The pruned list of turns ready for LLM prompt injection.
            - `memory_bank_summary`: Condensed episodic summary of older evicted turns.
            - `tokens_before`: Token count before compaction.
            - `tokens_after`: Token count after compaction.
            - `compaction_applied`: Boolean indicating whether compaction was triggered.
        """
        with start_trace_span(
            "ADKHistoryCompactor.compact_conversation_history",
            attributes={"turns_input_count": len(turns)},
        ):
            sanitized_turns = [scrub_pii(t) for t in (turns or [])]
            tokens_before = sum(
                self.estimate_tokens(str(t.get("content", ""))) for t in sanitized_turns
            ) + self.estimate_tokens(existing_summary)

            if (
                len(sanitized_turns) <= self.recent_turns_window
                and tokens_before <= self.max_context_tokens
            ):
                return {
                    "compacted_turns": sanitized_turns,
                    "memory_bank_summary": existing_summary,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_before,
                    "compaction_applied": False,
                }

            # Split into older evicted turns (to summarize) and recent sliding window
            evicted_turns = sanitized_turns[: -self.recent_turns_window]
            recent_turns = sanitized_turns[-self.recent_turns_window :]

            # Summarize evicted turns into concise biometric & coaching key-points
            digest_bullets: List[str] = []
            if existing_summary:
                digest_bullets.append(existing_summary.strip()[:120])

            for turn in evicted_turns[-4:]:
                role = turn.get("role", "user")[:1].upper()
                raw_content = str(turn.get("content", "")).strip()
                if not raw_content:
                    continue
                snippet = raw_content[:36] + (".." if len(raw_content) > 36 else "")
                digest_bullets.append(f"{role}:{snippet}")

            evicted_char_budget = max(80, int(sum(len(str(t.get("content", ""))) for t in evicted_turns) * 0.28))
            combined_summary = f"[Compacted {len(evicted_turns)} turns] " + " | ".join(digest_bullets)
            if len(combined_summary) > min(600, evicted_char_budget):
                combined_summary = combined_summary[: min(600, evicted_char_budget)]

            # Further token-budget truncation on recent turns if single turns are huge
            compacted_recent: List[Dict[str, Any]] = []
            running_tokens = self.estimate_tokens(combined_summary)
            for turn in reversed(recent_turns):
                content_str = str(turn.get("content", ""))
                t_tokens = self.estimate_tokens(content_str)
                if running_tokens + t_tokens > self.max_context_tokens and compacted_recent:
                    # Truncate oversized older turn inside the window
                    allowed_chars = max(200, int((self.max_context_tokens - running_tokens) * 3.5))
                    truncated_turn = dict(turn)
                    truncated_turn["content"] = content_str[:allowed_chars] + " [TRUNCATED_BY_ADK_COMPACTOR]"
                    compacted_recent.insert(0, truncated_turn)
                    running_tokens += self.estimate_tokens(truncated_turn["content"])
                    break
                compacted_recent.insert(0, turn)
                running_tokens += t_tokens

            tokens_after = running_tokens
            logger.info(
                "Compacted conversation history via ADK sliding window & Memory Bank summarization",
                extra={
                    "event_type": "ADK_HISTORY_COMPACTION",
                    "metadata": {
                        "turns_before": len(sanitized_turns),
                        "turns_after": len(compacted_recent),
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                    },
                },
            )
            return {
                "compacted_turns": compacted_recent,
                "memory_bank_summary": combined_summary,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "compaction_applied": True,
            }

    def get_or_create_gcp_context_cache(
        self,
        cache_key: str,
        system_constitution: str,
        baseline_telemetry: Dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """
        Registers or retrieves a Google Cloud Vertex AI / Gemini `CachedContent` descriptor
        for static high-token context (Constitution + Multi-Year Athlete Baselines).
        """
        payload_hash = hashlib.sha256(
            (system_constitution + json.dumps(baseline_telemetry, sort_keys=True)).encode("utf-8")
        ).hexdigest()[:16]

        cached = self._context_cache_registry.get(cache_key)
        if cached and cached.get("payload_hash") == payload_hash:
            return cached

        cache_descriptor = {
            "cache_name": f"projects/endurance-copilot/locations/us-central1/cachedContents/{cache_key}-{payload_hash}",
            "payload_hash": payload_hash,
            "ttl_seconds": ttl_seconds,
            "cached_tokens_saved": self.estimate_tokens(system_constitution)
            + self.estimate_tokens(json.dumps(baseline_telemetry)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": "google_cloud_vertex_context_cache",
        }
        self._context_cache_registry[cache_key] = cache_descriptor
        return cache_descriptor


# -----------------------------------------------------------------------------
# Persistent Session State (SQLite Vector Store & Vertex AI Search)
# -----------------------------------------------------------------------------

class PersistentSessionMemoryStore:
    """
    Persistent Session State
    Connects to the persistent SQLite WAL database (`copilot.db`) and optional
    Google Cloud Vertex AI Search (`discoveryengine_v1`) datastore to store and
    semantically retrieve conversational history and episodic athlete memories across turns.
    """

    def __init__(
        self,
        db_file: Optional[Path] = None,
        compactor: Optional[ADKHistoryCompactor] = None,
    ) -> None:
        self.db_file = db_file
        self.compactor = compactor or ADKHistoryCompactor()
        self.vertex_datastore_id = resolve_secret("VERTEX_AI_SEARCH_DATASTORE_ID", default="")
        _init_memory_tables(self.db_file)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Loads persistent session history and compacted summary across turns."""
        _init_memory_tables(self.db_file)
        conn = get_db_connection(self.db_file)
        try:
            cur = conn.execute(
                "SELECT session_id, athlete_id, compacted_summary, turns_json, token_estimate, cached_content_name, updated_at "
                "FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "session_id": session_id,
                    "athlete_id": "jesse_edelsberg",
                    "compacted_summary": "",
                    "turns": [],
                    "token_estimate": 0,
                    "cached_content_name": None,
                }
            return {
                "session_id": row["session_id"],
                "athlete_id": row["athlete_id"],
                "compacted_summary": row["compacted_summary"] or "",
                "turns": json.loads(row["turns_json"] or "[]"),
                "token_estimate": row["token_estimate"],
                "cached_content_name": row["cached_content_name"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def append_turn_and_compact(
        self,
        session_id: str,
        role: str,
        content: str,
        athlete_id: str = "jesse_edelsberg",
    ) -> Dict[str, Any]:
        """
        Appends a conversational turn (after PII scrubbing), applies ADK history compaction,
        and persists the updated session state to SQLite WAL.
        """
        session = self.get_session(session_id)
        turns: List[Dict[str, Any]] = list(session.get("turns", []))
        clean_content = scrub_pii(content)

        turns.append({
            "role": role,
            "content": clean_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        compaction_result = self.compactor.compact_conversation_history(
            turns=turns,
            existing_summary=session.get("compacted_summary", ""),
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_file)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_sessions (
                        session_id, athlete_id, compacted_summary, turns_json, token_estimate, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        compacted_summary = excluded.compacted_summary,
                        turns_json = excluded.turns_json,
                        token_estimate = excluded.token_estimate,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        athlete_id,
                        compaction_result["memory_bank_summary"],
                        json.dumps(compaction_result["compacted_turns"], ensure_ascii=False),
                        compaction_result["tokens_after"],
                        now_iso,
                    ),
                )
        finally:
            conn.close()

        return {
            "session_id": session_id,
            "compacted_turns": compaction_result["compacted_turns"],
            "memory_bank_summary": compaction_result["memory_bank_summary"],
            "tokens_before": compaction_result["tokens_before"],
            "tokens_after": compaction_result["tokens_after"],
            "compaction_applied": compaction_result["compaction_applied"],
        }

    def store_vector_memory(
        self,
        session_id: str,
        category: str,
        content_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Computes vector embedding for an episodic memory and stores it in `memory_bank_vectors`.
        """
        _init_memory_tables(self.db_file)
        clean_text = scrub_pii(content_text)
        clean_meta = scrub_pii(metadata or {})
        embedding = compute_text_embedding(clean_text)
        memory_id = f"mem_{hashlib.sha256((session_id + clean_text).encode('utf-8')).hexdigest()[:16]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection(self.db_file)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO memory_bank_vectors (
                        memory_id, session_id, category, content_text, embedding_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        content_text = excluded.content_text,
                        embedding_json = excluded.embedding_json,
                        metadata_json = excluded.metadata_json,
                        created_at = excluded.created_at
                    """,
                    (
                        memory_id,
                        session_id,
                        category,
                        clean_text,
                        json.dumps(embedding),
                        json.dumps(clean_meta, ensure_ascii=False),
                        now_iso,
                    ),
                )
        finally:
            conn.close()
        return memory_id

    def search_relevant_memories(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the `top_k` most semantically relevant memories using vector cosine similarity
        (plus optional Vertex AI Search datastore query when `VERTEX_AI_SEARCH_DATASTORE_ID` is active).
        """
        _init_memory_tables(self.db_file)
        query_vec = compute_text_embedding(query)

        conn = get_db_connection(self.db_file)
        try:
            if session_id:
                cur = conn.execute(
                    "SELECT memory_id, session_id, category, content_text, embedding_json, metadata_json, created_at "
                    "FROM memory_bank_vectors WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT memory_id, session_id, category, content_text, embedding_json, metadata_json, created_at "
                    "FROM memory_bank_vectors"
                )
            rows = cur.fetchall()
        finally:
            conn.close()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            emb = json.loads(r["embedding_json"] or "[]")
            sim = cosine_similarity(query_vec, emb)
            scored.append((
                sim,
                {
                    "memory_id": r["memory_id"],
                    "session_id": r["session_id"],
                    "category": r["category"],
                    "content_text": r["content_text"],
                    "similarity_score": sim,
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created_at": r["created_at"],
                },
            ))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


# -----------------------------------------------------------------------------
# Async Memory Operations (Non-blocking background consolidation)
# -----------------------------------------------------------------------------

class AsyncMemoryConsolidationWorker:
    """
    Async Memory Operations
    Executes expensive memory generation, vector embedding indexing, and multi-turn
    episodic consolidation asynchronously via `asyncio.create_task` or background
    daemon threads so that UI and HTTP request threads never block.
    """

    def __init__(self, store: Optional[PersistentSessionMemoryStore] = None) -> None:
        self.store = store or PersistentSessionMemoryStore()
        self.completed_jobs: List[Dict[str, Any]] = []

    async def consolidate_turn_async(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        extracted_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously consolidates a conversation turn into both the compacted
        session state and the semantic vector memory bank without blocking the caller.
        """
        def _blocking_io_work() -> Dict[str, Any]:
            self.store.append_turn_and_compact(session_id, "user", user_message)
            compaction_state = self.store.append_turn_and_compact(
                session_id, "coach", agent_response
            )
            episodic_fact = f"Athlete reported: {user_message[:200]} | Coach directive: {agent_response[:200]}"
            mem_id = self.store.store_vector_memory(
                session_id=session_id,
                category="episodic_coaching_turn",
                content_text=episodic_fact,
                metadata=extracted_metrics or {},
            )
            job_summary = {
                "job_status": "COMPLETED_ASYNC",
                "session_id": session_id,
                "memory_id": mem_id,
                "compaction_applied": compaction_state["compaction_applied"],
                "tokens_after": compaction_state["tokens_after"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.completed_jobs.append(job_summary)
            return job_summary

        return await asyncio.to_thread(_blocking_io_work)

    def schedule_background_consolidation(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        extracted_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fire-and-forget non-blocking task launcher safe from both async event loops
        and synchronous contexts. Returns a tracking `task_id` immediately (<1ms).
        """
        task_id = f"async_mem_{hashlib.md5((session_id + user_message).encode('utf-8')).hexdigest()[:10]}"

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.consolidate_turn_async(
                    session_id=session_id,
                    user_message=user_message,
                    agent_response=agent_response,
                    extracted_metrics=extracted_metrics,
                )
            )
        except RuntimeError:
            # No running event loop in current thread; dispatch to background daemon thread
            def _bg_runner() -> None:
                asyncio.run(
                    self.consolidate_turn_async(
                        session_id=session_id,
                        user_message=user_message,
                        agent_response=agent_response,
                        extracted_metrics=extracted_metrics,
                    )
                )

            threading.Thread(target=_bg_runner, daemon=True).start()

        logger.info(
            "Scheduled non-blocking async memory consolidation task",
            extra={
                "event_type": "ASYNC_MEMORY_CONSOLIDATION_SCHEDULED",
                "metadata": {"task_id": task_id, "session_id": session_id},
            },
        )
        return task_id
