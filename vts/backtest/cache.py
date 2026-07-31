"""On-disk cache for LLM decisions, keyed by (ticker, date, model, prompt-hash, sample).

Step 2 mandate: *"동일 (ticker, date) 호출 결과를 디스크 캐시."* Caching makes a
backtest reproducible and cheap to re-run: identical inputs return the stored
decision instead of re-calling the API. Each of the N samples is stored under its
own key so the majority vote is stable across reruns.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from vts.decision import Decision


def decision_key(
    ticker: str, as_of_iso: str, model_id: str, prompt: str, sample: int
) -> str:
    """Deterministic cache key. ``prompt`` is hashed so prompt changes bust the cache."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    raw = f"{ticker.upper()}|{as_of_iso}|{model_id}|{prompt_hash}|{sample}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DecisionCache:
    """A tiny SQLite key→Decision store."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Decision | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM decisions WHERE key = ?", (key,)
            ).fetchone()
        return Decision.model_validate_json(row[0]) if row else None

    def put(self, key: str, decision: Decision) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions (key, payload) VALUES (?, ?)",
                (key, decision.model_dump_json()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
