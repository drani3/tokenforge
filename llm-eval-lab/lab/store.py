"""SQLite cache of model calls, so identical requests are never re-run."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .client import ChatResult


def request_key(model: str, messages: list[dict], options: dict, sample: int = 0) -> str:
    """Stable hash of a request. `sample` distinguishes repeated draws (e.g. self-consistency)."""
    payload = json.dumps(
        {"model": model, "messages": messages, "options": options, "sample": sample},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                request TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )

    def get(self, key: str) -> ChatResult | None:
        row = self.db.execute("SELECT result FROM calls WHERE key = ?", (key,)).fetchone()
        return ChatResult(**json.loads(row[0])) if row else None

    def put(self, key: str, model: str, request: dict, result: ChatResult) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?)",
            (key, model, json.dumps(request), json.dumps(result.to_dict()),
             datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()
