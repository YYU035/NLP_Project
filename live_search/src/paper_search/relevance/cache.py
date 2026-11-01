"""Durable reservations and conservative local quota accounting."""

import json
import sqlite3
import time
from pathlib import Path


class Cache:
    """Reserve before sending: interrupted calls never retry automatically."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS judgments (
                key TEXT PRIMARY KEY, state TEXT NOT NULL,
                value TEXT, timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                timestamp REAL NOT NULL, key TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                name TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)

    def get(self, key: str) -> tuple[str, dict | None] | None:
        """Return persisted state, including ambiguous interrupted requests."""
        row = self.db.execute(
            "SELECT state,value FROM judgments WHERE key=?", (key,)
        ).fetchone()
        return (
            (row[0], json.loads(row[1]) if row[1] else None)
            if row
            else None
        )

    def setting(self, name: str) -> str | None:
        """Read cooldown/model-version state."""
        row = self.db.execute(
            "SELECT value FROM settings WHERE name=?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_setting(self, name: str, value: str) -> None:
        """Persist a safe nonsecret state value."""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                (name, value),
            )

    def reserve(
        self, key: str, daily_limit: int, now: float
    ) -> bool:
        """Atomically enforce one attempt per key and a rolling 24h cap."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.get(key):
                return False
            count = self.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE timestamp>?",
                (now - 86400,),
            ).fetchone()[0]
            if count >= daily_limit:
                return False
            self.db.execute(
                "INSERT INTO judgments VALUES (?,?,?,?)",
                (key, "inflight", None, now),
            )
            self.db.execute(
                "INSERT INTO attempts VALUES (?,?)", (now, key)
            )
        return True

    def finish(self, key: str, state: str, value: dict) -> None:
        """Persist judgments or terminal failures without secret text."""
        with self.db:
            self.db.execute(
                "UPDATE judgments SET state=?,value=? WHERE key=?",
                (state, json.dumps(value), key),
            )

    def rejected(self, key: str, cooldown: float) -> None:
        """A 429 is an explicit rejection; resume only after its cooldown."""
        with self.db:
            self.db.execute(
                "DELETE FROM judgments WHERE key=?", (key,)
            )
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                ("not_before", str(cooldown)),
            )

    def close(self) -> None:
        """Release the SQLite connection explicitly."""
        self.db.close()

    def wait_seconds(
        self, interval: float, current: float
    ) -> float:
        """Maintain paced starts across resumptions, not just within a run."""
        last = self.db.execute(
            "SELECT MAX(timestamp) FROM attempts"
        ).fetchone()[0]
        return max(0.0, (last or 0) + interval - current)


def now() -> float:
    """Wall time for persistent quota and cooldown windows."""
    return time.time()
