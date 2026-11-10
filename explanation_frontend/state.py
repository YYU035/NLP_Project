"""Per-session request identity and cancellable-result background work."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable

from .models import digest


@dataclass
class Workspace:
    request_key: str | None = None
    token: str | None = None
    result: Any = None

    def begin(self, claim: str, options: dict) -> str:
        key = digest({"claim": claim, "options": options})
        if key != self.request_key:
            self.request_key, self.token, self.result = key, uuid.uuid4().hex, None
        return self.token

    def accept(self, token: str, result: Any) -> bool:
        if self.token is None or token != self.token:
            return False
        self.result = result
        return True

    def invalidate(self):
        self.request_key = self.token = self.result = None


@dataclass
class Job:
    future: Future
    key: str
    deadline: float
    expired: bool = False


def start_job(work: Callable, key: str, *, timeout: float = 50) -> Job:
    future = Future()
    def run():
        try:
            future.set_result(work())
        except Exception:
            future.set_exception(RuntimeError("BACKGROUND_PROCESSING_FAILURE"))
    threading.Thread(target=run, daemon=True).start()
    return Job(future, key, time.monotonic() + timeout)


def poll_job(job: Job) -> tuple[str, Any]:
    if job.expired or time.monotonic() > job.deadline:
        job.expired = True
        return "timeout", None
    if not job.future.done():
        return "pending", None
    try:
        return "done", job.future.result()
    except Exception:
        return "failed", None


def install_discovery_state(state, snapshot):
    """Shared UI handoff for imported, live and demonstrator candidate snapshots."""
    state.discovery_snapshot=snapshot
    if state.get('discovery_claim')!=snapshot.result.original_claim:
        state.discovery_claim=snapshot.result.original_claim
    state.discovery_selection={p.identity:p.identity in snapshot.selected_paper_ids for p in snapshot.result.papers}
    state.discovery_page=1
    state.discovery_filter=''
    state.discovery_provider_filter=[]
    state.discovery_pdf_only=False
    state.pop('discovery_inspect',None)
