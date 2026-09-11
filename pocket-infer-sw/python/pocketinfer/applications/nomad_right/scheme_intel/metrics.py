"""
Latency instrumentation: P50/P95/P99 per pipeline stage over a bounded
window (METRICS_WINDOW samples per stage - memory can never grow).

The running app records into LATENCY; every METRICS_SNAPSHOT_EVERY records
a small JSON snapshot is written atomically to METRICS_SNAPSHOT_PATH so the
diagnostics command can read live numbers from outside the process.
Nothing runs in the background.
"""

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Dict, Optional

from pocketinfer.applications.nomad_right.scheme_intel import si_constants as C


def _pct(sorted_xs, p: float) -> float:
    if not sorted_xs:
        return 0.0
    k = max(0, min(len(sorted_xs) - 1, int(round(p / 100.0 * (len(sorted_xs) - 1)))))
    return round(sorted_xs[k], 1)


class LatencyRecorder:
    def __init__(self, window: int = C.METRICS_WINDOW, snapshot_path: Optional[str] = C.METRICS_SNAPSHOT_PATH,
                 snapshot_every: int = C.METRICS_SNAPSHOT_EVERY):
        self._window = window
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._window))
        self._lock = threading.Lock()
        self._path = snapshot_path
        self._every = max(1, snapshot_every)
        self._count = 0
        self._started = time.time()

    def record(self, stage: str, ms: float) -> None:
        with self._lock:
            self._data[stage].append(float(ms))
            self._count += 1
            due = self._count % self._every == 0
        if due:
            self.write_snapshot()

    @contextmanager
    def timed(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def percentiles(self, stage: str) -> Dict[str, float]:
        with self._lock:
            xs = sorted(self._data.get(stage, ()))
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "p50": _pct(xs, 50), "p95": _pct(xs, 95), "p99": _pct(xs, 99),
                "max": round(xs[-1], 1), "min": round(xs[0], 1)}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stages = list(self._data.keys())
        return {"written_at": time.time(), "since": self._started, "pid": os.getpid(),
                "stages": {s: self.percentiles(s) for s in sorted(stages)}}

    def write_snapshot(self) -> None:
        if not self._path:
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = f"{self._path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f)
            os.replace(tmp, self._path)
        except OSError:
            pass  # instrumentation must never break a turn


LATENCY = LatencyRecorder()


def load_snapshot(path: str = C.METRICS_SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
