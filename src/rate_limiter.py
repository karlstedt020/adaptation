"""Rate-limited parallel execution for LLM and translation API calls."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Each call to ``acquire()`` blocks until the global rate (≤ rps) allows.
    Works correctly across all threads: every request takes the *next*
    available time slot, so bursts are impossible.
    """

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError(f"rps must be positive, got {rps}")
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self) -> None:
        """Block the calling thread until it may proceed."""
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                # Reserve the next slot for this thread
                wake_at = self._next_allowed
                self._next_allowed += self._interval
            time.sleep(max(0.0, wake_at - time.monotonic()))


def parallel_apply(
    func,
    items: list,
    max_workers: int = 4,
    rps: float = 4.0,
    desc: str = "",
) -> list:
    """Apply *func* to each item in parallel with a shared RPS budget.

    Results are returned in the **same order** as *items*.
    Failed items produce ``None`` and emit a warning (they do not abort
    the remaining work).

    Args:
        func:        callable(item) → result
        items:       list of inputs
        max_workers: thread-pool size
        rps:         maximum requests per second across all threads
        desc:        tqdm progress-bar label
    """
    if not items:
        return []

    limiter = RateLimiter(rps)
    results: list = [None] * len(items)

    def _bounded(idx: int):
        limiter.acquire()
        return idx, func(items[idx])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_bounded, i): i for i in range(len(items))}
        with tqdm(total=len(items), desc=desc or "processing") as bar:
            for fut in as_completed(futures):
                bar.update(1)
                if fut.exception() is not None:
                    logger.warning(
                        "parallel_apply: item %d failed: %s",
                        futures[fut], fut.exception(),
                    )
                else:
                    idx, val = fut.result()
                    results[idx] = val

    return results
