"""Rate-limited parallel execution for LLM and translation API calls."""

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

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
        """Block the calling thread until it may proceed.

        Each caller reserves the next free time slot atomically under the
        lock, then sleeps at most once until that slot arrives.
        """
        with self._lock:
            now = time.monotonic()
            wake_at = max(now, self._next_allowed)
            self._next_allowed = wake_at + self._interval
        delay = wake_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def parallel_apply(
    func,
    items: list,
    max_workers: int = 4,
    rps: float = 4.0,
    desc: str = "",
) -> list:
    """Apply *func* to each item with a shared RPS budget.

    Results are returned in the same order as *items*.
    Failures yield ``None`` and a warning — they do not abort the rest.

    ``max_workers <= 1`` runs sequentially (no ThreadPoolExecutor) — useful
    for diagnostics in Jupyter/Kaggle, where widget bars may mis-render.
    """
    if not items:
        return []

    print(f"[parallel_apply] start: {desc or 'processing'} "
          f"n={len(items)} workers={max_workers} rps={rps}",
          file=sys.stderr, flush=True)

    limiter = RateLimiter(rps)
    results: list = [None] * len(items)
    bar = tqdm(
        total=len(items),
        desc=desc or "processing",
        file=sys.stderr,
        mininterval=0.3,
        dynamic_ncols=True,
    )

    def _run(idx: int):
        limiter.acquire()
        try:
            return func(items[idx])
        except Exception as exc:
            logger.warning("parallel_apply: item %d failed: %s", idx, exc)
            return None

    try:
        if max_workers <= 1:
            for i in range(len(items)):
                results[i] = _run(i)
                bar.update(1)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, i): i for i in range(len(items))}
            print(f"[parallel_apply] submitted {len(futures)} futures",
                  file=sys.stderr, flush=True)
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.warning("parallel_apply: item %d raised: %s", idx, exc)
                done += 1
                bar.update(1)
                if done <= 5 or done % 50 == 0:
                    print(f"[parallel_apply] {desc}: {done}/{len(items)}",
                          file=sys.stderr, flush=True)
    finally:
        bar.close()

    return results
