"""Incremental checkpoint for long-running translation jobs.

Usage pattern:
    ck = Checkpointer(path, result_cols=["text_ru"])
    done_ids, existing = ck.load()
    # ... process only rows not in done_ids ...
    full_df = ck.merge_and_save(existing, new_results_df)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class Checkpointer:
    """Load/save translation progress so crashed jobs can resume."""

    def __init__(self, path: Path, result_cols: list[str]) -> None:
        self._path = path
        self._result_cols = result_cols

    def load(self) -> tuple[set, pd.DataFrame | None]:
        """Return (set of already-done original indices, checkpoint df or None)."""
        if not self._path.exists():
            return set(), None

        logger.info("Checkpoint found at %s, loading …", self._path)
        existing = self._read()
        # A row counts as done if ALL result columns are non-empty strings
        mask = existing[self._result_cols].apply(
            lambda col: col.notna() & (col.astype(str) != "")
        ).all(axis=1)
        done = set(existing.index[mask].tolist())
        logger.info("Checkpoint: %d rows already done", len(done))
        return done, existing

    def save(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write(df)
        logger.info("Checkpoint saved: %d rows → %s", len(df), self._path)

    def merge_and_save(
        self, existing: pd.DataFrame | None, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge existing checkpoint with new results and save."""
        if existing is None:
            merged = new_df
        else:
            merged = pd.concat(
                [existing[~existing.index.isin(new_df.index)], new_df]
            ).sort_index()
        self.save(merged)
        return merged

    # ── I/O helpers ────────────────────────────────────────

    def _write(self, df: pd.DataFrame) -> None:
        if self._path.suffix == ".jsonl":
            df.to_json(
                self._path, orient="records",
                lines=True, force_ascii=False,
            )
        else:
            df.to_csv(self._path, index=True)

    def _read(self) -> pd.DataFrame:
        if self._path.suffix == ".jsonl":
            return pd.read_json(self._path, lines=True)
        return pd.read_csv(self._path, index_col=0)
