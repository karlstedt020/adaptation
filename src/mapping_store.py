"""Persistent mapping store for entity replacement consistency."""

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class MappingStore:
    """JSON-backed dictionary: English entity → Russian equivalent (thread-safe)."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            logger.info("Loaded %d mappings from %s", len(self._data), path)

    def get(self, original: str) -> str | None:
        with self._lock:
            return self._data.get(original)

    def add(self, original: str, adapted: str):
        with self._lock:
            self._data[original] = adapted

    def add_batch(self, pairs: dict[str, str]):
        with self._lock:
            self._data.update(pairs)

    def items(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._data.items())

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._data)

    def save(self):
        data = self.snapshot()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d mappings to %s", len(data), self._path)

    def as_context_string(self) -> str:
        """Format existing mappings for inclusion in LLM prompts."""
        items = self.items()
        if not items:
            return "No prior mappings."
        lines = [f"  {k} → {v}" for k, v in items]
        return "Known mappings:\n" + "\n".join(lines)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data
