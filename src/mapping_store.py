"""Persistent mapping store for entity replacement consistency."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MappingStore:
    """JSON-backed dictionary: English entity → Russian equivalent."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, str] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            logger.info("Loaded %d mappings from %s", len(self._data), path)

    def get(self, original: str) -> str | None:
        return self._data.get(original)

    def add(self, original: str, adapted: str):
        self._data[original] = adapted

    def add_batch(self, pairs: dict[str, str]):
        self._data.update(pairs)

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d mappings to %s", len(self._data), self._path)

    def as_context_string(self) -> str:
        """Format existing mappings for inclusion in LLM prompts."""
        if not self._data:
            return "No prior mappings."
        lines = [f"  {k} → {v}" for k, v in self._data.items()]
        return "Known mappings:\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
