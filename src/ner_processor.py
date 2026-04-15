"""LLM-based NER for culturally-specific entities.

Extracts entities following the Nevéol et al. (ACL 2022) typology for
cultural adaptation: names, locations, food, sports, institutions,
holidays/events, brands, nationalities, works of art, etc.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from .llm_client import LLMClient
from .json_utils import parse_json_response

logger = logging.getLogger(__name__)

# Adaptation categories from Nevéol et al. (ACL 2022)
ADAPTATION_TYPES = {
    "PERSON":      "name",                  # first/last names, fictional characters
    "LOCATION":    "country_or_city",       # countries, cities, regions
    "FOOD":        "food_or_drink",         # national dishes, drinks
    "SPORT":       "sport",                 # sports & teams
    "ORG":         "institution_or_brand",  # companies, schools, agencies
    "NORP":        "nationality_or_group",  # nationalities, ethnic & religious groups
    "HOLIDAY":     "holiday_or_event",      # holidays, national events
    "WORK_OF_ART": "creative_work",         # movies, books, songs
    "LAW":         "legal_reference",       # laws, legal concepts
    "CURRENCY":    "currency",              # currencies
    "MEASUREMENT": "measurement",           # culturally-bound measurements (miles, lbs)
    "OTHER":       "other",
}

# Types that are considered culturally-specific
CULTURAL_TYPES = set(ADAPTATION_TYPES.keys()) - {"OTHER"}


@dataclass
class Entity:
    """Culturally-specific named entity."""
    text: str
    label: str               # one of ADAPTATION_TYPES keys
    start: int               # char offset in the input sentence
    end: int
    adaptation_type: str     # human-readable category


# ── Prompt templates ────────────────────────────────────────

NER_SYSTEM_PROMPT = """\
You are a precise named-entity recognition system for cultural adaptation \
of English-language NLP datasets. You \
extract entities that are candidates for cultural replacement when adapting \
the sentence to a different (Russian) cultural context.

Entity categories (use these EXACT labels):
  PERSON       — given/family names, fictional characters (e.g. John, Kanye West)
  LOCATION     — countries, cities, regions, states (e.g. Texas, Los Angeles)
  FOOD         — culturally-specific foods & drinks (e.g. Thanksgiving turkey, mac and cheese, root beer)
  SPORT        — sports, leagues, teams (e.g. NFL, Super Bowl, baseball)
  ORG          — companies, schools, government agencies, brands (e.g. Walmart, Harvard, FBI)
  NORP         — nationalities, ethnic, religious or political groups (e.g. African Americans, Jews, Republicans)
  HOLIDAY      — holidays, cultural or national events (e.g. Thanksgiving, Super Bowl Sunday, Fourth of July)
  WORK_OF_ART  — movies, books, songs, TV shows (e.g. The Bible, Friends, Star Wars)
  LAW          — laws, legal concepts, acts (e.g. Second Amendment, Miranda rights)
  CURRENCY     — currencies (e.g. dollars, cents)
  MEASUREMENT  — culturally-bound measurements (e.g. miles, pounds, Fahrenheit)

RULES:
1. Extract ONLY entities that might need replacement when adapting to another culture.
2. Return exact surface form as it appears in the text (case-sensitive).
3. Provide character offsets (0-indexed, end-exclusive) matching the exact substring.
4. If an entity could belong to several categories, pick the most specific one \
(e.g. "Thanksgiving dinner" → HOLIDAY, not FOOD).
5. Do NOT extract pronouns, common nouns, or generic adjectives.
6. Return ONLY valid JSON, no commentary."""

NER_USER_TEMPLATE = """\
Extract culturally-specific named entities from the following sentence.

Sentence: {sentence}

Return JSON in this exact format:
{{"entities": [{{"text": "...", "label": "PERSON|LOCATION|FOOD|SPORT|ORG|NORP|HOLIDAY|WORK_OF_ART|LAW|CURRENCY|MEASUREMENT", "start": <int>, "end": <int>}}, ...]}}

If no culturally-specific entities are present, return: {{"entities": []}}"""


# ── JSON parsing helper ─────────────────────────────────────

def _fix_offsets(entity: dict, sentence: str) -> dict | None:
    """Validate/repair character offsets by searching for the surface form."""
    text = entity.get("text", "")
    if not text:
        return None
    start = entity.get("start", -1)
    end = entity.get("end", -1)
    if (
        isinstance(start, int) and isinstance(end, int)
        and 0 <= start < end <= len(sentence)
        and sentence[start:end] == text
    ):
        return entity
    # Fallback: search
    idx = sentence.find(text)
    if idx < 0:
        return None
    return {**entity, "start": idx, "end": idx + len(text)}


# ── Main processor ──────────────────────────────────────────

class NERProcessor:
    """LLM-based extractor of culturally-specific named entities."""

    def __init__(self, client: LLMClient):
        self._client = client
        # Per-instance memoisation of NER results (cheap cache)
        self._cache: dict[str, list[Entity]] = {}

    def extract_entities(self, text: str) -> list[Entity]:
        """Return all entities (from the cultural typology) in *text*."""
        if text in self._cache:
            return self._cache[text]
        entities = self._call_llm(text)
        self._cache[text] = entities
        return entities

    def extract_cultural_entities(self, text: str) -> list[Entity]:
        """Alias — every entity returned is already cultural."""
        return [
            e for e in self.extract_entities(text)
            if e.label in CULTURAL_TYPES
        ]

    def diff_entities(
        self, sent_a: str, sent_b: str
    ) -> tuple[list[Entity], list[Entity]]:
        """Entities that differ between two sentences of a minimal pair."""
        ents_a = self.extract_entities(sent_a)
        ents_b = self.extract_entities(sent_b)
        texts_b = {e.text for e in ents_b}
        texts_a = {e.text for e in ents_a}
        only_a = [e for e in ents_a if e.text not in texts_b]
        only_b = [e for e in ents_b if e.text not in texts_a]
        return only_a, only_b

    def has_cultural_entities(self, text: str) -> bool:
        return len(self.extract_cultural_entities(text)) > 0

    # ── internal ─────────────────────────────────────────────

    def _call_llm(self, sentence: str) -> list[Entity]:
        prompt = NER_USER_TEMPLATE.format(sentence=sentence)
        try:
            raw = self._client.complete(
                prompt, NER_SYSTEM_PROMPT, temperature=0.0
            )
            parsed = parse_json_response(raw)
            return self._to_entities(parsed.get("entities", []), sentence)
        except Exception as exc:
            logger.warning("LLM NER failed for '%s…': %s", sentence[:50], exc)
            return []

    @staticmethod
    def _to_entities(raw: list[dict], sentence: str) -> list[Entity]:
        entities = []
        for e in raw:
            fixed = _fix_offsets(e, sentence)
            if not fixed:
                continue
            label = fixed.get("label", "OTHER")
            if label not in ADAPTATION_TYPES:
                label = "OTHER"
            entities.append(Entity(
                text=fixed["text"],
                label=label,
                start=fixed["start"],
                end=fixed["end"],
                adaptation_type=ADAPTATION_TYPES[label],
            ))
        return entities
