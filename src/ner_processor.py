"""LLM-based NER for culturally-specific entities.

Extracts entities following the Nevéol et al. (ACL 2022) typology for
cultural adaptation: names, locations, food, sports, institutions,
holidays/events, brands, nationalities, works of art, etc.

Results are cached at the LLMClient level (temperature=0 → deterministic).
Entity dicts (via ``entity_to_dict`` / ``entities_from_dicts``) are stored
in DataFrames so the pipeline never repeats the same API call.
"""

import logging
from dataclasses import dataclass, asdict

from .llm_client import LLMClient
from .json_utils import parse_json_response

logger = logging.getLogger(__name__)

ADAPTATION_TYPES = {
    "PERSON":      "name",
    "LOCATION":    "country_or_city",
    "FOOD":        "food_or_drink",
    "SPORT":       "sport",
    "ORG":         "institution_or_brand",
    "NORP":        "nationality_or_group",
    "HOLIDAY":     "holiday_or_event",
    "WORK_OF_ART": "creative_work",
    "LAW":         "legal_reference",
    "CURRENCY":    "currency",
    "MEASUREMENT": "measurement",
    "OTHER":       "other",
}

CULTURAL_TYPES = set(ADAPTATION_TYPES.keys()) - {"OTHER"}

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
4. If an entity belongs to several categories, pick the most specific one.
5. Do NOT extract pronouns, common nouns, or generic adjectives.
6. Return ONLY valid JSON, no commentary."""

NER_USER_TEMPLATE = """\
Extract culturally-specific named entities from the following sentence.

Sentence: {sentence}

Return JSON in this exact format:
{{"entities": [{{"text": "...", "label": "PERSON|LOCATION|FOOD|SPORT|ORG|NORP|HOLIDAY|WORK_OF_ART|LAW|CURRENCY|MEASUREMENT", "start": <int>, "end": <int>}}, ...]}}

If no culturally-specific entities are present, return: {{"entities": []}}"""


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    adaptation_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def entities_from_dicts(dicts: list[dict]) -> list["Entity"]:
    """Reconstruct Entity objects from serialised dicts (stored in df)."""
    return [Entity(**d) for d in (dicts or [])]


def _fix_offsets(entity: dict, sentence: str) -> dict | None:
    text = entity.get("text", "")
    if not text:
        return None
    start, end = entity.get("start", -1), entity.get("end", -1)
    if (isinstance(start, int) and isinstance(end, int)
            and 0 <= start < end <= len(sentence)
            and sentence[start:end] == text):
        return entity
    idx = sentence.find(text)
    if idx < 0:
        return None
    return {**entity, "start": idx, "end": idx + len(text)}


class NERProcessor:
    """LLM-based extractor of culturally-specific named entities.

    Caching is handled by ``LLMClient`` (temp=0, hash-keyed).
    NERProcessor itself is stateless — all results should be stored in df
    columns (``ner_entities``) and passed to downstream components.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def extract_entities(self, text: str) -> list[Entity]:
        """Return all cultural entities for *text* (cached via LLMClient)."""
        prompt = NER_USER_TEMPLATE.format(sentence=text)
        try:
            raw = self._client.complete(prompt, NER_SYSTEM_PROMPT, temperature=0.0)
            parsed = parse_json_response(raw)
            return _to_entities(parsed.get("entities", []), text)
        except Exception as exc:
            logger.warning("LLM NER failed for '%s…': %s", text[:50], exc)
            return []

    def extract_cultural_entities(self, text: str) -> list[Entity]:
        """Every entity returned by extract_entities is already cultural."""
        return self.extract_entities(text)

    def diff_entities(
        self, sent_a: str, sent_b: str,
        ents_a: list[Entity] | None = None,
        ents_b: list[Entity] | None = None,
    ) -> tuple[list[Entity], list[Entity]]:
        """Entities unique to each sentence (pre-computed lists accepted)."""
        if ents_a is None:
            ents_a = self.extract_entities(sent_a)
        if ents_b is None:
            ents_b = self.extract_entities(sent_b)
        texts_b = {e.text for e in ents_b}
        texts_a = {e.text for e in ents_a}
        return (
            [e for e in ents_a if e.text not in texts_b],
            [e for e in ents_b if e.text not in texts_a],
        )


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
