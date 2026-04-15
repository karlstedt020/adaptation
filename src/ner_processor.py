"""NER-based detection of culturally-specific entities."""

import logging
from dataclasses import dataclass

import spacy

logger = logging.getLogger(__name__)

# Entity labels considered culturally relevant
CULTURAL_LABELS = {
    "PERSON", "GPE", "LOC", "ORG", "NORP",
    "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW",
}

ADAPTATION_TYPES = {
    "PERSON":      "name",
    "GPE":         "country_or_city",
    "LOC":         "location",
    "ORG":         "institution_or_brand",
    "NORP":        "nationality_or_group",
    "FAC":         "facility",
    "EVENT":       "holiday_or_event",
    "PRODUCT":     "brand_or_product",
    "WORK_OF_ART": "creative_work",
    "LAW":         "legal_reference",
}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    adaptation_type: str


class NERProcessor:
    """Extract and classify culturally-specific named entities."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self._nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> list[Entity]:
        """Return all NER entities from *text*."""
        doc = self._nlp(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                adaptation_type=ADAPTATION_TYPES.get(ent.label_, "other"),
            )
            for ent in doc.ents
        ]

    def extract_cultural_entities(self, text: str) -> list[Entity]:
        """Return only entities that likely need cultural adaptation."""
        return [
            e for e in self.extract_entities(text)
            if e.label in CULTURAL_LABELS
        ]

    def diff_entities(
        self, sent_a: str, sent_b: str
    ) -> tuple[list[Entity], list[Entity]]:
        """Find entities that differ between two sentences (minimal pair)."""
        ents_a = {e.text for e in self.extract_entities(sent_a)}
        ents_b = {e.text for e in self.extract_entities(sent_b)}
        only_a = [
            e for e in self.extract_entities(sent_a)
            if e.text not in ents_b
        ]
        only_b = [
            e for e in self.extract_entities(sent_b)
            if e.text not in ents_a
        ]
        return only_a, only_b

    def has_cultural_entities(self, text: str) -> bool:
        return len(self.extract_cultural_entities(text)) > 0
