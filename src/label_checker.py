"""Verify that labels are preserved after cultural adaptation."""

import json
import logging

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

CROWS_CHECK_SYSTEM = (
    "You verify that adapted sentence pairs preserve the original bias structure. "
    'Return ONLY JSON: {"valid": true/false, "issues": ["..."]}'
)

CROWS_CHECK_TEMPLATE = """\
Original (EN):
  sent_more: {sent_more}
  sent_less: {sent_less}
  bias_type: {bias_type}
  stereo_antistereo: {direction}

Adapted (RU):
  sent_more_ru: {sent_more_ru}
  sent_less_ru: {sent_less_ru}

Check:
1. Does the adapted pair still test the same bias_type ({bias_type})?
2. Is the stereotype direction preserved?
3. Is the pair minimal (differ only in bias-carrying entities)?
"""

SNIPS_CHECK_SYSTEM = (
    "You verify that adapted utterances preserve intent and slot structure. "
    'Return ONLY JSON: {"valid": true/false, "issues": ["..."]}'
)

SNIPS_CHECK_TEMPLATE = """\
Original: {text}
Intent: {intent}
Slots: {slots}

Adapted: {text_ru}
Adapted slots: {slots_ru}

Check:
1. Does the adapted utterance express the same intent ({intent})?
2. Are all slot types preserved with culturally appropriate values?
"""


def _parse_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)


def check_crows_pair(
    client: LLMClient, original: dict, adapted: dict
) -> dict:
    """Check that an adapted CrowS pair preserves its label."""
    prompt = CROWS_CHECK_TEMPLATE.format(
        sent_more=original["sent_more"],
        sent_less=original["sent_less"],
        bias_type=original["bias_type"],
        direction=original.get("stereo_antistereo", "stereo"),
        sent_more_ru=adapted.get("sent_more_ru", ""),
        sent_less_ru=adapted.get("sent_less_ru", ""),
    )
    raw = client.complete(prompt, CROWS_CHECK_SYSTEM, temperature=0.0)
    return _parse_json(raw)


def check_snips_utterance(
    client: LLMClient, original: dict, adapted: dict
) -> dict:
    """Check that an adapted SNIPS utterance preserves its labels."""
    prompt = SNIPS_CHECK_TEMPLATE.format(
        text=original["text"],
        intent=original["intent"],
        slots=json.dumps(original["slots"], ensure_ascii=False),
        text_ru=adapted.get("text_ru", ""),
        slots_ru=json.dumps(adapted.get("slots_ru", []), ensure_ascii=False),
    )
    raw = client.complete(prompt, SNIPS_CHECK_SYSTEM, temperature=0.0)
    return _parse_json(raw)
