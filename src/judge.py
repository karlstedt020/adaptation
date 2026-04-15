"""LLM-as-a-Judge for selecting the best adaptation variant."""

import json
import logging

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict quality judge for cultural adaptation of NLP datasets "
    "from English to Russian. Evaluate each variant on:\n"
    "1. Grammatical correctness (natural Russian)\n"
    "2. Cultural adequacy (no leftover American entities, substitutions "
    "make sense in Russian context)\n"
    "3. Preservation of the original structure and label\n"
    "4. Minimal-pair property (for CrowS-Pairs: sentences differ only in "
    "bias-carrying entities)\n"
    "Return ONLY the index (0-based) of the best variant as a JSON: "
    '{"best_index": N, "reason": "..."}'
)

CROWS_JUDGE_TEMPLATE = """\
Original pair (English):
  sent_more: {sent_more}
  sent_less: {sent_less}
  bias_type: {bias_type}

Adaptation variants:
{variants_text}

Pick the best variant. Consider: grammatical quality, cultural naturalness \
for Russia, and preservation of the minimal-pair structure (sentences must \
differ ONLY in bias-carrying entities).
"""

SNIPS_JUDGE_TEMPLATE = """\
Original utterance: {text}
Intent: {intent}

Adaptation variants:
{variants_text}

Pick the best variant. Consider: grammatical quality, cultural naturalness \
for Russia, and preservation of intent and slot structure.
"""


class JudgeEvaluator:
    """Use a separate LLM call to pick the best adaptation."""

    def __init__(self, client: LLMClient):
        self._client = client

    def select_best_crows(
        self,
        sent_more: str,
        sent_less: str,
        bias_type: str,
        variants: list[dict],
    ) -> dict:
        if len(variants) == 1:
            return variants[0]
        variants_text = self._format_variants(variants)
        prompt = CROWS_JUDGE_TEMPLATE.format(
            sent_more=sent_more,
            sent_less=sent_less,
            bias_type=bias_type,
            variants_text=variants_text,
        )
        return self._pick(variants, prompt)

    def select_best_snips(
        self,
        text: str,
        intent: str,
        variants: list[dict],
    ) -> dict:
        if len(variants) == 1:
            return variants[0]
        variants_text = self._format_variants(variants)
        prompt = SNIPS_JUDGE_TEMPLATE.format(
            text=text,
            intent=intent,
            variants_text=variants_text,
        )
        return self._pick(variants, prompt)

    def _pick(self, variants: list[dict], prompt: str) -> dict:
        try:
            raw = self._client.complete(
                prompt, JUDGE_SYSTEM, temperature=0.0
            )
            result = self._parse_judge(raw)
            idx = result.get("best_index", 0)
            if 0 <= idx < len(variants):
                return variants[idx]
        except Exception as exc:
            logger.warning("Judge failed, defaulting to first: %s", exc)
        return variants[0]

    @staticmethod
    def _format_variants(variants: list[dict]) -> str:
        lines = []
        for i, v in enumerate(variants):
            lines.append(f"[{i}] {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)

    @staticmethod
    def _parse_judge(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
