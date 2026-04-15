"""Full cultural adaptation pipeline with anthropological prompting."""

import json
import logging

import pandas as pd
from tqdm import tqdm

from .llm_client import LLMClient
from .ner_processor import NERProcessor
from .mapping_store import MappingStore
from .judge import JudgeEvaluator

logger = logging.getLogger(__name__)

ANTHROPOLOGICAL_SYSTEM = """\
You are an expert in Russian culture and society performing cultural \
adaptation of English-language NLP datasets for the Russian context.

Background knowledge you MUST apply:
- Replace American names with common Russian names (Иван, Мария, Алексей…).
- Replace US cities/states with Russian ones (Москва, Санкт-Петербург, Казань…).
- Replace American brands/chains with Russian equivalents \
  (Walmart→Лента, Starbucks→Кофемания, Olive Garden→Теремок).
- Replace American holidays/events with Russian ones \
  (Thanksgiving→Новый год, Super Bowl→Чемпионат мира по хоккею, SAT→ЕГЭ).
- Replace culturally-specific foods (mac and cheese→пельмени, \
  hot dog→шаурма).
- Replace institutions (401k→пенсионные накопления, \
  Social Security→пенсионный фонд).
- Stereotypes about African Americans should be replaced with \
  stereotypes relevant to the Russian context (ethnic stereotypes \
  about peoples of the RF and CIS).

{mappings_context}
"""

CROWS_USER_TEMPLATE = """\
Adapt the following sentence pair for the Russian cultural context.

CRITICAL RULES:
1. The two sentences MUST differ ONLY in the named entities related to \
the cultural shift being tested (bias_type: {bias_type}).
2. Keep the minimal-pair structure: everything except the bias-carrying \
entities must be IDENTICAL between sent_more_ru and sent_less_ru.
3. Translate into natural Russian.

Original pair:
  sent_more: {sent_more}
  sent_less: {sent_less}
  bias_type: {bias_type}
  Differing entities: {diff_entities}

Return ONLY valid JSON:
{{"sent_more_ru": "...", "sent_less_ru": "...", "mappings": {{"original_entity": "russian_entity", ...}}}}
"""

SNIPS_USER_TEMPLATE = """\
Adapt the following utterance for the Russian cultural context.
Replace culturally-specific named entities with Russian equivalents.
Keep the intent ({intent}) and slot structure intact.

Original: {text}
Slots: {slots}

Return ONLY valid JSON:
{{"text_ru": "...", "slots_ru": [...], "mappings": {{"original": "russian", ...}}}}
"""


def _parse_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)


def _build_system_prompt(mapping_store: MappingStore) -> str:
    return ANTHROPOLOGICAL_SYSTEM.format(
        mappings_context=mapping_store.as_context_string()
    )


def adapt_crows_pair(
    client: LLMClient,
    ner: NERProcessor,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    row: pd.Series,
    n_variants: int = 3,
) -> dict:
    """Generate n_variants adaptations of a CrowS pair, pick best."""
    diff_a, diff_b = ner.diff_entities(row["sent_more"], row["sent_less"])
    diff_str = ", ".join(
        [e.text for e in diff_a] + [e.text for e in diff_b]
    ) or "none detected"

    prompt = CROWS_USER_TEMPLATE.format(
        sent_more=row["sent_more"],
        sent_less=row["sent_less"],
        bias_type=row["bias_type"],
        diff_entities=diff_str,
    )
    system = _build_system_prompt(mapping_store)
    variants = client.complete_n(
        prompt, system, n=n_variants, temperature=0.9
    )
    parsed = _try_parse_variants(variants)
    if not parsed:
        return {"sent_more_ru": "", "sent_less_ru": "", "mappings": {}}

    best = judge.select_best_crows(
        row["sent_more"], row["sent_less"], row["bias_type"], parsed
    )
    if "mappings" in best:
        mapping_store.add_batch(best["mappings"])
    return best


def _try_parse_variants(variants: list[str]) -> list[dict]:
    parsed = []
    for v in variants:
        try:
            parsed.append(_parse_json(v))
        except (json.JSONDecodeError, ValueError):
            continue
    return parsed


def adapt_crows_pairs(
    client: LLMClient,
    ner: NERProcessor,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    df: pd.DataFrame,
    n_variants: int = 3,
) -> pd.DataFrame:
    """Adapt entire CrowS-Pairs dataset through the full pipeline."""
    out = df.copy()
    more_ru, less_ru = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pipeline CrowS"):
        try:
            res = adapt_crows_pair(
                client, ner, mapping_store, judge, row, n_variants
            )
            more_ru.append(res.get("sent_more_ru", ""))
            less_ru.append(res.get("sent_less_ru", ""))
        except Exception as exc:
            logger.warning("Pipeline failed row %s: %s", row.name, exc)
            more_ru.append("")
            less_ru.append("")
    out["sent_more_ru"] = more_ru
    out["sent_less_ru"] = less_ru
    mapping_store.save()
    return out


def adapt_snips_row(
    client: LLMClient,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    row: pd.Series,
    n_variants: int = 3,
) -> dict:
    """Adapt a single SNIPS utterance."""
    prompt = SNIPS_USER_TEMPLATE.format(
        text=row["text"],
        intent=row["intent"],
        slots=json.dumps(row["slots"], ensure_ascii=False),
    )
    system = _build_system_prompt(mapping_store)
    variants = client.complete_n(
        prompt, system, n=n_variants, temperature=0.9
    )
    parsed = _try_parse_variants(variants)
    if not parsed:
        return {"text_ru": "", "slots_ru": [], "mappings": {}}

    best = judge.select_best_snips(row["text"], row["intent"], parsed)
    if "mappings" in best:
        mapping_store.add_batch(best["mappings"])
    return best


def adapt_snips(
    client: LLMClient,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    df: pd.DataFrame,
    n_variants: int = 3,
) -> pd.DataFrame:
    """Adapt entire SNIPS dataset through the full pipeline."""
    out = df.copy()
    texts_ru, slots_ru = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pipeline SNIPS"):
        try:
            res = adapt_snips_row(
                client, mapping_store, judge, row, n_variants
            )
            texts_ru.append(res.get("text_ru", ""))
            slots_ru.append(res.get("slots_ru", []))
        except Exception as exc:
            logger.warning("Pipeline failed row %s: %s", row.name, exc)
            texts_ru.append("")
            slots_ru.append([])
    out["text_ru"] = texts_ru
    out["slots_ru"] = slots_ru
    mapping_store.save()
    return out
