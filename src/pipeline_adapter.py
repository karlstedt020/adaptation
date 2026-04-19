"""Full cultural adaptation pipeline with anthropological prompting (parallel)."""

import json
import logging

import pandas as pd

from .llm_client import LLMClient
from .ner_processor import NERProcessor, Entity, entities_from_dicts
from .mapping_store import MappingStore
from .judge import JudgeEvaluator
from .json_utils import parse_json_response
from .rate_limiter import parallel_apply
from .checkpoint import Checkpointer

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
  (Thanksgiving→Новый год, Super Bowl→финал КХЛ, SAT→ЕГЭ).
- Replace culturally-specific foods (mac and cheese→пельмени, hot dog→шаурма).
- Replace institutions (401k→пенсионные накопления, Social Security→пенсионный фонд).
- Stereotypes about African Americans → stereotypes about peoples of the RF/CIS.

Always return ONE valid JSON object — no prose before or after.

{mappings_context}
"""

CROWS_USER_TEMPLATE = """\
Adapt the following sentence pair for the Russian cultural context.

CRITICAL RULES:
1. The two Russian sentences MUST differ ONLY in the named entities related \
to the cultural shift being tested (bias_type: {bias_type}).
2. Keep the minimal-pair structure: everything except the bias-carrying \
entities must be IDENTICAL (same word order, grammatical case, number).
3. Produce natural, idiomatic Russian.

Original pair:
  sent_more: {sent_more}
  sent_less: {sent_less}
  bias_type: {bias_type}
  Differing named entities (NER): {diff_entities}

Return ONLY this JSON:
{{
  "sent_more_ru": "...",
  "sent_less_ru": "...",
  "mappings": {{"original_entity": "russian_entity"}}
}}"""

SNIPS_USER_TEMPLATE = """\
Adapt the following utterance for the Russian cultural context.
Replace culturally-specific named entities with Russian equivalents.
Keep the intent ({intent}) and all slot types. Match slot spans to the adapted text.

Original: {text}
Named entities detected: {entities}
Original slots: {slots}

Return ONLY this JSON:
{{
  "text_ru": "...",
  "slots_ru": [{{"text": "value in text_ru", "slot": "slot_type", "start": <int>, "end": <int>}}],
  "mappings": {{"original": "russian"}}
}}"""


def _build_system(mapping_store: MappingStore) -> str:
    return ANTHROPOLOGICAL_SYSTEM.format(
        mappings_context=mapping_store.as_context_string()
    )


def _parse_ner(val) -> list[dict]:
    """Accept list[dict] or JSON-string forms stored in the dataframe."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
            return loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _parse_variants(variants: list[str]) -> list[dict]:
    result = []
    for v in variants:
        try:
            result.append(parse_json_response(v))
        except (json.JSONDecodeError, ValueError):
            continue
    return result


def _valid_crows(v: dict) -> bool:
    return bool(v.get("sent_more_ru")) and bool(v.get("sent_less_ru"))


def _valid_snips(v: dict) -> bool:
    return bool(v.get("text_ru"))


def _diff_str(ents_a: list[Entity], ents_b: list[Entity]) -> str:
    parts = (
        [f"{e.text} ({e.label})" for e in ents_a]
        + [f"{e.text} ({e.label})" for e in ents_b]
    )
    return ", ".join(parts) or "none detected"


def _slice(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return df.head(limit).copy() if limit is not None else df.copy()


# ── CrowS-Pairs ──────────────────────────────────────────────

def _adapt_crows_row(
    client: LLMClient,
    judge: JudgeEvaluator,
    mapping_store: MappingStore,
    row: dict,
    n_variants: int,
) -> dict:
    """Adapt one CrowS pair; use pre-computed NER entities from row."""
    ents_more = entities_from_dicts(_parse_ner(row.get("ner_more")))
    ents_less = entities_from_dicts(_parse_ner(row.get("ner_less")))
    diff_a = [e for e in ents_more if e.text not in {x.text for x in ents_less}]
    diff_b = [e for e in ents_less if e.text not in {x.text for x in ents_more}]

    prompt = CROWS_USER_TEMPLATE.format(
        sent_more=row["sent_more"], sent_less=row["sent_less"],
        bias_type=row["bias_type"],
        diff_entities=_diff_str(diff_a, diff_b),
    )
    system = _build_system(mapping_store)
    variants = client.complete_n(prompt, system, n=n_variants, temperature=0.9)
    parsed = [v for v in _parse_variants(variants) if _valid_crows(v)]
    if not parsed:
        return {"sent_more_ru": "", "sent_less_ru": "", "mappings": {}}

    best = judge.select_best_crows(
        row["sent_more"], row["sent_less"], row["bias_type"], parsed,
    )
    if isinstance(best.get("mappings"), dict):
        mapping_store.add_batch(best["mappings"])
    return best


def adapt_crows_pairs(
    client: LLMClient,
    ner: NERProcessor,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    df: pd.DataFrame,
    n_variants: int = 3,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 4.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Adapt CrowS-Pairs through the full pipeline (parallel, with checkpoint)."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("Pipeline CrowS: %d to adapt, %d cached", len(pending), len(done_ids))

    if len(pending):
        rows = [r for _, r in pending.iterrows()]

        def _call(row):
            try:
                return _adapt_crows_row(
                    client, judge, mapping_store, row.to_dict(), n_variants,
                )
            except Exception as exc:
                logger.warning("Pipeline CrowS failed idx %s: %s", row.name, exc)
                return {"sent_more_ru": "", "sent_less_ru": ""}

        results = parallel_apply(_call, rows, max_workers, rps, "Pipeline CrowS")
        pending["sent_more_ru"] = [r.get("sent_more_ru", "") for r in results]
        pending["sent_less_ru"] = [r.get("sent_less_ru", "") for r in results]
        mapping_store.save()

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result


# ── SNIPS ────────────────────────────────────────────────────

def _adapt_snips_row(
    client: LLMClient,
    judge: JudgeEvaluator,
    mapping_store: MappingStore,
    row: dict,
    n_variants: int,
) -> dict:
    """Adapt one SNIPS utterance using pre-computed NER entities."""
    ents = entities_from_dicts(_parse_ner(row.get("ner_entities")))
    ents_str = ", ".join(f"{e.text} ({e.label})" for e in ents) or "none"

    prompt = SNIPS_USER_TEMPLATE.format(
        text=row["text"],
        intent=row["intent"],
        entities=ents_str,
        slots=json.dumps(row.get("slots", []), ensure_ascii=False),
    )
    system = _build_system(mapping_store)
    variants = client.complete_n(prompt, system, n=n_variants, temperature=0.9)
    parsed = [v for v in _parse_variants(variants) if _valid_snips(v)]
    if not parsed:
        return {"text_ru": "", "slots_ru": [], "mappings": {}}

    best = judge.select_best_snips(row["text"], row["intent"], parsed)
    if isinstance(best.get("mappings"), dict):
        mapping_store.add_batch(best["mappings"])
    return best


def adapt_snips(
    client: LLMClient,
    mapping_store: MappingStore,
    judge: JudgeEvaluator,
    df: pd.DataFrame,
    n_variants: int = 3,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 4.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Adapt SNIPS through the full pipeline (parallel, with checkpoint)."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("Pipeline SNIPS: %d to adapt, %d cached", len(pending), len(done_ids))

    if len(pending):
        rows = [r for _, r in pending.iterrows()]

        def _call(row):
            try:
                return _adapt_snips_row(
                    client, judge, mapping_store, row.to_dict(), n_variants,
                )
            except Exception as exc:
                logger.warning("Pipeline SNIPS failed idx %s: %s", row.name, exc)
                return {"text_ru": "", "slots_ru": []}

        results = parallel_apply(_call, rows, max_workers, rps, "Pipeline SNIPS")
        pending["text_ru"] = [r.get("text_ru", "") for r in results]
        pending["slots_ru"] = [r.get("slots_ru", []) for r in results]
        mapping_store.save()

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result
