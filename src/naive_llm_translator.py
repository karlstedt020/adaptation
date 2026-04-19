"""Zero-shot LLM translation — translate without structured adaptation."""

import json
import logging

import pandas as pd

from .llm_client import LLMClient
from .json_utils import parse_json_response
from .rate_limiter import parallel_apply
from .checkpoint import Checkpointer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CROWS = (
    "You are a professional translator from English to Russian. "
    "Translate the given pair of sentences into Russian. "
    "Only lightly adapt obvious cultural references — do NOT apply any "
    "structured cultural-adaptation methodology. "
    "Return ONLY a single valid JSON object with keys "
    '"sent_more_ru" and "sent_less_ru". No prose before or after.'
)

SYSTEM_PROMPT_SNIPS = (
    "You are a professional translator from English to Russian. "
    "Translate the given utterance into Russian, preserving its intent "
    "and slot values. Only lightly adapt obvious cultural references. "
    'Return ONLY a single valid JSON object: {"text_ru": "..."}'
)


def _slice(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return df.head(limit).copy() if limit is not None else df.copy()


def _translate_crows_row(client: LLMClient, sent_more: str, sent_less: str) -> dict:
    prompt = (
        "Translate this pair of sentences to Russian, keeping them parallel:\n"
        f'sent_more: "{sent_more}"\n'
        f'sent_less: "{sent_less}"'
    )
    raw = client.complete(prompt, SYSTEM_PROMPT_CROWS, temperature=0.2)
    data = parse_json_response(raw)
    return {
        "sent_more_ru": str(data.get("sent_more_ru", "")).strip(),
        "sent_less_ru": str(data.get("sent_less_ru", "")).strip(),
    }


def _translate_snips_row(client: LLMClient, text: str) -> str:
    prompt = f'Translate this utterance to Russian:\n"{text}"'
    raw = client.complete(prompt, SYSTEM_PROMPT_SNIPS, temperature=0.2)
    data = parse_json_response(raw)
    return str(data.get("text_ru", "")).strip()


def translate_crows_pairs(
    client: LLMClient,
    df: pd.DataFrame,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 4.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Zero-shot LLM translation of CrowS-Pairs (parallel)."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("LLM CrowS: %d to translate, %d cached", len(pending), len(done_ids))

    if len(pending):
        rows = list(pending[["sent_more", "sent_less"]].itertuples(index=False))

        def _call(row):
            try:
                return _translate_crows_row(client, row.sent_more, row.sent_less)
            except Exception as exc:
                logger.warning("LLM translate failed: %s", exc)
                return {"sent_more_ru": "", "sent_less_ru": ""}

        results = parallel_apply(_call, rows, max_workers, rps, "LLM CrowS")
        pending["sent_more_ru"] = [r["sent_more_ru"] for r in results]
        pending["sent_less_ru"] = [r["sent_less_ru"] for r in results]

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result


def translate_snips(
    client: LLMClient,
    df: pd.DataFrame,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 4.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Zero-shot LLM translation of SNIPS utterances (parallel)."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("LLM SNIPS: %d to translate, %d cached", len(pending), len(done_ids))

    if len(pending):
        texts = pending["text"].tolist()

        def _call(text):
            try:
                return _translate_snips_row(client, text)
            except Exception as exc:
                logger.warning("LLM translate failed: %s", exc)
                return ""

        pending["text_ru"] = parallel_apply(_call, texts, max_workers, rps, "LLM SNIPS")

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result
