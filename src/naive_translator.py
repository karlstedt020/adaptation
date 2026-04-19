"""Naive machine translation (Google Translate) without cultural adaptation."""

import logging

import pandas as pd
from deep_translator import GoogleTranslator

from .rate_limiter import parallel_apply
from .checkpoint import Checkpointer

logger = logging.getLogger(__name__)


def _translate_one(text: str, src: str = "en", dest: str = "ru") -> str:
    """Translate a single string; returns '' on failure."""
    if not text or not str(text).strip():
        return ""
    try:
        return GoogleTranslator(source=src, target=dest).translate(str(text))
    except Exception as exc:
        logger.warning("Google Translate failed for '%s…': %s", str(text)[:40], exc)
        return ""


def _slice(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return df.head(limit).copy() if limit is not None else df.copy()


def translate_crows_pairs(
    df: pd.DataFrame,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 5.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Naive-translate CrowS-Pairs with parallel Google Translate calls."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("Naive MT CrowS: %d to translate, %d cached", len(pending), len(done_ids))

    if len(pending):
        pending["sent_more_ru"] = parallel_apply(
            lambda t: _translate_one(t), pending["sent_more"].tolist(),
            max_workers=max_workers, rps=rps, desc="MT CrowS sent_more",
        )
        pending["sent_less_ru"] = parallel_apply(
            lambda t: _translate_one(t), pending["sent_less"].tolist(),
            max_workers=max_workers, rps=rps, desc="MT CrowS sent_less",
        )

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result


def translate_snips(
    df: pd.DataFrame,
    limit: int | None = None,
    max_workers: int = 4,
    rps: float = 5.0,
    checkpoint: Checkpointer | None = None,
) -> pd.DataFrame:
    """Naive-translate SNIPS utterances with parallel Google Translate calls."""
    out = _slice(df, limit)

    done_ids, existing = (checkpoint.load() if checkpoint else (set(), None))
    pending = out[~out.index.isin(done_ids)].copy()
    logger.info("Naive MT SNIPS: %d to translate, %d cached", len(pending), len(done_ids))

    if len(pending):
        pending["text_ru"] = parallel_apply(
            lambda t: _translate_one(t), pending["text"].tolist(),
            max_workers=max_workers, rps=rps, desc="MT SNIPS",
        )

    result = checkpoint.merge_and_save(existing, pending) if checkpoint else pending
    return result
