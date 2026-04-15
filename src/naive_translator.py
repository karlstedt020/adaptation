"""Naive machine translation (Google Translate) without cultural adaptation."""

import logging
import time

import pandas as pd
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


def translate_text(text: str, src: str = "en", dest: str = "ru") -> str:
    """Translate a single string via Google Translate."""
    if not text or not str(text).strip():
        return ""
    try:
        return GoogleTranslator(source=src, target=dest).translate(str(text))
    except Exception as exc:
        logger.warning("Translation failed for '%s…': %s", str(text)[:40], exc)
        return ""


def translate_batch(
    texts: list[str], src: str = "en", dest: str = "ru", delay: float = 0.1
) -> list[str]:
    """Translate a list of texts with a small delay to avoid rate limits."""
    results = []
    for i, t in enumerate(texts):
        results.append(translate_text(t, src, dest))
        if delay and i % 50 == 49:
            time.sleep(delay)
    return results


def _slice(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Return df or df.head(limit) if limit is set."""
    return df.head(limit).copy() if limit is not None else df.copy()


def translate_crows_pairs(
    df: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Naive-translate CrowS-Pairs; adds *_ru columns.

    Args:
        limit: if set, only the first *limit* rows are processed (for smoke tests).
    """
    out = _slice(df, limit)
    logger.info("Naive-translating %d CrowS-Pairs …", len(out))
    out["sent_more_ru"] = translate_batch(out["sent_more"].tolist())
    out["sent_less_ru"] = translate_batch(out["sent_less"].tolist())
    return out


def translate_snips(
    df: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Naive-translate SNIPS utterances; adds text_ru column.

    Slot boundaries are NOT adapted — only raw text is translated.
    """
    out = _slice(df, limit)
    logger.info("Naive-translating %d SNIPS utterances …", len(out))
    out["text_ru"] = translate_batch(out["text"].tolist())
    return out
