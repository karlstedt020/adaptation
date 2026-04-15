"""Naive machine translation (Google Translate) without cultural adaptation."""

import logging
import time

import pandas as pd
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


def translate_text(text: str, src: str = "en", dest: str = "ru") -> str:
    """Translate a single string via Google Translate."""
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as exc:
        logger.warning("Translation failed for '%s…': %s", text[:40], exc)
        return text


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


def translate_crows_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Naive-translate CrowS-Pairs; adds *_ru columns."""
    out = df.copy()
    logger.info("Naive-translating %d CrowS-Pairs …", len(df))
    out["sent_more_ru"] = translate_batch(df["sent_more"].tolist())
    out["sent_less_ru"] = translate_batch(df["sent_less"].tolist())
    return out


def translate_snips(df: pd.DataFrame) -> pd.DataFrame:
    """Naive-translate SNIPS utterances; adds text_ru column.

    Slot boundaries are NOT adapted — only raw text is translated.
    """
    out = df.copy()
    logger.info("Naive-translating %d SNIPS utterances …", len(df))
    out["text_ru"] = translate_batch(df["text"].tolist())
    return out
