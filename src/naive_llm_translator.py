"""Zero-shot LLM translation — translate without structured adaptation."""

import json
import logging

import pandas as pd
from tqdm import tqdm

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CROWS = (
    "You are a professional translator from English to Russian. "
    "Translate the given pair of sentences into Russian. "
    "Try to adapt cultural references where obvious, but do not apply "
    "any structured methodology. Return ONLY valid JSON with keys "
    '"sent_more_ru" and "sent_less_ru".'
)

SYSTEM_PROMPT_SNIPS = (
    "You are a professional translator from English to Russian. "
    "Translate the given utterance into Russian. "
    "Try to adapt obvious cultural references. "
    'Return ONLY valid JSON with key "text_ru".'
)


def _parse_json(response: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)


def translate_crows_pair(
    client: LLMClient, sent_more: str, sent_less: str
) -> dict:
    prompt = (
        f"Translate this pair of sentences to Russian:\n"
        f"sent_more: {sent_more}\n"
        f"sent_less: {sent_less}"
    )
    raw = client.complete(prompt, SYSTEM_PROMPT_CROWS, temperature=0.3)
    return _parse_json(raw)


def translate_crows_pairs(
    client: LLMClient, df: pd.DataFrame
) -> pd.DataFrame:
    """Zero-shot LLM translation of CrowS-Pairs."""
    out = df.copy()
    more_ru, less_ru = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM-translate CrowS"):
        try:
            res = translate_crows_pair(
                client, row["sent_more"], row["sent_less"]
            )
            more_ru.append(res.get("sent_more_ru", ""))
            less_ru.append(res.get("sent_less_ru", ""))
        except Exception as exc:
            logger.warning("LLM translate failed row %s: %s", row.name, exc)
            more_ru.append("")
            less_ru.append("")
    out["sent_more_ru"] = more_ru
    out["sent_less_ru"] = less_ru
    return out


def translate_snips(
    client: LLMClient, df: pd.DataFrame
) -> pd.DataFrame:
    """Zero-shot LLM translation of SNIPS utterances."""
    out = df.copy()
    texts_ru = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM-translate SNIPS"):
        try:
            prompt = f"Translate this utterance to Russian:\n{row['text']}"
            raw = client.complete(prompt, SYSTEM_PROMPT_SNIPS, temperature=0.3)
            res = _parse_json(raw)
            texts_ru.append(res.get("text_ru", ""))
        except Exception as exc:
            logger.warning("LLM translate failed row %s: %s", row.name, exc)
            texts_ru.append("")
    out["text_ru"] = texts_ru
    return out
