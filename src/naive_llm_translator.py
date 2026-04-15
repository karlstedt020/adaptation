"""Zero-shot LLM translation — translate without structured adaptation."""

import json
import logging

import pandas as pd
from tqdm import tqdm

from .llm_client import LLMClient
from .json_utils import parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CROWS = (
    "You are a professional translator from English to Russian. "
    "Translate the given pair of sentences into Russian. "
    "Try to adapt cultural references where obvious, but do NOT apply "
    "any structured cultural-adaptation methodology (no brand/holiday "
    "replacement unless the English reference is completely opaque to "
    "a Russian reader). "
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


def translate_crows_pair(
    client: LLMClient, sent_more: str, sent_less: str
) -> dict:
    """Translate a single CrowS pair."""
    prompt = (
        "Translate this pair of sentences to Russian. "
        "Keep them parallel in structure:\n"
        f'sent_more: "{sent_more}"\n'
        f'sent_less: "{sent_less}"'
    )
    raw = client.complete(prompt, SYSTEM_PROMPT_CROWS, temperature=0.2)
    data = parse_json_response(raw)
    return {
        "sent_more_ru": str(data.get("sent_more_ru", "")).strip(),
        "sent_less_ru": str(data.get("sent_less_ru", "")).strip(),
    }


def translate_crows_pairs(
    client: LLMClient, df: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Zero-shot LLM translation of CrowS-Pairs."""
    out = _slice(df, limit)
    more_ru, less_ru = [], []
    for _, row in tqdm(out.iterrows(), total=len(out), desc="LLM-translate CrowS"):
        try:
            res = translate_crows_pair(
                client, row["sent_more"], row["sent_less"]
            )
            more_ru.append(res["sent_more_ru"])
            less_ru.append(res["sent_less_ru"])
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("LLM translate JSON error row %s: %s", row.name, exc)
            more_ru.append("")
            less_ru.append("")
        except Exception as exc:
            logger.warning("LLM translate failed row %s: %s", row.name, exc)
            more_ru.append("")
            less_ru.append("")
    out["sent_more_ru"] = more_ru
    out["sent_less_ru"] = less_ru
    return out


def translate_snips_utterance(client: LLMClient, text: str) -> str:
    prompt = f'Translate this utterance to Russian:\n"{text}"'
    raw = client.complete(prompt, SYSTEM_PROMPT_SNIPS, temperature=0.2)
    data = parse_json_response(raw)
    return str(data.get("text_ru", "")).strip()


def translate_snips(
    client: LLMClient, df: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Zero-shot LLM translation of SNIPS utterances."""
    out = _slice(df, limit)
    texts_ru = []
    for _, row in tqdm(out.iterrows(), total=len(out), desc="LLM-translate SNIPS"):
        try:
            texts_ru.append(translate_snips_utterance(client, row["text"]))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("LLM translate JSON error row %s: %s", row.name, exc)
            texts_ru.append("")
        except Exception as exc:
            logger.warning("LLM translate failed row %s: %s", row.name, exc)
            texts_ru.append("")
    out["text_ru"] = texts_ru
    return out
