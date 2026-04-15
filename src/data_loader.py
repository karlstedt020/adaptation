"""Loaders for CrowS-Pairs and SNIPS datasets."""

import json
import csv
import io
import logging
from pathlib import Path

import pandas as pd
import requests

from .config import PathConfig

logger = logging.getLogger(__name__)

CROWS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
)
SNIPS_BASE = (
    "https://raw.githubusercontent.com/sonos/nlu-benchmark/master/"
    "2017-06-custom-intent-engines/{intent}/train_{intent}_full.json"
)
SNIPS_INTENTS = [
    "AddToPlaylist", "BookRestaurant", "GetWeather",
    "PlayMusic", "RateBook", "SearchCreativeWork",
    "SearchScreeningEvent",
]


# ── CrowS-Pairs ─────────────────────────────────────────────

def download_crows_pairs(dest: Path) -> Path:
    """Download CrowS-Pairs CSV if not cached."""
    out = dest / "crows_pairs.csv"
    if out.exists():
        logger.info("CrowS-Pairs already cached at %s", out)
        return out
    logger.info("Downloading CrowS-Pairs …")
    resp = requests.get(CROWS_URL, timeout=60)
    resp.raise_for_status()
    out.write_text(resp.text, encoding="utf-8")
    return out


def load_crows_pairs(paths: PathConfig) -> pd.DataFrame:
    """Load CrowS-Pairs into a DataFrame."""
    csv_path = download_crows_pairs(paths.data_raw)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = {"sent_more", "sent_less", "stereo_antistereo", "bias_type"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    logger.info("Loaded CrowS-Pairs: %d pairs", len(df))
    return df


# ── SNIPS ────────────────────────────────────────────────────

def _download_snips_intent(intent: str, dest: Path) -> Path:
    out = dest / f"snips_{intent}.json"
    if out.exists():
        return out
    url = SNIPS_BASE.format(intent=intent)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out.write_text(resp.text, encoding="utf-8")
    return out


def _parse_snips_json(path: Path, intent: str) -> list[dict]:
    """Parse one SNIPS intent JSON into flat records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    utterances = raw.get(intent, [])
    records = []
    for utt in utterances:
        segments = utt.get("data", [])
        text_parts, slots = [], []
        offset = 0
        for seg in segments:
            txt = seg["text"]
            if "entity" in seg:
                slots.append({
                    "text": txt,
                    "slot": seg["entity"],
                    "start": offset,
                    "end": offset + len(txt),
                })
            text_parts.append(txt)
            offset += len(txt)
        records.append({
            "text": "".join(text_parts),
            "intent": intent,
            "slots": slots,
        })
    return records


def load_snips(paths: PathConfig) -> pd.DataFrame:
    """Download & parse all SNIPS intents into a single DataFrame."""
    all_records = []
    for intent in SNIPS_INTENTS:
        p = _download_snips_intent(intent, paths.data_raw)
        all_records.extend(_parse_snips_json(p, intent))
    df = pd.DataFrame(all_records)
    logger.info("Loaded SNIPS: %d utterances, %d intents",
                len(df), df["intent"].nunique())
    return df


def snips_to_bio(row: dict) -> list[str]:
    """Convert a SNIPS row with slot spans to BIO tag list (char-aligned)."""
    text = row["text"]
    tokens = text.split()
    tags = ["O"] * len(tokens)
    char2tok = {}
    idx = 0
    for ti, tok in enumerate(tokens):
        for c in range(idx, idx + len(tok)):
            char2tok[c] = ti
        idx += len(tok) + 1  # +1 for space

    for slot in row["slots"]:
        start_tok = char2tok.get(slot["start"])
        end_tok = char2tok.get(slot["end"] - 1)
        if start_tok is None or end_tok is None:
            continue
        tags[start_tok] = f"B-{slot['slot']}"
        for t in range(start_tok + 1, end_tok + 1):
            tags[t] = f"I-{slot['slot']}"
    return tags
