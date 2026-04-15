"""LLM-based cultural shift verification.

Uses an LLM to evaluate whether the adapted dataset actually reflects
a cultural shift towards Russian context (vs. remaining American).
"""

import json
import logging

import pandas as pd
from tqdm import tqdm

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert evaluator of cultural adaptation quality. "
    "Given an original English sentence and its Russian adaptation, "
    "assess whether a genuine cultural shift to the Russian context occurred.\n"
    "Score from 1 to 5:\n"
    "  1 = No adaptation, just literal translation with American realities\n"
    "  2 = Partial adaptation, some entities changed but core context is American\n"
    "  3 = Moderate adaptation, most entities changed but some inconsistencies\n"
    "  4 = Good adaptation, natural Russian context with minor issues\n"
    "  5 = Excellent adaptation, reads as natively Russian\n"
    'Return ONLY JSON: {"score": N, "reason": "..."}'
)

CHECK_TEMPLATE = """\
Original (EN): {original}
Adapted (RU): {adapted}
Category: {category}

Rate the cultural shift quality (1-5).
"""


def _parse_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)


def check_cultural_shift(
    client: LLMClient,
    original: str,
    adapted: str,
    category: str = "",
) -> dict:
    """Score a single adaptation for cultural shift quality."""
    prompt = CHECK_TEMPLATE.format(
        original=original, adapted=adapted, category=category,
    )
    raw = client.complete(prompt, SYSTEM_PROMPT, temperature=0.0)
    return _parse_json(raw)


def check_crows_dataset(
    client: LLMClient,
    df: pd.DataFrame,
    sent_col: str = "sent_more_ru",
    sample_size: int = 100,
) -> pd.DataFrame:
    """Evaluate cultural shift on a sample of CrowS-Pairs."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    records = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Shift check"):
        try:
            res = check_cultural_shift(
                client,
                row["sent_more"],
                row.get(sent_col, ""),
                row.get("bias_type", ""),
            )
            records.append({
                "index": row.name,
                "score": res.get("score", 0),
                "reason": res.get("reason", ""),
            })
        except Exception as exc:
            logger.warning("Shift check failed: %s", exc)
            records.append({"index": row.name, "score": 0, "reason": str(exc)})
    return pd.DataFrame(records)


def run_shift_evaluation(
    client: LLMClient,
    datasets: dict[str, pd.DataFrame],
    sent_col: str = "sent_more_ru",
    sample_size: int = 100,
    n_runs: int = 3,
) -> pd.DataFrame:
    """Run cultural shift check across multiple datasets and runs.

    Args:
        datasets: {"method_name": dataframe, ...}
    """
    records = []
    for method, df in datasets.items():
        for run_i in range(n_runs):
            logger.info("Shift eval: %s, run %d", method, run_i)
            result_df = check_crows_dataset(
                client, df, sent_col, sample_size
            )
            records.append({
                "method": method,
                "run": run_i,
                "mean_score": result_df["score"].mean(),
                "std_score": result_df["score"].std(),
                "n_evaluated": len(result_df),
            })
    return pd.DataFrame(records)
