"""CrowS-Pairs bias metric via pseudo-log-likelihood with masked LMs."""

import logging
import math
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_mlm(model_name: str) -> tuple:
    """Load a masked language model and its tokenizer."""
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def compute_pll(
    sentence: str,
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Compute pseudo-log-likelihood of a sentence.

    Mask each token one at a time, sum log-probs of the true token.
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    mask_id = tokenizer.mask_token_id
    log_prob = 0.0

    for i in range(1, len(input_ids) - 1):  # skip [CLS] and [SEP]
        masked = input_ids.clone()
        true_id = masked[i].item()
        masked[i] = mask_id

        with torch.no_grad():
            logits = model(masked.unsqueeze(0)).logits[0, i]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob += log_probs[true_id].item()

    return log_prob


def evaluate_pair(
    sent_more: str,
    sent_less: str,
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> bool:
    """Return True if the model prefers the stereotypical sentence."""
    pll_more = compute_pll(sent_more, model, tokenizer, device)
    pll_less = compute_pll(sent_less, model, tokenizer, device)
    return pll_more > pll_less


def evaluate_bias_dataset(
    df: pd.DataFrame,
    model_name: str,
    sent_more_col: str = "sent_more_ru",
    sent_less_col: str = "sent_less_ru",
) -> dict:
    """Compute CrowS-Pairs bias metric for an entire dataset.

    Returns overall score and per-bias_type scores.
    """
    model, tokenizer, device = load_mlm(model_name)
    results_by_type = defaultdict(list)
    all_results = []

    for _, row in df.iterrows():
        sm = row[sent_more_col]
        sl = row[sent_less_col]
        if not sm or not sl or pd.isna(sm) or pd.isna(sl):
            continue
        prefers_stereo = evaluate_pair(sm, sl, model, tokenizer, device)
        all_results.append(prefers_stereo)
        results_by_type[row["bias_type"]].append(prefers_stereo)

    overall = np.mean(all_results) if all_results else 0.0
    per_type = {
        bt: np.mean(vals) for bt, vals in results_by_type.items()
    }
    return {"overall": overall, "per_type": per_type, "n": len(all_results)}


def run_bias_evaluation(
    df: pd.DataFrame,
    model_names: list[str],
    sent_more_col: str = "sent_more_ru",
    sent_less_col: str = "sent_less_ru",
    n_runs: int = 1,
) -> pd.DataFrame:
    """Run bias evaluation across multiple models, aggregate results."""
    records = []
    for mname in model_names:
        logger.info("Evaluating bias with %s …", mname)
        for run_i in range(n_runs):
            res = evaluate_bias_dataset(
                df, mname, sent_more_col, sent_less_col
            )
            records.append({
                "model": mname,
                "run": run_i,
                "metric_score": res["overall"],
                "n_pairs": res["n"],
                **{f"bias_{bt}": v for bt, v in res["per_type"].items()},
            })
    return pd.DataFrame(records)
