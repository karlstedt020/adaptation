"""CrowS-Pairs bias metric — works for both masked and causal LMs."""

import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

logger = logging.getLogger(__name__)


# ── Device helpers ──────────────────────────────────────────

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_model(model) -> None:
    """Release a model's GPU memory and trigger GC."""
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Model loading ───────────────────────────────────────────

def load_model(model_name: str, model_type: str) -> tuple:
    """Load an MLM or causal LM together with its tokenizer."""
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if model_type == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
    elif model_type == "causal":
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    return model, tokenizer, device


# ── Scoring: MLM pseudo-log-likelihood ──────────────────────

def compute_pll_mlm(sentence: str, model, tokenizer, device) -> float:
    """Pseudo-log-likelihood via one-at-a-time masking (CrowS-Pairs style)."""
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
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        log_prob += log_probs[true_id].item()

    return log_prob


# ── Scoring: causal LM log-likelihood ───────────────────────

def compute_ll_causal(sentence: str, model, tokenizer, device) -> float:
    """Sum of log P(x_i | x_<i) across the sentence."""
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq, vocab)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item()


# ── Pair-level comparison ───────────────────────────────────

def _score(sentence: str, model, tokenizer, device, model_type: str) -> float:
    if model_type == "mlm":
        return compute_pll_mlm(sentence, model, tokenizer, device)
    return compute_ll_causal(sentence, model, tokenizer, device)


def evaluate_pair(
    sent_more: str, sent_less: str,
    model, tokenizer, device, model_type: str,
) -> bool:
    """Return True if the model prefers the stereotypical sentence."""
    s_more = _score(sent_more, model, tokenizer, device, model_type)
    s_less = _score(sent_less, model, tokenizer, device, model_type)
    return s_more > s_less


# ── Dataset-level evaluation ────────────────────────────────

def evaluate_bias_dataset(
    df: pd.DataFrame,
    model, tokenizer, device, model_type: str,
    sent_more_col: str = "sent_more_ru",
    sent_less_col: str = "sent_less_ru",
) -> dict:
    """Compute bias metric for all pairs using a preloaded model."""
    results_by_type = defaultdict(list)
    all_results = []

    for _, row in df.iterrows():
        sm = row[sent_more_col]
        sl = row[sent_less_col]
        if not sm or not sl or pd.isna(sm) or pd.isna(sl):
            continue
        prefers_stereo = evaluate_pair(
            sm, sl, model, tokenizer, device, model_type,
        )
        all_results.append(prefers_stereo)
        results_by_type[row["bias_type"]].append(prefers_stereo)

    overall = float(np.mean(all_results)) if all_results else 0.0
    per_type = {bt: float(np.mean(v)) for bt, v in results_by_type.items()}
    return {"overall": overall, "per_type": per_type, "n": len(all_results)}


def run_bias_evaluation(
    df: pd.DataFrame,
    model_names: list,
    sent_more_col: str = "sent_more_ru",
    sent_less_col: str = "sent_less_ru",
    n_runs: int = 1,
) -> pd.DataFrame:
    """Run bias evaluation across models, aggregate, free memory between models.

    Args:
        model_names: list of (model_name, model_type) tuples, OR bare strings
                     (treated as "mlm" for backwards compatibility).
    """
    records = []
    for entry in model_names:
        mname, mtype = entry if isinstance(entry, tuple) else (entry, "mlm")
        logger.info("Evaluating bias with %s (%s) …", mname, mtype)
        model, tok, dev = load_model(mname, mtype)
        try:
            for run_i in range(n_runs):
                res = evaluate_bias_dataset(
                    df, model, tok, dev, mtype,
                    sent_more_col, sent_less_col,
                )
                records.append({
                    "model": mname,
                    "model_type": mtype,
                    "run": run_i,
                    "metric_score": res["overall"],
                    "n_pairs": res["n"],
                    **{f"bias_{bt}": v for bt, v in res["per_type"].items()},
                })
        finally:
            _free_model(model)
            del tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Freed memory after %s", mname)

    return pd.DataFrame(records)
