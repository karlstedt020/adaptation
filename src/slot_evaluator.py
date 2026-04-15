"""Joint intent classification & slot filling evaluation on SNIPS."""

import logging
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────

class NLUDataset(Dataset):
    """Token-level dataset for joint intent + slot filling."""

    def __init__(self, texts, intents, bio_tags, tokenizer, intent2id, tag2id, max_len=128):
        self.encodings = []
        self.intent_ids = []
        self.tag_ids_list = []

        for text, intent, tags in zip(texts, intents, bio_tags):
            tokens = text.split()
            enc = tokenizer(
                tokens, is_split_into_words=True,
                truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt",
            )
            word_ids = enc.word_ids()
            aligned_tags = []
            for wid in word_ids:
                if wid is None:
                    aligned_tags.append(tag2id["O"])
                else:
                    aligned_tags.append(tag2id.get(tags[wid], tag2id["O"]))

            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            self.intent_ids.append(intent2id[intent])
            self.tag_ids_list.append(torch.tensor(aligned_tags))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["intent_label"] = torch.tensor(self.intent_ids[idx])
        item["tag_labels"] = self.tag_ids_list[idx]
        return item


# ── Model ────────────────────────────────────────────────────

class JointNLUModel(nn.Module):
    """BERT + linear heads for intent classification and slot tagging."""

    def __init__(self, model_name: str, n_intents: int, n_tags: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.intent_head = nn.Linear(hidden, n_intents)
        self.slot_head = nn.Linear(hidden, n_tags)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]
        intent_logits = self.intent_head(cls_repr)
        slot_logits = self.slot_head(out.last_hidden_state)
        return intent_logits, slot_logits


# ── Training ─────────────────────────────────────────────────

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_label_maps(df: pd.DataFrame, bio_tags_col: str):
    """Build intent2id and tag2id from the dataset."""
    intents = sorted(df["intent"].unique())
    intent2id = {i: idx for idx, i in enumerate(intents)}
    all_tags = {"O"}
    for tags in df[bio_tags_col]:
        all_tags.update(tags)
    tag_list = sorted(all_tags)
    tag2id = {t: idx for idx, t in enumerate(tag_list)}
    return intent2id, tag2id


def train_model(
    train_df: pd.DataFrame,
    model_name: str,
    text_col: str,
    bio_tags_col: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
    seed: int = 42,
) -> tuple:
    """Train joint NLU model, return (model, tokenizer, intent2id, tag2id)."""
    torch.manual_seed(seed)
    device = _get_device()

    intent2id, tag2id = build_label_maps(train_df, bio_tags_col)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = NLUDataset(
        train_df[text_col].tolist(),
        train_df["intent"].tolist(),
        train_df[bio_tags_col].tolist(),
        tokenizer, intent2id, tag2id,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = JointNLUModel(model_name, len(intent2id), len(tag2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(loader) * epochs,
    )
    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=tag2id["O"])

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            intent_logits, slot_logits = model(
                batch["input_ids"], batch["attention_mask"]
            )
            loss_i = intent_loss_fn(intent_logits, batch["intent_label"])
            loss_s = slot_loss_fn(
                slot_logits.view(-1, slot_logits.size(-1)),
                batch["tag_labels"].view(-1),
            )
            loss = loss_i + loss_s
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        logger.info("Epoch %d loss: %.4f", epoch + 1, total_loss / len(loader))

    return model, tokenizer, intent2id, tag2id


# ── Evaluation ───────────────────────────────────────────────

def evaluate_model(
    model: JointNLUModel,
    test_df: pd.DataFrame,
    tokenizer,
    intent2id: dict,
    tag2id: dict,
    text_col: str,
    bio_tags_col: str,
    batch_size: int = 64,
) -> dict:
    """Evaluate intent accuracy and slot F1."""
    device = _get_device()
    id2intent = {v: k for k, v in intent2id.items()}
    id2tag = {v: k for k, v in tag2id.items()}

    dataset = NLUDataset(
        test_df[text_col].tolist(),
        test_df["intent"].tolist(),
        test_df[bio_tags_col].tolist(),
        tokenizer, intent2id, tag2id,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    all_intent_pred, all_intent_true = [], []
    all_slot_pred, all_slot_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            intent_logits, slot_logits = model(
                batch["input_ids"], batch["attention_mask"]
            )
            # intent
            preds_i = intent_logits.argmax(dim=-1).cpu().tolist()
            all_intent_pred.extend(preds_i)
            all_intent_true.extend(batch["intent_label"].cpu().tolist())
            # slots
            preds_s = slot_logits.argmax(dim=-1).cpu()
            labels_s = batch["tag_labels"].cpu()
            mask = batch["attention_mask"].cpu().bool()
            for p_row, l_row, m_row in zip(preds_s, labels_s, mask):
                p_tags = [id2tag.get(t.item(), "O") for t, m in zip(p_row, m_row) if m]
                l_tags = [id2tag.get(t.item(), "O") for t, m in zip(l_row, m_row) if m]
                all_slot_pred.append(p_tags)
                all_slot_true.append(l_tags)

    intent_acc = accuracy_score(all_intent_true, all_intent_pred)
    slot_f1 = seq_f1(all_slot_true, all_slot_pred, average="micro")
    return {"intent_accuracy": intent_acc, "slot_f1": slot_f1}


def run_slot_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str],
    text_col: str = "text_ru",
    bio_tags_col: str = "bio_tags",
    n_runs: int = 3,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
) -> pd.DataFrame:
    """Train & evaluate across models and runs, return aggregated results."""
    records = []
    for mname in model_names:
        for run_i in range(n_runs):
            logger.info("Slot eval: %s run %d", mname, run_i)
            model, tok, i2id, t2id = train_model(
                train_df, mname, text_col, bio_tags_col,
                epochs=epochs, batch_size=batch_size,
                lr=lr, seed=42 + run_i,
            )
            metrics = evaluate_model(
                model, test_df, tok, i2id, t2id,
                text_col, bio_tags_col, batch_size,
            )
            records.append({
                "model": mname,
                "run": run_i,
                **metrics,
            })
            # free GPU memory
            del model
            torch.cuda.empty_cache()
    return pd.DataFrame(records)
