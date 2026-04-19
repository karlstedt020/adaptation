"""Centralized configuration for the adaptation pipeline."""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """All project paths."""
    root: Path = Path(".")

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_naive_translated(self) -> Path:
        return self.root / "data" / "naive_translated"

    @property
    def data_naive_llm(self) -> Path:
        return self.root / "data" / "naive_llm"

    @property
    def data_adapted(self) -> Path:
        return self.root / "data" / "adapted"

    def ensure_dirs(self):
        for d in [self.data_raw, self.data_naive_translated,
                  self.data_naive_llm, self.data_adapted]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class OpenRouterConfig:
    """OpenRouter API settings — user fills in key and model."""
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""                                   # <-- SET YOUR KEY
    model: str = "deepseek/deepseek-chat-v3.1"          # main translation/adaptation model
    ner_model: str = "google/gemini-2.0-flash-001"      # NER model (cheap, great at JSON)
    max_retries: int = 3
    timeout: int = 60
    n_variants: int = 3          # adaptation variants per example

    # ── Concurrency ──────────────────────────────────────────
    # These limits apply globally across all threads to the same API.
    max_workers: int = 4         # thread-pool size for parallel calls
    rps: float = 4.0             # max LLM requests per second (all models combined)
    translator_rps: float = 5.0  # max Google Translate requests per second


@dataclass
class EvalConfig:
    """Models used for evaluation."""
    # Each entry: (hf_model_name, model_type) where model_type is "mlm" or "causal".
    # All entries below fit in a 15 GB VRAM Colab GPU in fp16.
    bias_models: list = field(default_factory=lambda: [
        ("DeepPavlov/rubert-base-cased",            "mlm"),
        ("bert-base-multilingual-cased",            "mlm"),
        ("xlm-roberta-base",                        "mlm"),
        ("RefalMachine/RuadaptQwen2.5-3B-Instruct", "causal"),  # ≈ 6 GB in fp16
    ])
    slot_models: list = field(default_factory=lambda: [
        "DeepPavlov/rubert-base-cased",
        "xlm-roberta-base",
    ])
    n_runs: int = 3
    slot_epochs: int = 5
    slot_batch_size: int = 32
    slot_lr: float = 5e-5


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
