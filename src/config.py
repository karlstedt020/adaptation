"""Centralized configuration for the adaptation pipeline."""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """All project paths."""
    root: Path = Path("D:/adaptation")

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
    api_key: str = ""       # <-- SET YOUR KEY
    model: str = ""         # <-- SET YOUR MODEL (e.g. "anthropic/claude-sonnet-4")
    max_retries: int = 3
    timeout: int = 120
    temperature: float = 0.7
    n_variants: int = 3     # how many adaptation variants to generate


@dataclass
class EvalConfig:
    """Models used for evaluation."""
    bias_models: list = field(default_factory=lambda: [
        "DeepPavlov/rubert-base-cased",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
    ])
    slot_models: list = field(default_factory=lambda: [
        "DeepPavlov/rubert-base-cased",
        "xlm-roberta-base",
    ])
    n_runs: int = 3          # number of evaluation runs for aggregation
    slot_epochs: int = 5
    slot_batch_size: int = 32
    slot_lr: float = 5e-5


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    spacy_model: str = "en_core_web_sm"
