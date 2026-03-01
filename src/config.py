import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    pretrained_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    seed: int = 42
    dataset_name: str = "imdb"
    save_dir: str = "checkpoints"
    whisper_model_size: str = "base"
    num_classes: int = 2
    num_workers: int = 4
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


LABEL_MAP = {0: "negative", 1: "positive"}
DEFAULT_CONFIG = Config()
