import argparse
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import torch
from transformers import BertTokenizer

from src.config import Config, DEFAULT_CONFIG
from src.data import load_imdb_dataset
from src.evaluator import ModelEvaluator
from src.model import BertSentimentClassifier
from src.trainer import SentimentTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis on IMDB")
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG.max_length)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_CONFIG.num_epochs)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_CONFIG.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_CONFIG.dropout_rate)
    parser.add_argument("--save-dir", default=DEFAULT_CONFIG.save_dir)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--plot-history", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(DEFAULT_CONFIG.device)
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    print("Loading IMDB dataset...")
    full_train = load_imdb_dataset("train", tokenizer, args.max_length)
    test_dataset = load_imdb_dataset("test", tokenizer, args.max_length)

    val_size = int(args.val_split * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")

    model = BertSentimentClassifier(
        model_name=args.model_name,
        num_classes=DEFAULT_CONFIG.num_classes,
        dropout_rate=args.dropout_rate,
    ).to(device)

    trainer = SentimentTrainer(model, device, args.save_dir)
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )

    print("\nEvaluating on test set...")
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_dataset, args.batch_size, args.num_workers)

    print(f"\nTest Accuracy : {results['accuracy']:.4f}")
    print(f"Test F1       : {results['f1']:.4f}")
    print(f"\n{results['classification_report']}")

    evaluator.plot_confusion_matrix(results["confusion_matrix"])

    if args.plot_history:
        evaluator.plot_training_history(history)


if __name__ == "__main__":
    main()
