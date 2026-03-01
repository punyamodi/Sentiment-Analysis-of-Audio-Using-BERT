import argparse

import torch
from transformers import BertTokenizer

from src.config import DEFAULT_CONFIG
from src.data import load_imdb_dataset
from src.evaluator import ModelEvaluator
from src.model import BertSentimentClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT sentiment model on IMDB test set")
    parser.add_argument("--model-path", required=True, metavar="PATH", help="Path to model checkpoint")
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG.max_length)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--dropout-rate", type=float, default=DEFAULT_CONFIG.dropout_rate)
    parser.add_argument("--plot-cm", action="store_true", help="Save confusion matrix plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(DEFAULT_CONFIG.device)
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    print("Loading IMDB test set...")
    test_dataset = load_imdb_dataset("test", tokenizer, args.max_length)

    model = BertSentimentClassifier.load(
        path=args.model_path,
        model_name=args.model_name,
        num_classes=DEFAULT_CONFIG.num_classes,
        dropout_rate=args.dropout_rate,
        map_location=str(device),
    ).to(device)

    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_dataset, args.batch_size, args.num_workers)

    print(f"\nAccuracy : {results['accuracy']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"\n{results['classification_report']}")

    if args.plot_cm:
        evaluator.plot_confusion_matrix(results["confusion_matrix"])


if __name__ == "__main__":
    main()
