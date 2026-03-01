import argparse
import json
import sys
from typing import List

import torch
from transformers import BertTokenizer

from src.audio import AudioTranscriber
from src.config import DEFAULT_CONFIG
from src.predictor import SentimentPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment analysis on text or audio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", nargs="+", metavar="TEXT", help="Text inputs to analyze")
    group.add_argument("--audio", nargs="+", metavar="FILE", help="Audio file paths to transcribe and analyze")
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        help="Path to a fine-tuned model checkpoint (uses pre-trained DistilBERT if omitted)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_CONFIG.model_name,
        help="BERT variant name (used with --model-path)",
    )
    parser.add_argument(
        "--pretrained-model",
        default=DEFAULT_CONFIG.pretrained_model,
        help="HuggingFace model to use when --model-path is not provided",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_CONFIG.whisper_model_size,
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG.max_length)
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output results as JSON")
    return parser.parse_args()


def build_predictor(args: argparse.Namespace) -> SentimentPredictor:
    device = DEFAULT_CONFIG.device

    if args.model_path:
        from src.model import BertSentimentClassifier

        model = BertSentimentClassifier.load(
            path=args.model_path,
            model_name=args.model_name,
            num_classes=DEFAULT_CONFIG.num_classes,
            dropout_rate=DEFAULT_CONFIG.dropout_rate,
            map_location=device,
        ).to(torch.device(device))
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        return SentimentPredictor(
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            device=device,
        )

    return SentimentPredictor(
        pretrained_model=args.pretrained_model,
        max_length=args.max_length,
        device=device,
    )


def print_results(results: List[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        label = result["label"].upper()
        score = result["score"]
        text_preview = result["text"][:100].replace("\n", " ")
        if len(result["text"]) > 100:
            text_preview += "..."
        print(f"[{label}] ({score:.4f})  {text_preview}")


def main() -> None:
    args = parse_args()
    predictor = build_predictor(args)

    if args.audio:
        transcriber = AudioTranscriber(model_size=args.whisper_model)
        print("Transcribing audio files...\n")
        texts = []
        for audio_path in args.audio:
            transcript = transcriber.transcribe(audio_path)
            print(f"  {audio_path}")
            print(f"  Transcript: {transcript}\n")
            texts.append(transcript)
    else:
        texts = args.text

    results = predictor.predict(texts)
    print_results(results, args.output_json)


if __name__ == "__main__":
    main()
