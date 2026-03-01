# Sentiment Analysis of Audio Using BERT

End-to-end sentiment analysis pipeline that transcribes audio files and classifies sentiment using BERT. Supports direct text input alongside audio analysis, fine-tuning on IMDB, and both pre-trained and custom model inference.

## Architecture

```
Audio File ──► [OpenAI Whisper ASR] ──► Transcript ──►
                                                        [BERT Classifier] ──► Sentiment + Confidence
Text Input ─────────────────────────────────────────►
```

## Features

- Audio-to-sentiment pipeline: speech transcription with [OpenAI Whisper](https://github.com/openai/whisper) followed by BERT classification
- Fine-tune `bert-base-uncased` on the IMDB dataset with a configurable training loop
- Zero-shot inference with `distilbert-base-uncased-finetuned-sst-2-english` (no training required)
- Evaluation with accuracy, F1, classification report, and confusion matrix plots
- Training history visualization
- JSON output mode for programmatic use
- Modular `src/` package — importable as a library

## Project Structure

```
├── src/
│   ├── config.py       # Hyperparameters and constants
│   ├── model.py        # BertSentimentClassifier
│   ├── data.py         # Dataset classes and IMDB loader
│   ├── trainer.py      # Training loop with AdamW + warmup
│   ├── evaluator.py    # Metrics, plots, confusion matrix
│   ├── predictor.py    # Text inference (pipeline or custom model)
│   └── audio.py        # Whisper-based audio transcription
├── train.py            # Training entry point
├── predict.py          # Prediction CLI (text + audio)
├── evaluate.py         # Evaluation entry point
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/punyamodi/Sentiment-Analysis-of-Audio-Using-BERT.git
cd Sentiment-Analysis-of-Audio-Using-BERT
pip install -r requirements.txt
```

> For GPU support, install the CUDA-enabled PyTorch build first from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Predict from text

```bash
python predict.py --text "The cinematography was stunning and the story deeply moving."
```

Multiple inputs:

```bash
python predict.py --text "Loved it!" "Waste of time." "Pretty decent overall."
```

### Predict from audio

```bash
python predict.py --audio review.wav
```

Multiple files with a specific Whisper model:

```bash
python predict.py --audio clip1.mp3 clip2.wav --whisper-model small
```

### Use a fine-tuned checkpoint

```bash
python predict.py --text "Absolutely brilliant." --model-path checkpoints/best_model.pt
```

### JSON output

```bash
python predict.py --text "Great film!" --json
```

```json
[
  {
    "text": "Great film!",
    "label": "positive",
    "score": 0.9987
  }
]
```

### Train

Fine-tune on the full IMDB dataset (25 K train / 2.5 K validation):

```bash
python train.py
```

Custom hyperparameters:

```bash
python train.py --num-epochs 5 --batch-size 16 --learning-rate 3e-5 --plot-history
```

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `bert-base-uncased` | HuggingFace model ID |
| `--batch-size` | `32` | Training batch size |
| `--learning-rate` | `2e-5` | Initial learning rate |
| `--num-epochs` | `3` | Training epochs |
| `--save-dir` | `checkpoints` | Checkpoint directory |
| `--plot-history` | off | Save loss/metric curves |

### Evaluate

```bash
python evaluate.py --model-path checkpoints/best_model.pt --plot-cm
```

## Models

| Use case | Model | Accuracy |
|---|---|---|
| Default inference | `distilbert-base-uncased-finetuned-sst-2-english` | ~93% SST-2 |
| Custom fine-tune | `bert-base-uncased` on IMDB | ~93% test |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.34+
- FFMPEG (required by Whisper for non-WAV audio): `sudo apt install ffmpeg` / `brew install ffmpeg`

## License

MIT
