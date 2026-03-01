import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def test(name):
    def decorator(fn):
        try:
            fn()
            print(f"  {PASS}  {name}")
            _results.append((name, True, None))
        except Exception as e:
            print(f"  {FAIL}  {name}")
            print(f"         {type(e).__name__}: {e}")
            _results.append((name, False, str(e)))
    return decorator


print("\n=== Config ===")

@test("Config dataclass instantiation")
def _():
    from src.config import Config, LABEL_MAP, DEFAULT_CONFIG
    assert DEFAULT_CONFIG.max_length == 128
    assert DEFAULT_CONFIG.num_classes == 2
    assert DEFAULT_CONFIG.device in ("cuda", "cpu")
    assert LABEL_MAP[0] == "negative"
    assert LABEL_MAP[1] == "positive"

@test("Config custom values")
def _():
    from src.config import Config
    c = Config(batch_size=16, num_epochs=5)
    assert c.batch_size == 16
    assert c.num_epochs == 5

print("\n=== Model ===")

@test("BertSentimentClassifier forward pass")
def _():
    from src.model import BertSentimentClassifier
    model = BertSentimentClassifier("distilbert-base-uncased", num_classes=2, dropout_rate=0.1)
    model.eval()
    batch = 2
    seq = 32
    input_ids = torch.randint(0, 100, (batch, seq))
    mask = torch.ones(batch, seq, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, mask)
    assert out.shape == (batch, 2)

@test("BertSentimentClassifier save and load")
def _():
    import tempfile
    from src.model import BertSentimentClassifier
    model = BertSentimentClassifier("distilbert-base-uncased", num_classes=2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        loaded = BertSentimentClassifier.load(path, "distilbert-base-uncased")
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
            assert torch.allclose(v1, v2), f"Mismatch at {k1}"
    finally:
        os.unlink(path)

print("\n=== Data ===")

@test("IMDBDataset item shape")
def _():
    from src.data import IMDBDataset
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = IMDBDataset(["I loved this film."], [1], tok, max_length=64)
    assert len(ds) == 1
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["labels"].item() == 1

@test("TextDataset without labels")
def _():
    from src.data import TextDataset
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = TextDataset(["hello world"], tok, max_length=32)
    item = ds[0]
    assert "labels" not in item
    assert item["input_ids"].shape == (32,)

print("\n=== Predictor (pre-trained pipeline) ===")

@test("Single text prediction")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    results = pred.predict("I loved this film!")
    assert len(results) == 1
    assert results[0]["label"] in ("positive", "negative")
    assert 0.0 <= results[0]["score"] <= 1.0

@test("Batch text prediction")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    texts = ["Amazing!", "Terrible.", "Okay I guess."]
    results = pred.predict(texts)
    assert len(results) == 3
    for r in results:
        assert r["label"] in ("positive", "negative")

@test("Positive text correctly classified")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    r = pred.predict("This film is a masterpiece.")
    assert r[0]["label"] == "positive"

@test("Negative text correctly classified")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    r = pred.predict("Absolute garbage. Worst movie I have ever seen.")
    assert r[0]["label"] == "negative"

@test("Empty-ish input does not crash")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    r = pred.predict("ok")
    assert r[0]["label"] in ("positive", "negative")

@test("Very long text is truncated without error")
def _():
    from src.predictor import SentimentPredictor
    from src.config import DEFAULT_CONFIG
    pred = SentimentPredictor(device=DEFAULT_CONFIG.device)
    long_text = "great movie " * 500
    r = pred.predict(long_text)
    assert r[0]["label"] in ("positive", "negative")

print("\n=== Evaluator ===")

@test("Plot training history produces file")
def _():
    import tempfile
    from src.evaluator import ModelEvaluator
    dummy_model = type("M", (), {"eval": lambda self: None})()
    ev = ModelEvaluator(dummy_model, torch.device("cpu"))
    history = {
        "train_loss": [0.5, 0.3, 0.2],
        "val_loss": [0.4, 0.25, 0.18],
        "val_accuracy": [0.85, 0.91, 0.93],
        "val_f1": [0.84, 0.90, 0.92],
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        ev.plot_training_history(history, output_path=path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)

@test("Plot confusion matrix produces file")
def _():
    import tempfile
    from src.evaluator import ModelEvaluator
    dummy_model = type("M", (), {"eval": lambda self: None})()
    ev = ModelEvaluator(dummy_model, torch.device("cpu"))
    cm = np.array([[4800, 200], [150, 4850]])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        ev.plot_confusion_matrix(cm, output_path=path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)

print("\n=== Audio Transcriber ===")

@test("Unsupported format raises ValueError")
def _():
    from src.audio import AudioTranscriber
    transcriber = AudioTranscriber.__new__(AudioTranscriber)
    transcriber.model = None
    try:
        transcriber.transcribe("file.xyz")
        assert False, "should have raised"
    except ValueError:
        pass

@test("Missing file raises FileNotFoundError")
def _():
    from src.audio import AudioTranscriber
    transcriber = AudioTranscriber.__new__(AudioTranscriber)
    transcriber.model = object()
    try:
        transcriber.transcribe("/nonexistent/path/file.wav")
        assert False, "should have raised"
    except FileNotFoundError:
        pass

print("\n" + "=" * 50)
passed = sum(1 for _, ok, _ in _results if ok)
total = len(_results)
failed = [(n, e) for n, ok, e in _results if not ok]
print(f"\n  Results: {passed}/{total} tests passed")
if failed:
    print("\n  Failed tests:")
    for name, err in failed:
        print(f"    - {name}: {err}")
print()
sys.exit(0 if not failed else 1)
