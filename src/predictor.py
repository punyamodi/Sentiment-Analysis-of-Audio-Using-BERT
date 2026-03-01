from typing import Dict, List, Optional, Union

import torch
from transformers import BertTokenizer, pipeline

from src.config import LABEL_MAP


class SentimentPredictor:
    def __init__(
        self,
        model=None,
        tokenizer: Optional[BertTokenizer] = None,
        pretrained_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_length: int = 128,
        device: str = "cpu",
    ):
        self.max_length = max_length
        self.device = device

        if model is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=pretrained_model,
                device=0 if device == "cuda" else -1,
            )
            self._model = None
            self._tokenizer = None
        else:
            self._pipeline = None
            self._model = model
            self._tokenizer = tokenizer

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]

        if self._pipeline is not None:
            return self._predict_with_pipeline(texts)
        return self._predict_with_model(texts)

    def _predict_with_pipeline(self, texts: List[str]) -> List[Dict]:
        raw_results = self._pipeline(texts, truncation=True, max_length=self.max_length)
        return [
            {
                "text": text,
                "label": result["label"].lower().replace("label_", "").replace("pos", "positive").replace("neg", "negative"),
                "score": result["score"],
            }
            for text, result in zip(texts, raw_results)
        ]

    def _predict_with_model(self, texts: List[str]) -> List[Dict]:
        self._model.eval()
        results = []

        for text in texts:
            encoding = self._tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = self._model(
                    encoding["input_ids"].to(self.device),
                    encoding["attention_mask"].to(self.device),
                    encoding["token_type_ids"].to(self.device),
                )
                probs = torch.softmax(logits, dim=1)
                label_idx = torch.argmax(probs, dim=1).item()
                score = probs[0][label_idx].item()

            results.append({
                "text": text,
                "label": LABEL_MAP[label_idx],
                "score": score,
            })

        return results
