from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, dataset: Dataset, batch_size: int, num_workers: int = 4) -> Dict:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"]

                logits = self.model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary"),
            "classification_report": classification_report(
                all_labels, all_preds, target_names=["negative", "positive"]
            ),
            "confusion_matrix": confusion_matrix(all_labels, all_preds),
        }

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        output_path: str = "training_history.png",
    ) -> None:
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, history["train_loss"], "r-o", label="Training Loss")
        axes[0].plot(epochs, history["val_loss"], "b-o", label="Validation Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["val_accuracy"], "g-o", label="Accuracy")
        axes[1].plot(epochs, history["val_f1"], "m-o", label="F1 Score")
        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Training history saved: {output_path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        output_path: str = "confusion_matrix.png",
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            title="Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved: {output_path}")
