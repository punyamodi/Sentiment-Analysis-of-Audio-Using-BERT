import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class SentimentTrainer:
    def __init__(self, model: nn.Module, device: torch.device, save_dir: str):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        warmup_ratio: float,
        weight_decay: float,
        num_workers: int = 4,
    ) -> Dict[str, List[float]]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            val_loss, val_accuracy, val_f1 = self._evaluate(val_loader, criterion)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["val_f1"].append(val_f1)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_checkpoint("best_model.pt")

        return history

    def _evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ):
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

        return avg_loss, accuracy, f1

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")
