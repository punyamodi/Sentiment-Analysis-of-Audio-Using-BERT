import torch
import torch.nn as nn
from transformers import BertModel


class BertSentimentClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, model_name: str, num_classes: int = 2, dropout_rate: float = 0.1, map_location: str = "cpu") -> "BertSentimentClassifier":
        model = cls(model_name=model_name, num_classes=num_classes, dropout_rate=dropout_rate)
        model.load_state_dict(torch.load(path, map_location=map_location))
        return model
