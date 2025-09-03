"""Utility for efficient batched predictions with PyTorch models."""
from typing import List
import torch
from biobert_ner.utils import Profile


class FastPredict:
    """Wraps a PyTorch model to provide a simple ``predict`` API.

    The model is kept in evaluation mode and moved to the desired device once
    during initialisation to avoid repeated graph construction similar to the
    original TensorFlow-based utility.
    """

    def __init__(self, model, tokenizer, device: str | None = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @Profile(__name__)
    def predict(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).cpu().tolist()
