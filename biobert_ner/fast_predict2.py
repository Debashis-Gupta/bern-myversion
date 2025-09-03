"""Fast prediction helper for PyTorch models.

The original TensorFlow version kept an ``Estimator`` graph open to avoid the
overhead of rebuilding it for every prediction call.  The same concept is
retained here by keeping the PyTorch model on the target device in evaluation
mode and batching incoming texts.
"""

from typing import List
import torch
from biobert_ner.utils import Profile


class FastPredict:
    """Wrap a PyTorch model to provide a ``predict`` API similar to TensorFlow."""

    def __init__(self, model, tokenizer, device: str | None = None):
        # Ensure the model stays on the device and in eval mode once, mirroring
        # the persistent graph behaviour of the TensorFlow utility.
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @Profile(__name__)
    def predict(self, texts: List[str]):
        """Tokenise ``texts`` and return argmax label IDs for each token."""

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).cpu().tolist()
