"""PyTorch implementation of the BioBERT NER model.

This module mirrors the original TensorFlow version by keeping a BERT
encoder followed by a token classification head, but the underlying
operations are expressed with PyTorch layers.  The conceptual flow of
``input -> BERT embeddings -> dropout -> classification`` therefore
remains unchanged while providing an easy drop‐in replacement.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoConfig, AutoModelForTokenClassification


class BertNERModel(nn.Module):
    """Thin wrapper around ``AutoModelForTokenClassification``.

    The wrapper exposes a minimal ``nn.Module`` style API so that existing
    training and inference code can interact with it just like the former
    TensorFlow estimator.  The model parameters (BERT encoder, dropout and
    classifier) are fully equivalent to the TensorFlow graph; we only swap
    the backend implementation.
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        """Run a forward pass of the model.

        The signature mirrors the original TensorFlow implementation to keep
        the surrounding training code intact.  ``TokenClassifierOutput`` is
        returned so callers can access both the loss and the logits.
        """

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    # ``AutoModelForTokenClassification`` already implements ``save_pretrained``
    # and ``from_pretrained`` methods.  Expose them here to mirror the
    # TensorFlow checkpoint saving/loading utilities.
    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, path: str) -> "BertNERModel":
        base = AutoModelForTokenClassification.from_pretrained(path)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model = base
        return obj
