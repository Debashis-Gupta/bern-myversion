import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel

class BertNERModel(nn.Module):
    """PyTorch BioBERT model for token classification.

    This wrapper uses ``transformers.AutoModel`` to load a pretrained
    BERT-style encoder and adds a token classification head on top. It
    provides an interface comparable to the former TensorFlow implementation
    while leveraging PyTorch modules.
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """Run a forward pass of the model.

        Args:
            input_ids: ``torch.LongTensor`` of shape ``(batch, seq_len)``.
            attention_mask: optional attention mask of the same shape.
            token_type_ids: optional segment IDs.
            labels: optional ``torch.LongTensor`` of shape ``(batch, seq_len)``
                containing label IDs for computing the loss.

        Returns:
            ``TokenClassifierOutput`` containing ``loss`` (if labels are
            provided) and ``logits`` of shape ``(batch, seq_len, num_labels)``.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)
