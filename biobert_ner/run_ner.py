"""Simplified training script for BioBERT NER using PyTorch."""
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .modeling import BertNERModel


def read_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """Read a CoNLL-style NER file."""
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            token, label = line.split()[:2]
            tokens.append(token)
            labels.append(label)
    if tokens:
        sentences.append((tokens, labels))
    return sentences


def encode_examples(examples, tokenizer, label2id, max_length=128):
    encodings = tokenizer(
        [ex[0] for ex in examples],
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    labels = []
    for i, (_, labs) in enumerate(examples):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        label_idx = 0
        previous_word = None
        for j, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word:
                label_ids[j] = label2id[labs[label_idx]]
                label_idx += 1
            previous_word = word_id
        labels.append(label_ids)
    encodings.pop("offset_mapping")
    return encodings, labels


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train(model, dataset, batch_size=8, epochs=3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    examples = read_conll(args.train_file)
    label_list = sorted({l for _, labs in examples for l in labs})
    label2id = {l: i for i, l in enumerate(label_list)}
    encodings, labels = encode_examples(examples, tokenizer, label2id)
    dataset = NERDataset(encodings, labels)

    model = BertNERModel(args.model_name, num_labels=len(label_list))
    train(model, dataset, batch_size=args.batch_size, epochs=args.epochs)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
