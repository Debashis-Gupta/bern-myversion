#!/usr/bin/env python3
"""Standalone BERN-style inference script using PyTorch.

Given one or more PubMed IDs, this script fetches the article text,
loads a pretrained BioBERT NER model (converted to PyTorch weights),
extracts entities and finally runs the BERN normalizer.  The conceptual
flow mirrors the original TensorFlow pipeline but relies entirely on
PyTorch and the Hugging Face ``transformers`` library.

Example::

    python pmid_bern_infer.py --model path/to/pytorch_model --pmids 35916466
"""

import argparse
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from biobert_ner.modeling import BertNERModel
from convert import pubtator_biocxml2dict_list
from normalize import Normalizer


def fetch_docs(pmids: List[str]) -> List[Dict]:
    """Return document dictionaries for each PMID."""
    return pubtator_biocxml2dict_list(pmids)


def collect_entities(tokens, labels, offsets, text, id2label):
    """Convert BIO labels into span dictionaries."""
    entities: Dict[str, List[Dict]] = {}
    cur_type = None
    cur_ent = None
    for tok, lid, (start, end) in zip(tokens, labels, offsets):
        if tok in ("[CLS]", "[SEP]") or start == end:
            cur_type = None
            cur_ent = None
            continue
        label = id2label.get(lid, "O")
        if label.startswith("B-"):
            cur_type = label[2:].lower()
            mention = text[start:end]
            cur_ent = {"start": start, "end": end, "mention": mention}
            entities.setdefault(cur_type, []).append(cur_ent)
        elif label.startswith("I-") and cur_ent and cur_type == label[2:].lower():
            cur_ent["end"] = end
            cur_ent["mention"] = text[cur_ent["start"]:end]
        else:
            cur_type = None
            cur_ent = None
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full BERN on PubMed IDs")
    parser.add_argument("--model", required=True, help="Directory with PyTorch weights")
    parser.add_argument("--pmids", nargs="+", help="One or more PubMed IDs")
    parser.add_argument("--device", default="cpu", help="torch device e.g. cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("biobert_ner/conf")
    model = BertNERModel.from_pretrained(args.model)

    docs = fetch_docs(args.pmids)
    texts = [f"{d['title']} {d['abstract']}".strip() for d in docs]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    return_offsets_mapping=True)
    tokens_batch = [tokenizer.convert_ids_to_tokens(ids)
                    for ids in enc["input_ids"]]
    offset_batch = enc.pop("offset_mapping").tolist()
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    id2label = model.model.config.id2label

    for doc, tokens, labels, offsets in zip(docs, tokens_batch, pred_ids, offset_batch):
        text = f"{doc['title']} {doc['abstract']}".strip()
        entities = collect_entities(tokens, labels, offsets, text, id2label)
        for t, lst in entities.items():
            doc['entities'][t] = lst

    normalizer = Normalizer()
    normalized = normalizer.normalize('pmid', docs, 'main', False)

    for doc in normalized:
        print(f"PMID {doc['pmid']}")
        for ent_type, locs in doc['entities'].items():
            for loc in locs:
                oid = loc.get('id', '')
                print(f"  {ent_type}: {loc['mention']}\t{oid}")
        print()


if __name__ == "__main__":
    main()
