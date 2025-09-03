#!/usr/bin/env python3
"""Convert an original TensorFlow checkpoint to Hugging Face PyTorch format.

This small helper wraps the official transformers conversion utility so the
pretrained weights used by the original TensorFlow code can be loaded by the
PyTorch implementation.
"""

import argparse
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import (
    convert_tf_checkpoint_to_pytorch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TF checkpoint to PyTorch"
    )
    parser.add_argument(
        "--tf_checkpoint_path",
        required=True,
        help="Prefix to TensorFlow checkpoint (without .index/.data)",
    )
    parser.add_argument(
        "--bert_config_file",
        default="biobert_ner/conf/bert_config.json",
        help="BERT config file",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        required=True,
        help="Output path for the converted PyTorch model",
    )
    args = parser.parse_args()

    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path
    )


if __name__ == "__main__":
    main()

