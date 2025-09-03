#!/usr/bin/env python3
"""Standalone script to run BioBERT NER on PubMed abstracts by PMID."""
import argparse
import json

from convert import pubtator_biocxml2dict_list
from biobert_ner.run_ner import BioBERT, FLAGS


def main():
    parser = argparse.ArgumentParser(description="Run BioBERT NER on PMIDs")
    parser.add_argument("--pmids", nargs="+", required=True,
                        help="Space separated list of PMIDs to annotate")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing entity-specific checkpoints")
    parser.add_argument("--bert_config_file", required=True,
                        help="Path to bert_config.json")
    parser.add_argument("--vocab_file", required=True,
                        help="Path to vocab.txt")
    parser.add_argument("--init_checkpoint", required=True,
                        help="Checkpoint prefix, e.g., model.ckpt-1000000")
    args = parser.parse_args()

    # Initialize absl.FLAGS without parsing command line flags from absl
    FLAGS(["pmid_inference"])
    FLAGS.model_dir = args.model_dir
    FLAGS.bert_config_file = args.bert_config_file
    FLAGS.vocab_file = args.vocab_file
    FLAGS.init_checkpoint = args.init_checkpoint

    biobert = BioBERT(FLAGS)
    pmid_list = [int(p) for p in args.pmids]
    docs = pubtator_biocxml2dict_list(pmid_list)
    results = biobert.recognize(docs)
    biobert.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
