"""NV-Embed-v2 dense retrieval baseline.

Example:
    python baselines/qa_rag_nvembed.py --data legalbench --model gpt-4o-mini --k 5
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import parse_args, read_corpus, run_baseline
from retrievers import NVEmbedRetriever


def main():
    args = parse_args("NV-Embed-v2 baseline.").parse_args()
    corpus = read_corpus(args.data)
    retriever = NVEmbedRetriever(corpus, cache_name=args.data)
    run_baseline("nvembed", retriever.retrieve, corpus, args)


if __name__ == "__main__":
    main()
