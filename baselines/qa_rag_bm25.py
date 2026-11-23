"""BM25 baseline.

Example:
    python baselines/qa_rag_bm25.py --data legalbench --model gpt-4o-mini --k 5
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import parse_args, read_corpus, run_baseline
from retrievers import BM25Retriever


def main():
    args = parse_args("BM25 baseline.").parse_args()
    corpus = read_corpus(args.data)
    retriever = BM25Retriever(corpus)
    run_baseline("bm25", retriever.retrieve, corpus, args)


if __name__ == "__main__":
    main()
