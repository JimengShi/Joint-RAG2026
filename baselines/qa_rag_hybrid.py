"""BM25 + NV-Embed-v2 hybrid baseline: the two rankings are fused by
reciprocal rank fusion (RRF) and the top-k fused documents are used.

Example:
    python baselines/qa_rag_hybrid.py --data legalbench --model gpt-4o-mini --k 5
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import parse_args, read_corpus, run_baseline
from retrievers import BM25Retriever, NVEmbedRetriever, HybridRetriever


def main():
    parser = parse_args("BM25 + NV-Embed-v2 hybrid (RRF) baseline.")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant: score = sum 1 / (rrf_k + rank).")
    args = parser.parse_args()
    corpus = read_corpus(args.data)
    retriever = HybridRetriever(BM25Retriever(corpus), NVEmbedRetriever(corpus, cache_name=args.data), rrf_k=args.rrf_k)
    run_baseline("hybrid", retriever.retrieve, corpus, args)


if __name__ == "__main__":
    main()
