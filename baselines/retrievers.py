"""BM25, NV-Embed-v2 and hybrid (reciprocal rank fusion) retrievers."""
import os

import numpy as np


class BM25Retriever:
    """BM25 (rank_bm25.BM25Okapi) over lowercased, whitespace-tokenized documents."""

    def __init__(self, corpus):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([doc.lower().split() for doc in corpus])

    def scores(self, query):
        return self.bm25.get_scores(query.lower().split())

    def ranking(self, query):
        return np.argsort(self.scores(query))[::-1].tolist()

    def retrieve(self, query, k):
        return self.ranking(query)[:k]


class NVEmbedRetriever:
    """Dense retrieval with nvidia/NV-Embed-v2, following the model card: queries
    carry a retrieval instruction, passages have no prefix, and an EOS token is
    appended to every input. Passage embeddings are cached in cache/."""

    QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

    def __init__(self, corpus, cache_name=None, batch_size=4):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

        cache_path = f"cache/nvembed_{cache_name}.npy" if cache_name else None
        if cache_path and os.path.exists(cache_path):
            self.doc_embeddings = np.load(cache_path)
            assert len(self.doc_embeddings) == len(corpus), f"{cache_path} does not match the corpus; delete it."
        else:
            self.doc_embeddings = self.model.encode(self._add_eos(corpus), batch_size=batch_size,
                                                    normalize_embeddings=True, show_progress_bar=True)
            if cache_path:
                os.makedirs("cache", exist_ok=True)
                np.save(cache_path, self.doc_embeddings)

    def _add_eos(self, texts):
        return [t + self.model.tokenizer.eos_token for t in texts]

    def scores(self, query):
        q = self.model.encode(self._add_eos([query]), prompt=self.QUERY_PREFIX, normalize_embeddings=True)
        return (self.doc_embeddings @ q[0]).tolist()

    def ranking(self, query):
        return np.argsort(self.scores(query))[::-1].tolist()

    def retrieve(self, query, k):
        return self.ranking(query)[:k]


class HybridRetriever:
    """BM25 + NV-Embed-v2 fused by reciprocal rank fusion (RRF):
    score(d) = sum over retrievers of 1 / (rrf_k + rank(d)), rank starting at 1."""

    def __init__(self, bm25, dense, rrf_k=60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def retrieve(self, query, k):
        fused = {}
        for ranking in (self.bm25.ranking(query), self.dense.ranking(query)):
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=lambda d: -fused[d])[:k]
