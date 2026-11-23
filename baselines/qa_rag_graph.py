"""Graph-based RAG baseline (GraphRAG in the paper).

A lightweight graph retriever: nodes are noun-phrase concepts extracted by spaCy,
and edges link concepts that co-occur in a sentence. A query is matched to seed
concepts, expanded to the most strongly connected 1-hop neighbors, and documents
are ranked by the IDF-weighted query concepts they contain plus a small bonus
for neighbor concepts.

Requires: python -m spacy download en_core_web_sm

Example:
    python baselines/qa_rag_graph.py --data legalbench --model gpt-4o-mini --k 5
"""
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import spacy
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import parse_args, read_corpus, run_baseline


class GraphRAG:
    """Concept co-occurrence graph: nodes are noun-chunk concepts, edges link concepts in the same sentence."""

    def __init__(self, docs, n_neighbors=50, neighbor_weight=0.3):
        self.docs = docs
        self.graph = nx.Graph()
        self.n_neighbors = n_neighbors
        self.neighbor_weight = neighbor_weight
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        self.nlp.max_length = 3_000_000
        self._build_graph()

    @staticmethod
    def _clean(chunk):
        toks = [t for t in chunk if not (t.is_stop or t.is_punct or t.like_num or t.pos_ in ("DET", "PRON"))]
        return " ".join(t.text.lower() for t in toks).strip()

    def _sentence_concepts(self, doc):
        sents = defaultdict(set)
        for chunk in doc.noun_chunks:
            c = self._clean(chunk)
            if len(c) > 1:
                sents[chunk.sent.start].add(c)
        return list(sents.values())

    def _extract_concepts(self, text):
        return set().union(*self._sentence_concepts(self.nlp(text)), set())

    def _build_graph(self):
        print("Building concept graph ...")
        self.doc_concepts = []
        for doc in tqdm(self.nlp.pipe(self.docs, batch_size=8), total=len(self.docs)):
            sents = self._sentence_concepts(doc)
            self.doc_concepts.append(set().union(*sents, set()))
            for concepts in sents:
                for a, b in combinations(sorted(concepts), 2):
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                    else:
                        self.graph.add_edge(a, b, weight=1)
            self.graph.add_nodes_from(self.doc_concepts[-1])

        df = Counter(c for cs in self.doc_concepts for c in cs)
        n = len(self.docs)
        self.idf = {c: math.log((n + 1) / (d + 0.5)) for c, d in df.items()}
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def retrieve_subgraph(self, query):
        """Seed nodes = graph concepts containing a query concept (grouped per query concept);
        expand to the most strongly connected 1-hop neighbors."""
        query_concepts = self._extract_concepts(query)
        seeds = {q: {n for n in self.graph.nodes if q in n} for q in query_concepts}
        seeds = {q: ns for q, ns in seeds.items() if ns}
        all_seeds = set().union(*seeds.values(), set())

        neighbor_strength = Counter()
        for s in all_seeds:
            for nb, attr in self.graph[s].items():
                if nb not in all_seeds:
                    neighbor_strength[nb] += attr["weight"] * self.idf.get(s, 0)
        neighbors = {nb for nb, _ in neighbor_strength.most_common(self.n_neighbors)}
        return seeds, neighbors

    def retrieve(self, query, k):
        """Each query concept counts once per doc, weighted by its doc-level IDF; neighbors add a small bonus."""
        seeds, neighbors = self.retrieve_subgraph(query)
        n = len(self.docs)
        hits = {q: [bool(cs & ns) for cs in self.doc_concepts] for q, ns in seeds.items()}
        q_idf = {q: math.log((n + 1) / (sum(h) + 0.5)) for q, h in hits.items()}
        scores = []
        for idx, cs in enumerate(self.doc_concepts):
            s = sum(q_idf[q] for q, h in hits.items() if h[idx])
            s += self.neighbor_weight * sum(self.idf[c] for c in cs & neighbors) / max(len(neighbors), 1)
            scores.append(s)
        return sorted(range(n), key=lambda i: -scores[i])[:k]


def main():
    args = parse_args("Graph-based RAG baseline.").parse_args()
    corpus = read_corpus(args.data)
    retriever = GraphRAG(corpus)
    run_baseline("graphrag", retriever.retrieve, corpus, args)


if __name__ == "__main__":
    main()
