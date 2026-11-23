import json
import os
import random

# Ablation variants of hypercube retrieval (paper Sec. "Ablation Study"):
#   none      - full method
#   no_sparse - drop exact key matching; every query value goes through embedding matching
#   no_dense  - drop embedding matching; only exact key matches count
#   no_rank   - drop ranking by joint constraint coverage; take k matched docs in random order
ABLATIONS = ["none", "no_sparse", "no_dense", "no_rank"]


def unranked_topk(doc_ids, k, seed=0):
    """k distinct matched docs in a seeded random order, ignoring how many constraints each satisfies."""
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    return ids[:k]


def load_cells_cache(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cells_cache(path, cache):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f, indent=1, ensure_ascii=False)
