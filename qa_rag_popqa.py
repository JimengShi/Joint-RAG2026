import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import zip_longest
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq
from openai import OpenAI
import pickle
import argparse
from utils.topk_most_freq import topk_most_freq_id
from utils.ablation import ABLATIONS, unranked_topk, load_cells_cache, save_cells_cache
import os
import re
import sys
import time


def _get_data_arg(default="popqa"):
    """Corpus/hypercube paths are loaded at module level (before argparse runs
    in main()), so --data is scanned from argv directly."""
    argv = sys.argv
    if "--data" in argv:
        idx = argv.index("--data")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default


DATA = _get_data_arg()
ABLATION = "none"  # set from --ablation in main()

# from IPython import embed
ent2emb = None
ENT2EMB_PATH = f'ent2emb/ent2emb_{DATA}_latest.pkl'


def load_ent2emb():
    global ent2emb
    if os.path.exists(ENT2EMB_PATH):
        with open(ENT2EMB_PATH, 'rb') as f:
            ent2emb = pickle.load(f)
    else:
        ent2emb = {}
        print("No saved dictionary found, initialized a new dict from string to corresponding embedding.")


def save_ent2emb():
    os.makedirs(os.path.dirname(ENT2EMB_PATH), exist_ok=True)
    with open(ENT2EMB_PATH, 'wb') as f:
        pickle.dump(ent2emb, f)


def embed_strings_batch(strings, emb_model, batch_size=256):
    """Embed any not-yet-cached strings in one batched pass (in-memory only)."""
    missing = [s for s in dict.fromkeys(strings) if s not in ent2emb]
    if not missing:
        return
    embs = emb_model.encode(['query: ' + s for s in missing], normalize_embeddings=True,
                             batch_size=batch_size, show_progress_bar=True)
    for s, e in zip(missing, embs):
        ent2emb[s] = e


def embed_string(target_str, emb_model):
    """Single-entity embed, in-memory cache only (no per-call disk write)."""
    if target_str in ent2emb:
        return
    ent2emb[target_str] = emb_model.encode('query: ' + target_str, normalize_embeddings=True)


# ================ Load Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")


# ================ read corpus ================
# Passage-level corpus (deduplicated), aligned by line index with popqa/<dim>.txt
HC_DIR = f'hypercube/{DATA}'
print('Reading the corpus file ...')
with open(f'{HC_DIR}/passages.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# ================ build embedding index ================
model = SentenceTransformer('intfloat/e5-base-v2')
doc_embeddings = model.encode(['passage: ' + text for text in corpus], normalize_embeddings=True, show_progress_bar=True)


# ================ construct hypercube ================
print('Reading the hypercube files ...')

dimensions = ['date', 'person', 'location', 'organization', 'occupation', 'nationality']
hypercube = {d: defaultdict(list) for d in dimensions}

load_ent2emb()

all_keys = []
dim_lines = {}
for dimension in dimensions:
    with open(f'{HC_DIR}/{dimension}.txt') as f:
        dim_lines[dimension] = f.readlines()

for dimension in dimensions:
    for i, line in tqdm(enumerate(dim_lines[dimension]), total=len(dim_lines[dimension]), desc=f"{dimension} (scan)"):
        tmp = json.loads(line)
        for k in tmp:
            hypercube[dimension][k].append(i)
            all_keys.append(k)

print(f'Embedding {len(set(all_keys))} unique entities (batched)...')
embed_strings_batch(all_keys, model)
save_ent2emb()


def get_docs_from_cells(cells, top_k, dense_scores=None, fill=False):
    """dense_scores: if given, docs tied on hypercube hit count are ordered by query-passage
    similarity instead of arbitrarily. fill: top up to top_k with dense results when the
    cells match fewer than top_k docs."""
    if cells is None:
        cells = {}
    tmp_ids = []
    for k, v in cells.items():
        assert k in hypercube

        for vv in v:
            vv = vv.lower()
            if vv in hypercube[k] and ABLATION != "no_sparse":
                tmp_ids.extend(hypercube[k][vv])
            elif ABLATION != "no_dense":
                vv_emb = model.encode(vv, normalize_embeddings=True)

                for cand in hypercube[k]:
                    if ent2emb[cand] @ vv_emb > 0.9:
                        tmp_ids.extend(hypercube[k][cand])

    if ABLATION == "no_rank":
        doc_ids = unranked_topk(tmp_ids, top_k)
    elif dense_scores is None:
        doc_ids = list(topk_most_freq_id(tmp_ids, top_k)) if tmp_ids else []
    else:
        counts = Counter(tmp_ids)
        doc_ids = sorted(counts, key=lambda d: (-counts[d], -dense_scores[d]))[:top_k]

    if fill and len(doc_ids) < top_k and dense_scores is not None:
        for d in np.argsort(-dense_scores):
            if len(doc_ids) == top_k:
                break
            if d not in doc_ids:
                doc_ids.append(int(d))
    return doc_ids


# ================ one-hop title expansion ================
# Each passage's title names the entity it is about, so an entity extracted from one passage (e.g. occupation
# "politician", location "philadelphia") links to the passage titled with it.
def _norm_title(title):
    # "Brendan Canty (cyclist) - Personal life" -> "brendan canty": drop the section suffix, then the disambiguator
    title = re.sub(r'\s+-(\s.*)?$', '', title)
    return re.sub(r'\s*\([^)]*\)\s*$', '', title).strip().lower()


title_index = defaultdict(list)
for i, text in enumerate(corpus):
    m = re.match(r'Title: (.*?) Content:', text)
    if m:
        title_index[_norm_title(m.group(1))].append(i)

doc_entities = defaultdict(set)
for dimension in dimensions:
    for ent, ids in hypercube[dimension].items():
        for i in ids:
            doc_entities[i].add(ent)


def expand_by_title_hop(seed_ids, dense_scores, top_k, n_seeds=2):
    """Keep the top n_seeds docs, then add docs titled with each seed's entities (any dimension), each seed's
    list ordered by query-passage similarity and the seeds taken in turn, then the remaining seed docs.
    Two seeds cover the common case where a same-name or section passage outranks the subject's own page."""
    seeds = seed_ids[:n_seeds]
    hops = [sorted({t for ent in doc_entities[s] for t in title_index.get(ent, []) if t not in seeds},
                   key=lambda d: -dense_scores[d]) for s in seeds]
    interleaved = [d for group in zip_longest(*hops) for d in group if d is not None]
    return list(dict.fromkeys(seeds + interleaved + seed_ids[n_seeds:]))[:top_k]


# ================ HippoRAG-style one-shot CoT prompt (ported from rag_qa_musique.py) ================
_HIPPORAG_ONE_SHOT_DOCS = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

_HIPPORAG_RAG_QA_SYSTEM = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
    # Token-level F1 is max'd over gold aliases, so listing several answers only dilutes the score; when the
    # passages lack the fact (e.g. a biography without a birthplace) the reader should still commit to an answer.
    ' The answer must be exactly ONE entity: the primary one the subject is notable for, stated as specifically as the passages '
    'support, as a short noun phrase (e.g. "politician", "Philadelphia"). Do not list several alternatives, do not generalize to a '
    'broader category, and do not add qualifiers such as nationality, party, state, or country. If the passages do not contain the '
    'answer, rely on your own knowledge together with any clues in the passages to give your single best answer; never answer '
    '"none" or "unknown", and never leave the answer empty.'
)

_HIPPORAG_ONE_SHOT_INPUT = (
    f"{_HIPPORAG_ONE_SHOT_DOCS}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

_HIPPORAG_ONE_SHOT_OUTPUT = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


def build_hipporag_style_messages(docs: str, query: str) -> list[dict]:
    prompt_user = docs + '\n\nQuestion: ' + query + '\nThought: '
    return [
        {"role": "system", "content": _HIPPORAG_RAG_QA_SYSTEM},
        {"role": "user", "content": _HIPPORAG_ONE_SHOT_INPUT},
        {"role": "assistant", "content": _HIPPORAG_ONE_SHOT_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]


def parse_hipporag_style_answer(response_content: str) -> str:
    try:
        return response_content.split('Answer:')[1].strip()
    except Exception:
        return response_content


# ================ LLMs ================
def llm_answer(model_type, query, cells, k, retrieval_method='hypercube', hc_rank='freq', fill=False, hop=False):
    assert model_type in ['gpt-4o-mini', 'gpt-4o', 'llama3']
    assert retrieval_method in ['hypercube', 'semantic', 'union']

    ### set up retriever searching combination
    # get query embedding (semantic return)
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:, 0]

    # get docs from cells (hypercube return)
    if hop:
        structure_docs = expand_by_title_hop(get_docs_from_cells(cells, max(k, 10), dense_scores=scores), scores, k)
    else:
        structure_docs = get_docs_from_cells(cells, k, dense_scores=scores if hc_rank == 'dense_tiebreak' else None,
                                             fill=fill)

    semantic_docs = [index for _, index in heapq.nlargest(k, ((v, i) for i, v in enumerate(scores)))]

    if retrieval_method == 'hypercube':
        doc_ids = structure_docs
        print(f'>>> Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'semantic':
        doc_ids = semantic_docs
        print(f'>>> Doc ids by semantic retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'union':
        doc_ids = list(set(structure_docs).union(set(semantic_docs)))
        doc_ids = doc_ids[:5]
        print(f'>>> Doc ids by union retrieval: {[id + 1 for id in doc_ids]} \n')

    # Document formatting matches HippoRAG's rag_qa_musique template: "Wikipedia Title: <passage>"
    docs = ''.join([f"Wikipedia Title: {corpus[doc_id]}\n\n" for doc_id in doc_ids])

    messages = build_hipporag_style_messages(docs, query)

    if model_type == 'llama3':
        from together import Together
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            temperature=0,
        )
        res = completion.choices[0].message.content
        return parse_hipporag_style_answer(res), [id + 1 for id in doc_ids]

    elif model_type == 'gpt-4o':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0,
        )
        res = completion.choices[0].message.content
        return parse_hipporag_style_answer(res), [id + 1 for id in doc_ids]

    elif model_type == 'gpt-4o-mini':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0,
        )
        res = completion.choices[0].message.content
        return parse_hipporag_style_answer(res), [id + 1 for id in doc_ids]


from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List,
    Literal
)


_DECOMPOSE_SYSTEM_PROMPT = (
    f"You are an expert on question understanding. "
    f"Your task is to:\n"
    f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
    f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
    f"date dimension can be specific dates, years, decades, centuries, and descriptive time phrases.\n"
    f"person dimension is for names of real people (e.g. the subject of a biography).\n"
    f"location dimension can be geographic places such as countries, cities, regions, birthplaces, places of residence or death.\n"
    f"organization dimension can be named institutions, companies, government bodies, political parties, military units, sports teams, and other formal groups.\n"
    f"occupation dimension can be a person's job, profession, role, or vocation (e.g. 'politician', 'journalist', 'poet').\n"
    f"nationality dimension can be nationality or ethnicity-related descriptors tied to a person or place (e.g. 'Australian', 'Scottish').\n"
    f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
    f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
    f"Example Input:\n"
    f"Question: What is George Rankin's occupation?\n"
    f"Example Output:\n"
    f"Query 1:\n"
    f"query_dimension: 'person'; query_content: 'George Rankin';\n"
)


class Query(BaseModel):
    query_content: str = Field(
        ...,
        title='Entity or phrase to query the documents'
    )
    query_dimension: Literal['date', 'person', 'location', 'organization', 'occupation', 'nationality'] = Field(
        ...,
        title='Dimension of the entity or phrase to query documents'
    )


class AllQueries(BaseModel):
    list_of_queries: List[Query] = Field(
        ...,
        title='List of queries following the required format based on the question comprehension'
    )


def decompose_query_openai(query):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': _DECOMPOSE_SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Question: {query}"}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )
    detected_ents = response.choices[0].message.parsed
    if detected_ents is None:
        return []
    return [(e.query_dimension, e.query_content) for e in detected_ents.list_of_queries]


def decompose_query_together(query, together_model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    from together import Together
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    json_instructions = (
        "\nOutput ONLY a JSON object of the form "
        '{"list_of_queries": [{"query_dimension": "<one of date|person|location|'
        'organization|occupation|nationality>", "query_content": "<entity or short phrase>"}, ...]}. '
        "No markdown, no explanation, just the JSON object."
    )
    completion = client.chat.completions.create(
        model=together_model,
        messages=[
            {'role': 'system', 'content': _DECOMPOSE_SYSTEM_PROMPT + json_instructions},
            {'role': 'user', 'content': f"Question: {query}"}
        ],
        max_tokens=1024,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    parsed = json.loads(raw)
    out = []
    for e in parsed.get("list_of_queries", []):
        dim = e.get("query_dimension")
        content = e.get("query_content")
        if dim in dimensions and content:
            out.append((dim, content))
    return out


def decompose_query(query, model_type="gpt-4o"):
    try:
        if model_type == "llama3":
            pairs = decompose_query_together(query)
        else:
            pairs = decompose_query_openai(query)

        if not pairs:
            return None
        cells = defaultdict(list)
        for dim, content in pairs:
            cells[dim].append(content)
            embed_string(target_str=content, emb_model=model)
        return cells

    except Exception as e:
        print(str(e))
        return None


# ================ Evaluation Metrics ================
from utils.metric import semantic_score, f1_score


def max_over_references(score_fn, predicted: str, gold_answers: list[str]) -> float:
    """HippoRAG's QAF1Score/QAExactMatch aggregate over multiple accepted answers
    using max, rather than scoring against one joined string."""
    if not gold_answers:
        return 0.0
    return max(score_fn(predicted, g) for g in gold_answers)


# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (expects hypercube/<data>/passages.txt and hypercube/<data>/<dim>.txt).")
    parser.add_argument("--qa", type=str, default="popqa", help="QA set name (expects QA/<qa>/<qa>_qa_pairs.json).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4o-mini", "gpt-4o", 'llama3'], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, required=True, default="hypercube", choices=['hypercube', 'semantic', 'union'], help="Retrieval methods.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (for smoke-testing).")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N questions (combine with --limit to run a slice).")
    parser.add_argument("--hc_rank", type=str, default="freq", choices=["freq", "dense_tiebreak"],
                        help="Order of hypercube docs: hit count only, or hit count with ties broken by query similarity.")
    parser.add_argument("--fill", action="store_true", help="Top up hypercube results to k with dense results when cells match fewer than k docs.")
    parser.add_argument("--hop", action="store_true", help="Expand the top hypercube doc with docs titled by its entities (one-hop title expansion).")
    parser.add_argument("--ablation", type=str, default="none", choices=ABLATIONS, help="Retrieval ablation variant.")
    parser.add_argument("--cells_cache", type=str, default=None, help="JSON file caching query decompositions (read if present, updated after the run).")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Whether to save per-sample outputs and scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    global ABLATION
    args = parse_args()
    ABLATION = args.ablation
    cells_cache = load_cells_cache(args.cells_cache)

    data_path = f"QA/{args.qa}/{args.qa}_qa_pairs.json"

    qa_samples = load_dataset(data_path)
    qa_samples = qa_samples[args.start:]
    if args.limit:
        qa_samples = qa_samples[:args.limit]

    llm_output = []
    total_semantic, total_f1 = 0.0, 0.0

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]
        gold_answers = sample.get("answers") or [a.strip() for a in gold_answer.split(',')]

        print(f"Q{i+1}: {question}")

        if question in cells_cache:
            cells = defaultdict(list, cells_cache[question]) if cells_cache[question] else None
        else:
            cells = decompose_query(query=question, model_type=args.model)
            cells_cache[question] = cells
        print(f">>> Identified cells from the query: {cells} \n")

        predicted_answer, return_doc_ids = llm_answer(args.model, question, cells, args.k, args.retrieval_method,
                                                      hc_rank=args.hc_rank, fill=args.fill, hop=args.hop)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answers}")

        f1 = max_over_references(f1_score, predicted_answer, gold_answers)
        sem = max_over_references(semantic_score, predicted_answer, gold_answers)
        print(f'>>> F1: {f1:.4f}, Semantic: {sem:.4f}\n')

        total_f1 += f1
        total_semantic += sem

        llm_output.append({
            "id": sample.get("id", i + 1),
            "question": question,
            "gold_answers": gold_answers,
            "predicted_answer": predicted_answer,
            "f1_score": f1,
            "semantic_score": sem,
            "return_doc_ids": return_doc_ids,
        })

    save_ent2emb()
    save_cells_cache(args.cells_cache, cells_cache)

    num_samples = len(qa_samples)
    avg_f1 = total_f1 / num_samples
    avg_semantic = total_semantic / num_samples
    print(f"Total samples: {num_samples}, average F1: {avg_f1:.4f}, average Semantic: {avg_semantic:.4f}")

    results = {
        "total_samples": num_samples,
        "average_F1": avg_f1,
        "average_Semantic": avg_semantic,
    }

    output_dir = f"output/{args.data}/{args.model}/"
    tag = f"{args.retrieval_method}_k{args.k}"
    if args.start:
        tag += f"_start{args.start}"
    if args.hc_rank != "freq":
        tag += f"_{args.hc_rank}"
    if args.fill:
        tag += "_fill"
    if args.hop:
        tag += "_hop"
    if args.limit:
        tag += f"_first{args.limit}"
    if args.ablation != "none":
        tag += f"_{args.ablation}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.save == "true":
        with open(f"{output_dir}/llm_output_{tag}_responses.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        with open(f"{output_dir}/llm_output_{tag}_scores.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()
