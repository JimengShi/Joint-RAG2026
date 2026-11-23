import os
import json
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq
from openai import OpenAI
import pickle
import argparse
from utils.topk_most_freq import topk_most_freq_id
from together import Together
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor


LOCAL_LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _local_client():
    """Llama-3.3-70B-Instruct served by vLLM (scripts/serve_llama70b.sbatch); same env vars as dimloop."""
    return OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.environ.get("LLM_API_KEY", "EMPTY"))

# from IPython import embed
ent2emb = None


def embed_string(target_str, emb_model, dict_path='ent2emb/ent2emb_hurricane.pkl', ):
    global ent2emb
    if ent2emb is None:
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                ent2emb = pickle.load(f)      
        else:
            ent2emb = {}
            print(f"No saved dictionary found, initialized a new dict from string to corresponding embedding.")

    if target_str in ent2emb:
        return 
    else:
        # in-memory only; save_ent2emb() writes the cache once (rewriting it per string is very slow)
        ent2emb[target_str] = emb_model.encode('query: ' + target_str, normalize_embeddings=True)


def save_ent2emb(dict_path='ent2emb/ent2emb_hurricane.pkl'):
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    with open(dict_path, 'wb') as f:
        pickle.dump(ent2emb, f)


# ================ Load Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")
    

# ================ read corpus ================
print('Reading the corpus file ...')
with open('corpus/hurricane/SciDCC-Hurricane.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# ================ build embedding index ================
model = SentenceTransformer('intfloat/e5-base-v2')
# model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
doc_embeddings = model.encode(['passage: ' + text for text in corpus], normalize_embeddings=True)


# ================ construct hypercube ================
print('Reading the hypercube files ...')

dimensions = ["date", "location", "person", "quantity", "organizations_research_initiatives", "weather_event_topic", "weather_event_type"]
hypercube = {
    'date': defaultdict(list),
    'location': defaultdict(list),
    'person': defaultdict(list),
    'quantity': defaultdict(list),
    'organizations_research_initiatives': defaultdict(list),
    'weather_event_topic': defaultdict(list),
    'weather_event_type': defaultdict(list),
}


for dimension in dimensions:
    with open(f'hypercube/hurricane/{dimension}.txt') as f:
        readin = f.readlines()
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                embed_string(target_str=k, emb_model=model, dict_path='ent2emb/ent2emb_hurricane.pkl', )
save_ent2emb()


def get_docs_from_cells(cells, top_k):
    """
    cells: query extraction
    Each value in cells is now a list of (content, operator) tuples.
    operator is one of: '=', '>', '<', '>=', '<=', 'after', 'before'
    For '=' (no relational constraint), falls back to original exact/embedding lookup.
    For relational operators on 'date' or 'quantity' dimensions, scans all index keys
    and filters by the extracted numeric value — generalizes to any anchor year or amount.
    """
    if cells is None: return []
    tmp_ids = []
    # doc_ids = set(list(range(len(corpus))))

    # helpers to extract numeric anchor from an index key
    def _extract_year(s):
        m = re.search(r'\b(1\d{3}|2\d{3})\b', s)
        return int(m.group()) if m else None

    def _extract_billion(s):
        m = re.search(r'(\d+\.?\d*)\s*billion', s, re.I)
        return float(m.group(1)) if m else None

    RANGE_EXTRACTORS = {
        'date':     _extract_year,
        'quantity': _extract_billion,
    }

    OPS = {
        'after': lambda a, b: a > b,
        'since': lambda a, b: a >= b,
        '>':     lambda a, b: a > b,
        '>=':    lambda a, b: a >= b,
        'before': lambda a, b: a < b,
        'until':  lambda a, b: a <= b,
        '<':     lambda a, b: a < b,
        '<=':    lambda a, b: a <= b,
    }

    # Collect per-dimension doc sets, then vote across dimensions.
    # Each dimension contributes at most 1 vote per doc regardless of how many
    # keys matched — prevents broad range scans from inflating hit counts.
    # OLD: tmp_ids was a flat list; every key-match added entries, so docs
    # indexed under many post-2010 date keys unfairly outranked precise matches.
    dim_doc_sets = []   # (matched_set, is_range) — tracked for mandatory filtering
    # OLD: dim_doc_sets = []  — no is_range tracking

    # Similarity threshold for equality operator embedding lookup.
    # 0.9 is the right value once the "query: " prefix bug is fixed — e.g.
    # "hurricane" now scores 0.9008 against "Hurricane Harvey" (was 0.8648
    # without the prefix), so named storms are correctly matched at 0.9.
    # OLD (before prefix fix): SIM_THRESHOLD = 0.75  — too broad, matched 831/844 docs
    SIM_THRESHOLD = 0.9

    # Mandatory range filter threshold: if a range dimension matches ≤ this many
    # unique docs, it is treated as a hard filter — results must be a subset of it.
    # This ensures selective constraints (e.g. qty > 50B → 40 docs) always gate
    # the final ranking, preventing unrelated high-frequency docs from winning.
    MANDATORY_K = 150

    for k, v in cells.items():
        assert k in hypercube

        for item in v:
            # support both old-style plain strings (operator defaults to '=')
            # and new-style (content, operator) tuples
            if isinstance(item, tuple):
                vv, operator = item
            else:
                vv, operator = item, '='

            vv = vv.lower()
            matched = set()  # one set per query item — at most 1 vote per doc
            is_range = False

            if operator != '=' and k in RANGE_EXTRACTORS:
                # ---- range scan: walk all keys in this dimension ----
                extractor = RANGE_EXTRACTORS[k]
                anchor = extractor(vv)
                cmp_fn = OPS.get(operator)
                if anchor is not None and cmp_fn is not None:
                    for cand_key in hypercube[k]:
                        cand_val = extractor(cand_key)
                        if cand_val is not None and cmp_fn(cand_val, anchor):
                            matched.update(hypercube[k][cand_key])
                is_range = True
                # OLD: tmp_ids.extend(...) — added one entry per key, inflating counts
            else:
                # ---- exact / embedding lookup (operator '=' or non-range dim) ----
                # OLD threshold was 0.9 — too strict, missed "hurricane harvey" for "hurricane"
                # if vv in hypercube[k]:
                #     tmp_ids.extend(hypercube[k][vv])
                # else:
                #     vv_emb = model.encode(vv, normalize_embeddings=True)
                #     for cand in hypercube[k]:
                #         if ent2emb[cand] @ vv_emb > 0.9:
                #             tmp_ids.extend(hypercube[k][cand])
                # OLD: if exact → return exact only; else → embed.
                # BUG: when an exact key exists (e.g. lowercase "hurricane"), the
                # embedding path never ran, so docs indexed under related variants
                # ("Hurricane Harvey", "hurricanes") were silently excluded.
                # FIX: always run embedding; exact match just seeds the set faster.
                # if vv in hypercube[k]:
                #     matched.update(hypercube[k][vv])
                # else:
                #     vv_emb = model.encode(vv, normalize_embeddings=True)
                #     for cand in hypercube[k]:
                #         if ent2emb[cand] @ vv_emb > 0.9:
                #             matched.update(hypercube[k][cand])
                if vv in hypercube[k]:
                    matched.update(hypercube[k][vv])
                vv_emb = model.encode('query: ' + vv, normalize_embeddings=True)
                for cand in hypercube[k]:
                    if ent2emb[cand] @ vv_emb > SIM_THRESHOLD:
                        matched.update(hypercube[k][cand])

            if matched:
                dim_doc_sets.append((k, matched, is_range))
                # OLD: dim_doc_sets.append((matched, is_range))

    # Mandatory filter: find the most selective range set (fewest docs).
    # If it has ≤ MANDATORY_K docs, intersect every other set with it so that
    # only docs satisfying the hard constraint can receive votes.
    range_sets = [s for _, s, is_range in dim_doc_sets if is_range]
    if range_sets:
        most_selective = min(range_sets, key=len)
        if len(most_selective) <= MANDATORY_K:
            dim_doc_sets = [(k, s & most_selective, is_range) for k, s, is_range in dim_doc_sets]

    # Vote: 1 point per DIMENSION TYPE (not per query item).
    # Multiple queries for the same dimension (e.g. location='Texas', location='U.S.',
    # location='Puerto Rico') are unioned before voting — a doc matching any of
    # them gets exactly 1 location vote, not 3. Without this, overview articles
    # that mention all US states + all named storms accumulate many more votes
    # than focused documents that answer the question directly.
    # OLD: each query item contributed 1 vote independently:
    # for doc_set, _ in dim_doc_sets:
    #     tmp_ids.extend(doc_set)
    from itertools import groupby
    # group by dimension key, union their matched sets, cast 1 vote per dim
    dim_doc_sets.sort(key=lambda x: x[0])  # sort by dimension key for groupby
    for dim_key, group in groupby(dim_doc_sets, key=lambda x: x[0]):
        union_set = set()
        for _, s, _ in group:
            union_set |= s
        tmp_ids.extend(union_set)  # 1 vote per doc per dimension type

    doc_ids = topk_most_freq_id(tmp_ids, top_k)

    # print(">>> doc_ids:", doc_ids)
    # doc_ids = doc_ids.intersection(set(tmp_ids))
    return list(doc_ids)


# ================ LLMs ================
def llm_answer(model_type, query, cells, k=3, retrieval_method='hypercube', fill=False):
    assert model_type in ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'llama3', 'llama3-local']
    assert retrieval_method in ['hypercube', 'semantic', 'union']


    ### set up retriever searching combination
    # get query embedding (semantic return) — computed once, used by all paths
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:,0]
    semantic_docs = [index for _, index in heapq.nlargest(k, ((v, i) for i, v in enumerate(scores)))]

    # get docs from cells (hypercube return)
    # Fetch 3k candidates so the semantic tiebreaker has enough headroom to
    # re-rank tied-vote docs (e.g. the drought doc at 4 votes vs Harvey at 4 votes).
    t1 = time.time()
    structure_docs_wide = get_docs_from_cells(cells, k * 3)
    

    if structure_docs_wide:
        structure_docs = sorted(structure_docs_wide, key=lambda doc_id: scores[doc_id], reverse=True)[:k]
    else:
        structure_docs = structure_docs_wide

    if retrieval_method == 'hypercube':
        doc_ids = list(structure_docs)
        if fill:
            # top up with dense (e5) results when the cells match fewer than k docs, as in qa_rag_nq.py --fill
            for d in np.argsort(-scores):
                if len(doc_ids) >= k:
                    break
                if d not in doc_ids:
                    doc_ids.append(int(d))
        print(f'>>> Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'semantic':
        doc_ids = semantic_docs
        print(f'>>> Doc ids by semantic retrieval: {[id + 1 for id in doc_ids]} \n')
    elif retrieval_method == 'union':
        doc_ids = list(set(structure_docs).union(set(semantic_docs)))
        doc_ids = doc_ids[:5]
        print(f'>>> Doc ids by union retrieval: {[id + 1 for id in doc_ids]} \n') 
    
    # limit retrieved documents of hypercube as input to llm
    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])


    # set up instruction
    instruction = 'Answer the query based on the given retrieved documents. ' \
    'If the query asks the quantitative analysis or results, such as starting with "How many", "How much", "How much greater", "How wide", "What percentage", "How far", "How long", "How old", "What portion", "What depth", "What is the sum", you must directly output the quantitative answer without any descriptive explanation or other information.' \
    'If the query starting with "what percentage" or the query includes the word "likelihood", please directly output the number with %. ' \
    'If the query starting with "what specific", "which years", "who", please directly output the answer without explanation or other information.' \
    'Otherwise, please provide as precise information as possible based the retrieved documents. ' \
    'Documents:\n'

    # set up the LLM
    if model_type == 'llama3-local':
        completion = _local_client().chat.completions.create(
            model=LOCAL_LLAMA_MODEL,
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ],
            max_tokens=2048,
        )
        res = completion.choices[0].message.content
        return res, [id + 1 for id in doc_ids]

    elif model_type == 'llama3':
        os.environ.setdefault("TOGETHER_API_KEY", "your TOGETHER_API_KEY")  # or export TOGETHER_API_KEY
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ]
        )
        res = completion.choices[0].message.content  
        return res, [id + 1 for id in doc_ids]
    
    elif model_type in ('gpt-4o', 'gpt-4o-mini'):
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06" if model_type == 'gpt-4o' else "gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": instruction + docs},
                {
                    "role": "user",
                    "content": 'Query: ' + query + '\nAnswer:'
                }
            ]
        )
        res = completion.choices[0].message.content
        return res, [id + 1 for id in doc_ids]


from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List, 
    Literal
)


def decompose_query(query, extraction_model='gpt-4o'):
    """
    Entity extraction from query using a configurable LLM.

    extraction_model choices:
      'gpt-4o'      – OpenAI gpt-4o-2024-08-06, structured-output (.parse)
      'gpt-4o-mini' – OpenAI gpt-4o-mini-2024-07-18, structured-output (.parse)
      'llama3'      – Together Llama-3.3-70B-Instruct-Turbo, JSON-mode
      'llama3-local' – local vLLM Llama-3.3-70B-Instruct, structured-output (.parse)
    """
    if extraction_model == 'llama3-local':
        client, decompose_model = _local_client(), LOCAL_LLAMA_MODEL
    else:
        client, decompose_model = OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o-2024-08-06"
    system_prompt = (
        f"You are an expert on question understanding. "
        f"Your task is to:\n"
        f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
        f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
        f"date dimension can be specific time descriptions on weather events, such as period, duration, hour, minute, day, week, month, year, also include the descriptive phrases/terms, e.g., six months, half hour, one decade. \n"
        f"location dimension can be specific geographic locations, nationality, city, state, province, zipcodes etc. \n"
        f"person dimension can be specific names of people and the descriptive role/occupration phrases/terms such as reporter, researcher, meteorologist etc. \n"
        f"quantity dimension can be related quantitative numbers and descriptions on weather events, the descriptive phrases/terms such as such as increase, decrease, decline etc. \n"
        f"organizations_research_initiatives dimension can be names of organization/university/initiative/project/journal on weather research. \n"
        f"weather_event_topic dimension can be specific weather/climate/disaster events and topics, such as hurricane, hurricane path, storm, El Niño, flood, etc. \n"
        f"weather_event_type dimension can be weather_event_type: rain, wind, snow, temperature, etc. \n"
        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
        f"For relational constraints (e.g. 'after 2010', 'before 1990', '> 50 billion', 'since 2005'):\n"
        f"  - Set query_content to the bare anchor value only (e.g. '2010', '1990', '50 billion', '2005').\n"
        f"  - Set query_operator to the corresponding operator: 'after'/'since' for date lower bounds, 'before'/'until' for date upper bounds, '>'/'>='/'<'/'<=' for quantity thresholds.\n"
        f"  - Do NOT embed the operator word into query_content (e.g. never write 'after 2010' or '> 50 billion' as the content).\n"
        f"  - For dimensions without a relational constraint, leave query_operator as '='.\n\n"
        f"For weather_event_topic: always emit BOTH a generic term (e.g. 'hurricane') AND any specific named events "
        f"that are likely candidates for answering the question (e.g. 'Hurricane Harvey', 'Hurricane Sandy', 'Hurricane Maria'). "
        f"Specific named-event queries greatly improve retrieval precision because each named storm indexes only a small subset of documents.\n\n"
        f"For location: emit multiple plausible geographic variants "
        f"(e.g. 'U.S.', 'United States', 'Gulf Coast', 'Texas', 'Florida', 'Puerto Rico') "
        f"so that documents using any of these terms are retrieved.\n\n"
        # OLD example (no operator field, no named storms, no location variants):
        # f"Example Input:\n"
        # f"Question: How Indian Monsoons Influence Atlantic Hurricane Paths, any rainfall caused?\n"
        # f"Example Output:\n"
        # f"query_dimension: 'location'; query_content: 'Atlantic';\n"
        # f"query_dimension: 'weather_event_topic'; query_content: 'Indian Monsoons';\n"
        # f"query_dimension: 'weather_event_type'; query_content: 'rain';\n"
        f"Example Input:\n"
        f"Question: which hurricanes caused > 50 billion USD damage in the U.S. after 2010?\n"
        f"Example Output:\n"
        f"Query 1:  query_dimension: 'date';                query_content: '2010';             query_operator: 'after';\n"
        f"Query 2:  query_dimension: 'quantity';            query_content: '50 billion';        query_operator: '>';\n"
        f"Query 3:  query_dimension: 'location';            query_content: 'U.S.';              query_operator: '=';\n"
        f"Query 4:  query_dimension: 'location';            query_content: 'United States';     query_operator: '=';\n"
        f"Query 5:  query_dimension: 'location';            query_content: 'Gulf Coast';        query_operator: '=';\n"
        f"Query 6:  query_dimension: 'location';            query_content: 'Texas';             query_operator: '=';\n"
        f"Query 7:  query_dimension: 'location';            query_content: 'Puerto Rico';       query_operator: '=';\n"
        f"Query 8:  query_dimension: 'weather_event_topic'; query_content: 'hurricane';         query_operator: '=';\n"
        f"Query 9:  query_dimension: 'weather_event_topic'; query_content: 'Hurricane Harvey';  query_operator: '=';\n"
        f"Query 10: query_dimension: 'weather_event_topic'; query_content: 'Hurricane Sandy';   query_operator: '=';\n"
        f"Query 11: query_dimension: 'weather_event_topic'; query_content: 'Hurricane Maria';   query_operator: '=';\n"
        f"Query 12: query_dimension: 'weather_event_topic'; query_content: 'Hurricane Irma';    query_operator: '=';\n"
    )
    
    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents — strip any relational operator, keep only the anchor value (e.g. "2010" not "after 2010", "50 billion" not "> 50 billion")'
        )

        # NEW: relational operator extracted from the question; '=' means no constraint
        query_operator: Literal["=", ">", "<", ">=", "<=", "after", "before", "since", "until"] = Field(
            default="=",
            title='Relational operator from the question for this dimension. Use "after"/"before" for date ranges, ">/"<" for quantities, "=" when no comparison is expressed.'
        )

        query_dimension: Literal["date", "location", "person", "quantity", "organizations_research_initiatives", "weather_event_topic", "weather_event_type"] = Field(
            ...,
            title='Dimension of the entity or phrase to query documents'
        )

    # OLD schema (no operator field):
    # class Query(BaseModel):
    #     query_content: str = Field(...)
    #     query_dimension: Literal[...] = Field(...)

    class AllQueries(BaseModel):
        list_of_queries: List[Query] = Field(
            ...,
            title='List of queries following the required format based on the question comprehension'
        )
        
    response = client.beta.chat.completions.parse(
        model=decompose_model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_prompt}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )


    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            # NEW: store (content, operator) tuple so get_docs_from_cells can range-scan
            cells[ent.query_dimension].append((ent.query_content, ent.query_operator))
            # OLD: cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hurricane.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None



# ================ Evaluation Metrics ================
from utils.metric import bleu_score, semantic_score, f1_score
from utils.metric import precision_at_k, recall_at_k

# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, default=None, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--question", type=str, default=None, help="Run a single ad-hoc question directly.")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=["gpt-4", "gpt-4o", "gpt-4o-mini", 'llama3', 'llama3-local'], help="Select llm to get answer.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--qa_file", type=str, default="synthetic_qa.json",
                        help="QA file under QA/<data>/ (default: the 303-question SciDCC-Hurricane set).")
    parser.add_argument("--fill", action="store_true",
                        help="Top up hypercube results to k with dense (e5) results when the cells match fewer than k docs.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Answer this many questions concurrently (e.g. against a local vLLM server); results keep question order.")
    parser.add_argument("--retrieval_method", type=str, default="hypercube", choices=['hypercube', 'semantic', 'union'], help="Retrieval methods.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents.")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()


def decompose_model_for(model_type):
    # local Llama decomposes its own queries; every other reader keeps the original gpt-4o decomposition
    return 'llama3-local' if model_type == 'llama3-local' else 'gpt-4o'


# ================ Main Function ================
def main():
    args = parse_args()

    if args.question:
        print(f"Q: {args.question}")
        cells = decompose_query(query=args.question, extraction_model=decompose_model_for(args.model))
        print(f">>> Identified cells: {cells}\n")
        predicted_answer, return_doc_ids = llm_answer(args.model, args.question, cells, args.k, args.retrieval_method)
        print(f">>> Answer: {predicted_answer}")
        print(f">>> Retrieved doc ids: {return_doc_ids}")
        return

    if args.data is None:
        raise ValueError("Provide either --data or --question.")

    data_path = f"QA/{args.data}/{args.qa_file}"

    # Load dataset
    qa_samples = load_dataset(data_path)
    if args.limit:
        qa_samples = qa_samples[:args.limit]

    # Initialize evaluation metrics
    llm_output = []
    total_bleu, total_semantic, total_f1 = 0, 0, 0
    total_precison, total_recall = 0.0, 0.0
    num_with_gold_docs = 0

    def run_sample(sample):
        cells = decompose_query(query=sample["question"], extraction_model=decompose_model_for(args.model))
        return cells, llm_answer(args.model, sample["question"], cells, args.k, args.retrieval_method, fill=args.fill)

    pool = ThreadPoolExecutor(args.workers) if args.workers > 1 else None
    # map() yields in question order, so scoring and output order match the serial run
    concurrent_results = pool.map(run_sample, qa_samples) if pool else None

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answers"]
        # the older 303-question set has no supporting facts: skip precision/recall for it
        relevant_docs = sample.get("supporting_facts_index")

        print(f"Q{i+1}: {question}")
        cells, (predicted_answer, return_doc_ids) = next(concurrent_results) if pool else run_sample(sample)
        print(f">>> Identified cells from the query: {cells} \n")
        
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f">>> relevant_docs: {relevant_docs}")
        print(f">>> return_doc_ids: {return_doc_ids}")
        print(f'F1 score: {f1_score(predicted_answer, gold_answer):.4f}')
        print(f'Semantic_score: {semantic_score(predicted_answer, gold_answer):.4f}')
        print(f"\n")

        # total_bleu += bleu_score(predicted_answer, gold_answer)
        total_semantic += semantic_score(predicted_answer, gold_answer)
        total_f1 += f1_score(predicted_answer, gold_answer)

        if len(return_doc_ids) == 0:
            return_doc_ids = [0]
        if relevant_docs is not None:
            num_with_gold_docs += 1
            total_precison += precision_at_k(return_doc_ids, relevant_docs)
            total_recall += recall_at_k(return_doc_ids, relevant_docs)

        # Save output for each sample
        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "f1_score": f1_score(predicted_answer, gold_answer),
            "semantic_score": semantic_score(predicted_answer, gold_answer),
            "return_doc_ids": return_doc_ids,
            "relevant_docs": relevant_docs,
            "precision": precision_at_k(return_doc_ids, relevant_docs) if relevant_docs is not None else None,
            "recall": recall_at_k(return_doc_ids, relevant_docs) if relevant_docs is not None else None
        })

    if pool:
        pool.shutdown()
    save_ent2emb()

    # Averaging Summary Information
    num_samples = len(qa_samples)
    num_with_gold_docs = max(num_with_gold_docs, 1)
    # print(f"Total samples: {num_samples}, average Semantic: {total_semantic / num_samples:.4f}")
    print(f"Total samples: {num_samples}, average F1: {total_f1 / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}, average precision: {total_precison / num_with_gold_docs:.4f}, average reall: {total_recall / num_with_gold_docs:.4f}")


    # Assume you have these variables already defined
    results = {
        "total_samples": num_samples,
        # "average_BLEU": round(total_bleu / num_samples, 4),
        "average_F1": total_f1 / num_samples,
        "average_Semantic": total_semantic / num_samples, 
        "average_Precision": total_precison / num_with_gold_docs,
        "average_Recall": total_recall / num_with_gold_docs
    }


    output_dir = f"output/{args.data}/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # the default QA file keeps the plain name read by evaluation/; other QA files are tagged by name
    suffix = "" if args.qa_file == "synthetic_qa.json" else "_" + os.path.splitext(args.qa_file)[0]
    if args.fill:
        suffix += "_fill"
    if args.save == "true":
        with open(f"{output_dir}/llm_output_{args.retrieval_method}_k{args.k}_responses{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=4, ensure_ascii=False)

        with open(f"{output_dir}/llm_output_{args.retrieval_method}_k{args.k}_scores{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("Saved llm output!")

if __name__ == "__main__":
    main()

