import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq
from openai import OpenAI
import pickle
import argparse
from utils.topk_most_freq import topk_most_freq_id
from utils.ablation import ABLATIONS, unranked_topk
import os
import re
import sys
import time


def _get_data_arg(default="nq"):
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
# Passage-level corpus (deduplicated), aligned by line index with hypercube/nq/<dim>.txt
print('Reading the corpus file ...')
with open(f'hypercube/{DATA}/passages.txt') as f:
    readin = f.readlines()
    corpus = [line.strip() for line in tqdm(readin)]


# ================ build embedding index ================
model = SentenceTransformer('intfloat/e5-base-v2')
doc_embeddings = model.encode(['passage: ' + text for text in corpus], normalize_embeddings=True, show_progress_bar=True)


# ================ construct hypercube ================
print('Reading the hypercube files ...')

dimensions = ['date', 'person', 'fictional_character', 'work_title', 'location', 'organization', 'event', 'concept']
hypercube = {d: defaultdict(list) for d in dimensions}

load_ent2emb()

all_keys = []
dim_lines = {}
for dimension in dimensions:
    with open(f'hypercube/{DATA}/{dimension}.txt') as f:
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


# ================ title dimension ================
# Entities are extracted from passage content only, so a passage often lacks its own subject
# (the Wikipedia title) as a key. Index the title, and the title without its "(disambiguation)"
# suffix, as document-level metadata that every query value is matched against.
title_index = defaultdict(set)
for i, text in enumerate(corpus):
    m = re.match(r'Title: (.*?) Content:', text)
    if not m:
        continue
    t = m.group(1).strip().lower()
    title_index[t].add(i)
    title_index[re.sub(r'\s*\(.*\)$', '', t)].add(i)
title_keys = list(title_index)
title_key_embs = model.encode(['query: ' + t for t in title_keys], normalize_embeddings=True, batch_size=256)


def get_title_docs(value, threshold=0.9):
    value = value.lower()
    if value in title_index:
        return title_index[value]
    sims = title_key_embs @ model.encode('query: ' + value, normalize_embeddings=True)
    docs = set()
    for j in np.where(sims > threshold)[0]:
        docs |= title_index[title_keys[j]]
    return docs


def get_docs_from_cells(cells, top_k, dense_scores=None, fill=False, use_title=False):
    """dense_scores: if given, docs tied on hypercube hit count are ordered by query-passage
    similarity instead of arbitrarily. fill: top up to top_k with dense results when the
    cells match fewer than top_k docs. use_title: each query value also votes once for the
    passages whose title it matches."""
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

            if use_title:
                tmp_ids.extend(get_title_docs(vv))

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


def hipporag_passage(passage: str) -> str:
    """'Title: X Content: Y' -> 'X\nY', the passage string HippoRAG 2 feeds its reader."""
    m = re.match(r'Title: (.*?) Content: (.*)$', passage, re.S)
    return f"{m.group(1)}\n{m.group(2)}" if m else passage


def build_hipporag_style_messages(docs: str, query: str) -> list[dict]:
    prompt_user = docs + '\n\nQuestion: ' + query + '\nThought: '
    return [
        {"role": "system", "content": _HIPPORAG_RAG_QA_SYSTEM},
        {"role": "user", "content": _HIPPORAG_ONE_SHOT_INPUT},
        {"role": "assistant", "content": _HIPPORAG_ONE_SHOT_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]


# Appended to HippoRAG 2's system prompt for qa_format=hipporag_norefusal.
_NO_REFUSAL_RULE = (
    ' Always give a concrete answer. If the passages do not state the answer explicitly, infer the most likely answer '
    'from the passages together with your own knowledge. Never answer that the information is not provided, '
    'not specified, or unknown.'
)


# qa_format=hipporag_norefusal2: a stricter rule, plus a second demonstration where the passages do
# not state the answer and the reader still commits to one using its own knowledge.
_NO_REFUSAL_RULE_V2 = (
    ' Always commit to a concrete answer. If the passages do not state the answer explicitly, infer the most likely '
    'answer from the passages together with your own knowledge. Never say that the information is not provided, '
    'not specified, not mentioned, or unknown, and do not hedge. The text after "Answer: " must be only the answer '
    'itself: an entity, date, number, or short phrase.'
)

_IMPLICIT_SHOT_INPUT = (
    f"{_HIPPORAG_ONE_SHOT_DOCS}"
    "\n\nQuestion: "
    "In which county is the University of Southampton located?"
    '\nThought: '
)

_IMPLICIT_SHOT_OUTPUT = (
    "The passages describe the University of Southampton but do not name its county. From general knowledge, "
    "Southampton is a city in Hampshire, England, so the university is located in Hampshire. "
    "\nAnswer: Hampshire."
)

_FALLBACK_REQUEST = (
    'You did not commit to an answer. Give your single best guess using the passages and your own knowledge. '
    'Reply in the same format, ending with "Answer: " followed by only the answer.'
)

_REFUSAL_PATTERN = re.compile(
    r"not (mentioned|specified|provided|available|stated|given|enough|clear)|no (information|mention)|insufficient|"
    r"unknown|does not (specify|provide|mention|say|state)|cannot be determined|none (specified|mentioned)", re.I)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(answer))


def build_hipporag2_messages(docs: str, query: str, no_refusal: bool = False, version: int = 1) -> list[dict]:
    """Exactly what HippoRAG 2 sends: its prompt_user has no blank lines between the passages and
    "Question:", and its template is rendered with string.Template, which turns "$$" into "$"."""
    prompt_user = docs + 'Question: ' + query + '\nThought: '
    if no_refusal and version == 2:
        return [
            {"role": "system", "content": _HIPPORAG_RAG_QA_SYSTEM + _NO_REFUSAL_RULE_V2},
            {"role": "user", "content": _HIPPORAG_ONE_SHOT_INPUT.replace('$$', '$')},
            {"role": "assistant", "content": _HIPPORAG_ONE_SHOT_OUTPUT},
            {"role": "user", "content": _IMPLICIT_SHOT_INPUT.replace('$$', '$')},
            {"role": "assistant", "content": _IMPLICIT_SHOT_OUTPUT},
            {"role": "user", "content": prompt_user},
        ]
    return [
        {"role": "system", "content": _HIPPORAG_RAG_QA_SYSTEM + (_NO_REFUSAL_RULE if no_refusal else '')},
        {"role": "user", "content": _HIPPORAG_ONE_SHOT_INPUT.replace('$$', '$')},
        {"role": "assistant", "content": _HIPPORAG_ONE_SHOT_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]


def parse_hipporag_style_answer(response_content: str) -> str:
    try:
        return response_content.split('Answer:')[1].strip()
    except Exception:
        return response_content


# ================ LLMs ================
def llm_answer(model_type, query, cells, k, retrieval_method='hypercube', hc_rank='freq', fill=False, use_title=False,
               qa_format='default'):
    assert model_type in ['gpt-4o-mini', 'gpt-4o', 'llama3']
    assert retrieval_method in ['hypercube', 'semantic', 'union']

    ### set up retriever searching combination
    # get query embedding (semantic return)
    query_embedding = model.encode(['query: ' + query])
    scores = np.matmul(doc_embeddings, query_embedding.transpose())[:, 0]

    # get docs from cells (hypercube return)
    structure_docs = get_docs_from_cells(cells, k, dense_scores=scores if hc_rank == 'dense_tiebreak' else None,
                                         fill=fill, use_title=use_title)

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
    if qa_format.startswith('hipporag'):
        # HippoRAG 2 passes "title\ntext" passages and decodes with temperature 0
        docs = ''.join([f"Wikipedia Title: {hipporag_passage(corpus[doc_id])}\n\n" for doc_id in doc_ids])
        sampling = {"temperature": 0}
    else:
        docs = ''.join([f"Wikipedia Title: {corpus[doc_id]}\n\n" for doc_id in doc_ids])
        sampling = {}

    if qa_format.startswith('hipporag'):
        messages = build_hipporag2_messages(docs, query, no_refusal=qa_format != 'hipporag',
                                            version=2 if qa_format == 'hipporag_norefusal2' else 1)
    else:
        messages = build_hipporag_style_messages(docs, query)

    res = _chat(model_type, messages, sampling)
    answer = parse_hipporag_style_answer(res)
    info = {"fallback": False, "first_answer": answer}
    if qa_format == 'hipporag_norefusal2' and is_refusal(answer):
        # still refused: ask once more, in the same conversation, for a best guess
        res = _chat(model_type, messages + [{"role": "assistant", "content": res},
                                            {"role": "user", "content": _FALLBACK_REQUEST}], sampling)
        answer = parse_hipporag_style_answer(res)
        info["fallback"] = True
    return answer, [id + 1 for id in doc_ids], info


def _chat(model_type, messages, sampling):
    if model_type == 'llama3':
        from together import Together
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            **sampling,
        )
    else:
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06" if model_type == 'gpt-4o' else "gpt-4o-mini-2024-07-18",
            messages=messages,
            **sampling,
        )
    return completion.choices[0].message.content


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
    f"person dimension is for REAL people only (actors, directors, athletes, politicians, etc.) — if the question is about an actor/author/creator of a fictional work, put their name here, not the character's name.\n"
    f"fictional_character dimension is for characters from movies, TV, books, games, or myth (e.g. 'Hagrid', 'Darth Vader') — NOT the real actor who plays them.\n"
    f"location dimension can be geographic places such as countries, cities, regions, venues, landmarks.\n"
    f"organization dimension can be named institutions, companies, government bodies, academic bodies, committees, and other formal groups.\n"
    f"event dimension can be named historical or notable events, ceremonies, releases, incidents.\n"
    f"work_title dimension is for titles of named creative works — movies, TV shows, episodes, songs, albums, books, symphonies/compositions, video games, plays — be as specific as possible, including disambiguating details if present (e.g. 'Symphony No. 25 (Mozart)' not just 'Mozart', 'Rip It Up (Orange Juice song)' not just 'Rip It Up').\n"
    f"concept dimension can be key scientific, technical, or cultural concepts and terminology.\n"
    f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
    f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
    f"Example Input:\n"
    f"Question: who sings the theme song for the proud family\n"
    f"Example Output:\n"
    f"Query 1:\n"
    f"query_dimension: 'work_title'; query_content: 'The Proud Family';\n"
    f"Query 2:\n"
    f"query_dimension: 'concept'; query_content: 'theme song';\n"
)


# v2: queries are matched against entity keys and passage titles, so steer the model toward
# literal, canonical entity names and away from descriptions of the answer type.
_DECOMPOSE_V2_RULES = (
    f"\nAdditional rules:\n"
    f"- Always include the main subject of the question (the entity whose Wikipedia article would answer it) "
    f"under its full canonical name, e.g. 'Nintendo', 'Chicago Bears', 'Latin Mass', 'Once Upon a Time (TV series)'.\n"
    f"- Only output entities or phrases that would literally appear in a passage answering the question. "
    f"Do NOT output descriptions of what is being asked or of the answer type, such as 'founding date', "
    f"'release date', 'winner', 'number of episodes', 'biggest country', 'end of martial law'.\n"
    f"- Only use the date dimension for explicit dates or years mentioned in the question; never output relative "
    f"or placeholder times such as 'last year', 'this year', 'year of win'.\n"
    f"- Name specific sub-topics directly when the question implies them, e.g. 'Curling at the 2018 Winter Olympics', "
    f"'Chicago Fire (season 4)'.\n"
)


def decompose_system_prompt(prompt_version):
    return _DECOMPOSE_SYSTEM_PROMPT + (_DECOMPOSE_V2_RULES if prompt_version == 'v2' else '')


class Query(BaseModel):
    query_content: str = Field(
        ...,
        title='Entity or phrase to query the documents'
    )
    query_dimension: Literal['date', 'person', 'fictional_character', 'work_title', 'location', 'organization', 'event', 'concept'] = Field(
        ...,
        title='Dimension of the entity or phrase to query documents'
    )


class AllQueries(BaseModel):
    list_of_queries: List[Query] = Field(
        ...,
        title='List of queries following the required format based on the question comprehension'
    )


def decompose_query_openai(query, prompt_version='v1'):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': decompose_system_prompt(prompt_version)},
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


def decompose_query_together(query, together_model="meta-llama/Llama-3.3-70B-Instruct-Turbo", prompt_version='v1'):
    from together import Together
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    json_instructions = (
        "\nOutput ONLY a JSON object of the form "
        '{"list_of_queries": [{"query_dimension": "<one of date|person|fictional_character|'
        'work_title|location|organization|event|concept>", "query_content": "<entity or short phrase>"}, ...]}. '
        "No markdown, no explanation, just the JSON object."
    )
    completion = client.chat.completions.create(
        model=together_model,
        messages=[
            {'role': 'system', 'content': decompose_system_prompt(prompt_version) + json_instructions},
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


def decompose_query(query, model_type="gpt-4o", prompt_version="v1"):
    try:
        if model_type == "llama3":
            pairs = decompose_query_together(query, prompt_version=prompt_version)
        else:
            pairs = decompose_query_openai(query, prompt_version=prompt_version)

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
    parser.add_argument("--data", type=str, required=True, help="Dataset name (expects QA/<data>/nq_qa_pairs.json).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4o-mini", "gpt-4o", 'llama3'], help="Select llm to get answer.")
    parser.add_argument("--retrieval_method", type=str, required=True, default="hypercube", choices=['hypercube', 'semantic', 'union'], help="Retrieval methods.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (for smoke-testing).")
    parser.add_argument("--hc_rank", type=str, default="freq", choices=["freq", "dense_tiebreak"],
                        help="Order of hypercube docs: hit count only, or hit count with ties broken by query similarity.")
    parser.add_argument("--fill", action="store_true", help="Top up hypercube results to k with dense results when cells match fewer than k docs.")
    parser.add_argument("--cells_from", type=str, default=None,
                        help="JSON {question id: cells}; reuse these decomposed cells instead of calling the LLM.")
    parser.add_argument("--gold", type=str, default=None,
                        help="JSON list (aligned with the QA file) of gold passage ids (0-based) to report passage recall@k.")
    parser.add_argument("--qa_format", type=str, default="default", choices=["default", "hipporag", "hipporag_norefusal", "hipporag_norefusal2"],
                        help="hipporag: format passages as 'title\\ntext' and answer with temperature 0, as in HippoRAG 2. hipporag_norefusal additionally tells the reader never to refuse; hipporag_norefusal2 adds a stricter rule, a second demonstration, and one retry when the reader still refuses.")
    parser.add_argument("--title_dim", action="store_true", help="Also match every query value against passage titles.")
    parser.add_argument("--decomp_prompt", type=str, default="v1", choices=["v1", "v2"],
                        help="v2 adds rules for canonical subject entities and drops answer-type descriptions.")
    parser.add_argument("--ablation", type=str, default="none", choices=ABLATIONS, help="Retrieval ablation variant.")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Whether to save per-sample outputs and scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    global ABLATION
    args = parse_args()
    ABLATION = args.ablation

    data_path = f"QA/{args.data}/nq_qa_pairs.json"

    qa_samples = load_dataset(data_path)
    if args.limit:
        qa_samples = qa_samples[:args.limit]

    llm_output = []
    recalls = []
    cells_cache = None
    if args.cells_from:
        with open(args.cells_from) as f:
            cells_cache = json.load(f)
    gold_ids = None
    if args.gold:
        with open(args.gold) as f:
            gold_ids = json.load(f)
    total_semantic, total_f1 = 0.0, 0.0

    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        gold_answer = sample["answer"]
        gold_answers = sample.get("answers") or [a.strip() for a in gold_answer.split(',')]

        print(f"Q{i+1}: {question}")

        if cells_cache is not None:
            cached = cells_cache.get(str(sample.get("id", i + 1)))
            cells = defaultdict(list, cached) if cached else None
        else:
            cells = decompose_query(query=question, model_type=args.model, prompt_version=args.decomp_prompt)
        print(f">>> Identified cells from the query: {cells} \n")

        predicted_answer, return_doc_ids, qa_info = llm_answer(args.model, question, cells, args.k, args.retrieval_method,
                                                      hc_rank=args.hc_rank, fill=args.fill,
                                                      use_title=args.title_dim, qa_format=args.qa_format)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answers}")

        f1 = max_over_references(f1_score, predicted_answer, gold_answers)
        sem = max_over_references(semantic_score, predicted_answer, gold_answers)
        print(f'>>> F1: {f1:.4f}, Semantic: {sem:.4f}\n')

        total_f1 += f1
        total_semantic += sem

        recall = None
        if gold_ids is not None and gold_ids[i]:
            gold = set(gold_ids[i])
            recall = len({d - 1 for d in return_doc_ids[:args.k]} & gold) / len(gold)
            recalls.append(recall)
            print(f'>>> Passage recall@{args.k}: {recall:.4f}\n')

        llm_output.append({
            "id": sample.get("id", i + 1),
            "question": question,
            "gold_answers": gold_answers,
            "predicted_answer": predicted_answer,
            "f1_score": f1,
            "semantic_score": sem,
            "return_doc_ids": return_doc_ids,
            "passage_recall": recall,
            "refusal_fallback": qa_info["fallback"],
            "first_answer": qa_info["first_answer"],
        })

    save_ent2emb()

    num_samples = len(qa_samples)
    avg_f1 = total_f1 / num_samples
    avg_semantic = total_semantic / num_samples
    print(f"Total samples: {num_samples}, average F1: {avg_f1:.4f}, average Semantic: {avg_semantic:.4f}")
    if recalls:
        print(f"Average passage recall@{args.k}: {sum(recalls) / len(recalls):.4f} over {len(recalls)} questions")

    results = {
        "total_samples": num_samples,
        "average_F1": avg_f1,
        "average_Semantic": avg_semantic,
    }
    if recalls:
        results[f"average_passage_recall@{args.k}"] = sum(recalls) / len(recalls)

    output_dir = f"output/{args.data}/{args.model}/"
    tag = f"{args.retrieval_method}_k{args.k}"
    if args.hc_rank != "freq":
        tag += f"_{args.hc_rank}"
    if args.fill:
        tag += "_fill"
    if args.title_dim:
        tag += "_title"
    if args.qa_format != "default":
        tag += f"_qa{args.qa_format}"
    if args.decomp_prompt != "v1":
        tag += f"_decomp{args.decomp_prompt}"
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
