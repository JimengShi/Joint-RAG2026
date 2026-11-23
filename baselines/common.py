"""Shared code for the baselines: dataset paths, answer prompts, LLM calls and
saving outputs. All baselines use the same prompts and LLM settings, so they
differ only in how the top-k documents are retrieved.

Outputs are saved to output/<data>/<model>/llm_output_<method>_k<k>_responses.json,
which evaluation/nlp_metric.py and evaluation/llm_as_judge.py read by default.
"""
import argparse
import json
import os
import sys
import time

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metric import em_check, f1_score

# corpus file (one document per line) and QA file of each dataset
DATASETS = {
    "nq": ("hypercube/nq/passages.txt", "QA/nq/nq_qa_pairs.json"),
    "popqa": ("hypercube/popqa/passages.txt", "QA/popqa/popqa_qa_pairs.json"),
    "legalbench": ("corpus/legalbench/contractnli.txt", "QA/legalbench/contractnli.json"),
    "geography": ("corpus/geography/SciDCC-Geography.txt", "QA/geography/synthetic_qa.json"),
    "hurricane": ("corpus/hurricane/SciDCC-Hurricane.txt", "QA/hurricane/synthetic_qa.json"),
    "aging_dam": ("corpus/aging_dam/corpus.txt", "QA/aging_dam/synthetic_qa.json"),
}

LLMS = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini-2024-07-18"),
    "llama3": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "gpt-4o": ("openai", "gpt-4o-2024-08-06"),
}

INSTRUCTIONS = {
    "open_domain": (
        "Answer the query based on the given retrieved documents. "
        "If the query asks what/which/who/when/where/how many/what year, you must directly output the final answer "
        "as precise as possible without any other explanations. "
        "If it is a yes-or-no query, only output yes or no. "
        "Documents:\n"
    ),
    "legalbench": (
        "Answer the query based on the given retrieved documents. "
        "Please keep the answer as precise as possible. "
        "Documents:\n"
    ),
    "scidcc": (
        "Answer the query based on the given retrieved documents. "
        'If the query asks the quantitative analysis, such as starting with "How many", "How much", "How much greater", "How wide", "What percentage", "How far", "How long", "How old", "What portion", "What depth", you must directly output the quantitative answer as short as possible.'
        'If the query starting with "what", "what percentage" or the query includes the word "likelihood", please directly output the number with %. '
        'If the query starting with "what specific", "which years", "who", please directly output the answer without explanation or other information.'
        "Otherwise, please provide as precise information as possible based the retrieved documents. "
        "Documents:\n"
    ),
}


def instruction_for(data):
    if data in ("nq", "popqa"):
        return INSTRUCTIONS["open_domain"]
    if data == "legalbench":
        return INSTRUCTIONS["legalbench"]
    return INSTRUCTIONS["scidcc"]


# ================ Data ================
def read_corpus(data):
    with open(DATASETS[data][0], encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_qa(data):
    with open(DATASETS[data][1], encoding="utf-8") as f:
        return json.load(f)


def gold_answers_of(sample):
    """List of accepted gold answers of a QA sample."""
    if isinstance(sample.get("answers"), list) and sample["answers"]:
        return [str(a) for a in sample["answers"]]
    return [a.strip() for a in str(sample["answer"]).split(",")] if "," in str(sample["answer"]) else [str(sample["answer"])]


def make_record(data, index, question, sample, predicted_answer, doc_ids):
    """Output record in the format read by the evaluation scripts."""
    record = {"index": index, "question": question, "predicted_answer": predicted_answer,
              "return_doc_ids": [d + 1 for d in doc_ids]}  # 1-based line ids in the corpus file
    if data in ("nq", "popqa"):
        record["gold_answers"] = gold_answers_of(sample)  # several accepted answers
    else:
        gold = " ".join(sample["answers"]) if "answers" in sample else sample["answer"]
        record["gold_answer"] = gold
        record["gold_answers"] = gold
    return record


# ================ LLM ================
def llm_answer(model, data, query, docs):
    """Answer the query with the retrieved documents (list of strings)."""
    provider, model_id = LLMS[model]
    context = "\n\n".join(f"Document {i + 1}: {d}" for i, d in enumerate(docs))
    messages = [
        {"role": "system", "content": instruction_for(data) + context},
        {"role": "user", "content": "Query: " + query + "\nAnswer:"},
    ]
    if provider == "together":
        from together import Together
        client = Together(api_key=os.getenv("TOGETHER_API_KEY", "your TOGETHER_API_KEY"))
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your OPENAI_API_KEY"))
    completion = client.chat.completions.create(model=model_id, messages=messages, temperature=0)
    return completion.choices[0].message.content


# ================ Running a baseline ================
def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=str, required=True, choices=list(DATASETS), help="Dataset.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=list(LLMS), help="LLM that generates the answers.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"], help="Save the outputs.")
    return parser


def run_baseline(method, retrieve, corpus, args):
    """retrieve(question, k) -> list of 0-based doc ids."""
    qa_samples = load_qa(args.data)
    if args.limit:
        qa_samples = qa_samples[:args.limit]

    outputs, total_em, total_f1, total_retrieval_time = [], 0.0, 0.0, 0.0
    for i, sample in enumerate(tqdm(qa_samples)):
        question = sample["question"]
        start = time.time()
        doc_ids = retrieve(question, args.k)
        total_retrieval_time += time.time() - start

        predicted_answer = llm_answer(args.model, args.data, question, [corpus[d] for d in doc_ids])
        record = make_record(args.data, i + 1, question, sample, predicted_answer, doc_ids)
        golds = record["gold_answers"] if isinstance(record["gold_answers"], list) else [record["gold_answer"]]
        total_em += em_check(predicted_answer, golds)
        total_f1 += max(f1_score(g, predicted_answer) for g in golds)
        outputs.append(record)

    n = len(qa_samples)
    print(f"[{method}] {args.data} / {args.model}: {n} samples, EM {total_em / n * 100:.1f}, F1 {total_f1 / n * 100:.1f}, "
          f"avg retrieval time {total_retrieval_time / n:.3f}s")

    if args.save == "true":
        output_dir = f"output/{args.data}/{args.model}"
        os.makedirs(output_dir, exist_ok=True)
        path = f"{output_dir}/llm_output_{method}_k{args.k}_responses.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Saved to {path}")
