"""Shared helpers for the evaluation scripts: argument definitions, locating the
LLM output file of a run, and reading gold answers from it."""
import json
import os

DATASETS = ["nq", "popqa", "legalbench", "hurricane", "geography", "aging_dam"]
MODELS = ["gpt-4o-mini", "llama3", "gpt-4o"]
RETRIEVAL_METHODS = ["hypercube", "bm25", "nvembed", "hybrid", "raptor", "graphrag", "hipporag2", "none"]


def add_common_args(parser):
    parser.add_argument("--data", type=str, required=True, choices=DATASETS, help="Dataset.")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help="LLM that generated the answers.")
    parser.add_argument("--retrieval_method", type=str, default="hypercube", choices=RETRIEVAL_METHODS,
                        help="Retrieval method of the run ('hypercube' is Joint-RAG, 'none' is direct inference).")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents of the run.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to the LLM output file. Overrides the path derived from --data/--model/--retrieval_method/--k.")


def resolve_output_path(args):
    """Default location of the file written by qa_norag.py / qa_rag_<data>.py."""
    if args.input:
        return args.input
    out_dir = f"output/{args.data}/{args.model}"
    if args.retrieval_method == "none":
        return f"{out_dir}/llm_output_no_rag_responses.json"
    if args.data in ("nq", "popqa", "legalbench"):
        return f"{out_dir}/llm_output_{args.retrieval_method}_k{args.k}_responses.json"
    if args.data == "aging_dam":
        return f"{out_dir}/llm_output_{args.retrieval_method}_nv.json"
    return f"{out_dir}/llm_output_{args.retrieval_method}.json"


def load_samples(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the QA script with --save true first, or pass --input.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gold_answers(sample):
    """List of accepted gold answers. NQ/PopQA outputs store a list under
    'gold_answers'; the other datasets store a single string under 'gold_answer'."""
    for key in ("gold_answers", "answers"):
        if isinstance(sample.get(key), list) and sample[key]:
            return [str(a) for a in sample[key]]
    return [str(sample["gold_answer"])]
