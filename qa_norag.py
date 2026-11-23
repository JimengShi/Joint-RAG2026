from openai import OpenAI
import json
import argparse
import pandas as pd
from together import Together
import os
import re

# ================ Load Dataset ================
def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            return json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")  # Converts to list of dicts
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")


# ================ LLMs ================
def llm_answer(question, data_type, model_type):
    if data_type == "scifact":
        instruction = 'Determine if the claim in the query is SUPPORT or CONTRADICT, based on the given retrieved documents. ' \
        'Please keep the answer as precise as possible. You must output either SUPPORT or CONTRADICT' \
        'Documents:\n'
    elif data_type == "popqa":
        # Closed-book version of the HippoRAG-style prompt used in qa_rag_popqa.py
        instruction = 'Answer the question from your own knowledge. ' \
        'Your response start after "Thought: ", where you will briefly reason about the question. ' \
        'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
        question = 'Question: ' + question + '\nThought: '
    else:
        instruction = 'Answer the query based on the given retrieved documents. ' \
        'Please keep the answer as precise as possible. ' \
        'Documents:\n'


    if model_type == 'llama3':
        os.environ.setdefault("TOGETHER_API_KEY", "your TOGETHER_API_KEY")  # or export TOGETHER_API_KEY
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        )
        res = completion.choices[0].message.content  
        if data_type == "popqa":
            res = res.split("Answer:")[-1].strip()
        return res
    
    elif model_type == 'deepseek':
        os.environ.setdefault("TOGETHER_API_KEY", "your TOGETHER_API_KEY")  # or export TOGETHER_API_KEY
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",  # deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free, deepseek-ai/DeepSeek-R1
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            extra_body={
                "include_reasoning": False #True
            },
            max_tokens=4096
        )
        res = completion.choices[0].message.content  
        # Remove content between <think> tags (including the tags themselves)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()
        cleaned_res = cleaned_res.split("Answer:")[-1].strip()
        return cleaned_res
    
    elif model_type == 'qwen':
        os.environ.setdefault("TOGETHER_API_KEY", "your TOGETHER_API_KEY")  # or export TOGETHER_API_KEY
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        )
        res = completion.choices[0].message.content  
        return res
    
    elif model_type == "gpt-4":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_type,  
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
    elif model_type == "gpt-4o":
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    elif model_type == "gpt-4o-mini":
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        res = response.choices[0].message.content
        if data_type == "popqa":
            res = res.split("Answer:")[-1].strip()
        return res


# ================ Evaluation Metrics ================
from utils.metric import bleu_score, semantic_score, f1_score


def max_over_references(score_fn, predicted, gold_answers):
    """Same aggregation as qa_rag_popqa.py: max score over accepted answers."""
    if not gold_answers:
        return 0.0
    return max(score_fn(predicted, g) for g in gold_answers)


# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4", "gpt-4o", "gpt-4o-mini", "llama3", "llama4"], help="Select llm to get answer.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    args = parse_args()
    if args.data == "scifact":
        data_path = f"QA/{args.data}/claims.json"
    elif args.data == "legalbench":
        data_path = f"QA/{args.data}/contractnli.json"
    elif args.data in ["geography", "hurricane", "aging_dam"]:
        data_path = f"QA/{args.data}/synthetic_qa.json"
    else:
        data_path = f"QA/{args.data}/{args.data}_qa_pairs.json"

    # Load dataset
    qa_samples = load_dataset(data_path)
    if args.limit:
        qa_samples = qa_samples[:args.limit]
    
    # Evaluation
    print("\n--- QA Evaluation ---\n")

    # Initialize evaluation metrics
    total_bleu, total_semantic, total_f1 = 0, 0, 0
    
    llm_output = []
    for i, sample in enumerate(qa_samples):
        question = sample["question"]
        if "answers" in sample:
            gold_answers = [" ".join(sample["answers"])]
        elif args.data in ["geography", "hurricane", "aging_dam"]:
            gold_answers = [sample["answer"]]  # free-form sentence, not a comma-separated alias list
        else:
            gold_answers = [a.strip() for a in sample["answer"].split(',')]
        gold_answer = gold_answers[0] if len(gold_answers) == 1 else gold_answers

        print(f"Q{i+1}: {question}")
        predicted_answer = llm_answer(question, args.data, args.model)
        bleu = max_over_references(bleu_score, predicted_answer, gold_answers)
        sem = max_over_references(semantic_score, predicted_answer, gold_answers)
        f1 = max_over_references(f1_score, predicted_answer, gold_answers)
        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f'Semantic_score: {sem:.4f}')
        print(f"\n")

        total_bleu += bleu
        total_semantic += sem
        total_f1 += f1

        # Save output for each sample
        llm_output.append({
            "index": i+1,
            "question": question,
            "gold_answer": gold_answer,
            "gold_answers": gold_answer,  # key read by evaluation/llm_as_judge.py
            "predicted_answer": predicted_answer,
            "f1_score": f1,
            "bleu_score": bleu,
            "semantic_score": sem
        })
    

    # Averaging Summary Information
    num_samples = len(qa_samples)
    print(f"Total samples: {num_samples}, average BLEU: {total_bleu / num_samples:.4f}, average F1: {total_f1 / num_samples:.4f}, average Semantic: {total_semantic / num_samples:.4f}")

    # Assume you have these variables already defined
    results = {
        "total_samples": num_samples,
        "average_BLEU": round(total_bleu / num_samples, 4),
        "average_F1": round(total_f1 / num_samples, 4),
        "average_Semantic": round(total_semantic / num_samples, 4)
    }

    output_dir = f"output/{args.data}/{args.model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    suffix = f"_n{args.limit}" if args.limit else ""
    if args.save == "true":
        with open(f"{output_dir}/llm_output_no_rag{suffix}_responses.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2, ensure_ascii=False)

        with open(f"{output_dir}/llm_output_no_rag{suffix}_scores.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("Saved llm output!")


if __name__ == "__main__":
    main()