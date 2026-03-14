import argparse
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from inference import generate_sql, MAX_SEQ_LENGTH

DEFAULT_MODEL_PATH = "outputs/sql_translator_model"
HF_MODEL_ID = "happyhackingspace/sql-translator-llama3"


def load_model(model_path, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def normalize_sql(sql):
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate(model, tokenizer, dataset_name, num_samples, split="train"):
    dataset = load_dataset(dataset_name, split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    exact_match = 0
    total = len(dataset)
    results = []

    for i, row in enumerate(dataset):
        result = generate_sql(model, tokenizer, row["sql_context"], row["sql_prompt"])
        predicted = normalize_sql(result["sql"])
        expected = normalize_sql(row["sql"])
        match = predicted == expected

        if match:
            exact_match += 1

        results.append({
            "index": i,
            "match": match,
            "expected": row["sql"],
            "predicted": result["sql"],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] Exact match so far: {exact_match}/{i+1} ({exact_match/(i+1)*100:.1f}%)")

    accuracy = exact_match / total * 100
    print(f"\nResults: {exact_match}/{total} exact match ({accuracy:.1f}%)")
    return results, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--dataset", type=str, default="gretelai/synthetic_text_to_sql")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    model_path = HF_MODEL_ID if args.hf else args.model
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path, load_in_4bit=not args.no_4bit)
    print(f"Evaluating on {args.num_samples} samples from {args.dataset}\n")

    results, accuracy = evaluate(model, tokenizer, args.dataset, args.num_samples)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
