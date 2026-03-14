import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import yaml
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel
from inference import generate_sql, MAX_SEQ_LENGTH

HF_MODEL_ID = "happyhackingspace/sql-translator-llama3"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_path, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_eval_dataset(cfg, num_samples):
    val_path = os.path.join(cfg["output"]["checkpoints_dir"], "val_dataset")
    seed = cfg["training"]["seed"]
    dataset_name = cfg["dataset"]["name"]
    val_split = cfg["dataset"].get("validation_split", 0.05)

    try:
        dataset = load_from_disk(val_path)
        print(f"Loaded validation set from {val_path} ({len(dataset)} samples)")
    except (FileNotFoundError, FileExistsError, OSError):
        print(f"No saved validation set found, splitting from {dataset_name}")
        full = load_dataset(dataset_name, split="train")
        dataset = full.train_test_split(test_size=val_split, seed=seed)["test"]

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def normalize_sql(sql):
    return " ".join(sql.lower().strip().rstrip(";").split())


def evaluate(model, tokenizer, dataset):
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
            print(f"  [{i+1}/{total}] Exact match: {exact_match}/{i+1} ({exact_match/(i+1)*100:.1f}%)")

    accuracy = exact_match / total * 100
    print(f"\nResults: {exact_match}/{total} exact match ({accuracy:.1f}%)")
    return results, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.hf:
        model_path = HF_MODEL_ID
    elif args.model:
        model_path = args.model
    else:
        model_path = cfg["output"]["dir"]

    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path, load_in_4bit=not args.no_4bit)

    dataset = load_eval_dataset(cfg, args.num_samples)
    print(f"Evaluating on {len(dataset)} samples\n")

    results, accuracy = evaluate(model, tokenizer, dataset)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
