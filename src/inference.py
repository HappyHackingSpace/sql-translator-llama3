import argparse
import torch
from unsloth import FastLanguageModel

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Company database: {}

### Input:
SQL Prompt: {}

### Response:
"""

DEFAULT_MODEL_PATH = "outputs/sql_translator_model"
HF_MODEL_ID = "happyhackingspace/sql-translator-llama3"
MAX_SEQ_LENGTH = 1024


def load_model(model_path, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_sql(model, tokenizer, db_schema, question, max_new_tokens=256, temperature=0.1):
    prompt = ALPACA_PROMPT.format(db_schema, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return parse_response(response)


def parse_response(response):
    sql = response.strip()
    explanation = ""

    if "Explanation:" in response:
        parts = response.split("Explanation:", 1)
        sql = parts[0].strip()
        explanation = parts[1].strip()

    if sql.startswith("SQL :"):
        sql = sql[len("SQL :"):].strip()
    elif sql.startswith("SQL:"):
        sql = sql[len("SQL:"):].strip()

    return {"sql": sql, "explanation": explanation}


def interactive_mode(model, tokenizer, max_new_tokens, temperature):
    print("\n" + "=" * 60)
    print("  SQL Translator - Interactive Mode")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    db_schema = input("\nDatabase schema (table definitions):\n> ").strip()
    if not db_schema:
        db_schema = "No schema provided"
        print(f"  Using: '{db_schema}'")

    while True:
        question = input("\nQuestion (natural language):\n> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nGenerating SQL...\n")
        result = generate_sql(model, tokenizer, db_schema, question, max_new_tokens, temperature)

        print(f"SQL:\n  {result['sql']}")
        if result["explanation"]:
            print(f"\nExplanation:\n  {result['explanation']}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="SQL Translator - Generate SQL from natural language")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hf", action="store_true", help=f"Load from HF Hub ({HF_MODEL_ID})")
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    model_path = HF_MODEL_ID if args.hf else args.model
    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path, load_in_4bit=not args.no_4bit)
    print("Model loaded.\n")

    if args.question:
        schema = args.schema or "No schema provided"
        result = generate_sql(model, tokenizer, schema, args.question, args.max_tokens, args.temperature)
        print(f"SQL:\n  {result['sql']}")
        if result["explanation"]:
            print(f"\nExplanation:\n  {result['explanation']}")
        return

    interactive_mode(model, tokenizer, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
