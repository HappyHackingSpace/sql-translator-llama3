import argparse
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Company database: {}

### Input:
SQL Prompt: {}

### Response:
SQL: {}

Explanation: {}
"""


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_prepare_dataset(tokenizer, cfg):
    dataset = load_dataset(cfg["dataset"]["name"])
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for db, prompt, sql, explanation in zip(
            examples["sql_context"],
            examples["sql_prompt"],
            examples["sql"],
            examples["sql_explanation"]):
            text = ALPACA_PROMPT.format(db, prompt, sql, explanation) + eos_token
            texts.append(text)
        return {"text": texts}

    train_dataset = dataset["train"]

    max_samples = cfg["dataset"].get("max_samples")
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=1)
    return train_dataset


def load_model(cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def train_model(model, tokenizer, dataset, cfg):
    t = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            num_train_epochs=t["num_epochs"],
            warmup_steps=t["warmup_steps"],
            learning_rate=t["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=t["logging_steps"],
            optim=t["optimizer"],
            weight_decay=t["weight_decay"],
            lr_scheduler_type=t["lr_scheduler"],
            seed=t["seed"],
            output_dir=cfg["output"]["checkpoints_dir"],
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(cfg["output"]["dir"])
    tokenizer.save_pretrained(cfg["output"]["dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg)
    train_dataset = load_and_prepare_dataset(tokenizer=tokenizer, cfg=cfg)
    train_model(model, tokenizer, train_dataset, cfg)
