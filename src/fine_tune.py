import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from packaging.version import Version as V
from torch import __version__ as torch_version

# Model and tokenizer config
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
DTYPE = None

# Training config
BATCH_SIZE = 2
GRAD_ACC_STEPS = 2
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
OUTPUT_DIR = "outputs/sql_translator_model"
SEED = 3407

# Prompt format
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction
Company database: {}

### Input:
SQL Prompt: {}

### Response:
SQL : {}

Explanation: {}
"""

# Load dataset
def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
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
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=1)
    return train_dataset

# Load model
def load_model():
    xformers = "xformers==0.0.27" if V(torch_version) < V("2.4.0") else "xformers"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer

# Training setup
def train_model(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=10,
            learning_rate=LEARNING_RATE,
            fp16=False,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir="outputs",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    model, tokenizer = load_model()
    train_dataset = load_and_prepare_dataset(tokenizer=tokenizer)
    train_model(model, tokenizer, train_dataset)
