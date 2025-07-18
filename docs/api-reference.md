# API Reference

This document explains the core components of the SQL Translator fine-tuning pipeline based on the `src/fine_tune.py` script.

---

## File: `src/fine_tune.py`

### Function: `load_model()`

Loads the Unsloth LLaMA 3 model and prepares it for LoRA fine-tuning.

**Returns:**  
- `model`: LoRA-enabled model ready for training  
- `tokenizer`: Corresponding tokenizer  

**Key Parameters:**  
- `model_name`: unsloth/Meta-Llama-3.1-8B  
- `load_in_4bit`: True  
- `max_seq_length`: 1024  
- `use_gradient_checkpointing`: Enabled  
- `lora_r`, `lora_alpha`, `lora_dropout`: Set to typical values

---

### Function: `load_and_prepare_dataset()`

Loads the `gretelai/synthetic_text_to_sql` dataset and formats each example using Alpaca-style prompts.

**Returns:**  
- `train_dataset`: Dataset containing formatted `text` fields for SFT training

**Prompt Format:**  
```text  
Below is an instruction that describes a task, paired with an input...

Company database: {db}

SQL Prompt: {prompt}

SQL : {sql}

Explanation: {explanation}
```

---

### Function: `train_model(model, tokenizer, dataset)`

Creates a `SFTTrainer` instance from Hugging Face TRL and performs one epoch of fine-tuning.

**Args:**  
- `model`: LoRA-wrapped LLaMA 3 model  
- `tokenizer`: Tokenizer from Unsloth  
- `dataset`: Formatted instruction dataset  

**Trainer Settings:**  
- `per_device_train_batch_size`: 2  
- `gradient_accumulation_steps`: 2  
- `learning_rate`: 2e-4  
- `optim`: adamw_8bit  
- `output_dir`: outputs/sql_translator_model  

---

### Constants

- `MAX_SEQ_LENGTH = 1024`  
- `LEARNING_RATE = 2e-4`  
- `SEED = 3407`  
- `OUTPUT_DIR = outputs/sql_translator_model`  

See `src/fine_tune.py` for full implementation.