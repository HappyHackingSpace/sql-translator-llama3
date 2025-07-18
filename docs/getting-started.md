# Getting Started

Welcome to the official SQL Translator fine-tuning project from the Happy Hacking Space community!

This guide will help you set up your development environment, install dependencies, and train your own SQL translation model using the LLaMA 3 8B base.

## Environment Setup

We recommend using `conda` to manage dependencies.

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
```

## Install Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Hugging Face Authentication

To push or pull models from the Hugging Face Hub:

```bash
huggingface-cli login
# Paste your token
```

## Training the Model

Use the following command to start fine-tuning:

```bash
python src/fine_tune.py
```

The training script will:
- Load the `gretelai/synthetic_text_to_sql` dataset
- Format the prompts into instruction-tuning format
- Fine-tune the `unsloth/Meta-Llama-3.1-8B` model
- Save outputs to `outputs/sql_translator_model`

## Upload to Hugging Face

To push the trained model to your organization:

```python
from huggingface_hub import create_repo, upload_folder

create_repo("HappyHackingSpace/sql-translator-llama3", private=False, exist_ok=True)
upload_folder(
    repo_id="HappyHackingSpace/sql-translator-llama3",
    folder_path="outputs/sql_translator_model",
    commit_message="Initial model push"
)
```

---

For any issues, check the [GitHub repo](https://github.com/HappyHackingSpace/sql-translator-llama3).