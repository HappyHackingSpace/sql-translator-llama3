# Getting Started

## Setup

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
pip install -r requirements.txt
```

## Hugging Face Authentication

Required for pushing models or pulling gated models:

```bash
huggingface-cli login
```

## Train

```bash
python src/fine_tune.py
```

This will:
- Load and split the dataset (95% train, 5% validation)
- Fine-tune the model with LoRA
- Save the model to `outputs/sql_translator_model/`
- Save the validation set to `outputs/val_dataset/`

Edit `config.yaml` to change hyperparameters, dataset, or model.

## Inference

```bash
python src/inference.py --hf --schema "employees(id, name, salary)" --question "List all employees"
python src/inference.py --hf  # interactive mode
```

## Evaluate

```bash
python src/evaluate.py --hf --num-samples 50
```

For more details, see [API Reference](api-reference.md).
