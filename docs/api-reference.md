# API Reference

## `src/fine_tune.py`

### `load_config(path) -> dict`
Loads YAML config file.

### `load_model(cfg) -> (model, tokenizer)`
Loads the base model and applies LoRA. All parameters come from `config.yaml`.

### `load_and_prepare_dataset(tokenizer, cfg) -> (train_dataset, val_dataset)`
Loads the dataset, applies train/validation split, and formats prompts in Alpaca style.

### `train_model(model, tokenizer, dataset, cfg)`
Runs SFT training and saves the model to the configured output directory.

---

## `src/inference.py`

### `load_model(model_path, load_in_4bit=True) -> (model, tokenizer)`
Loads a fine-tuned model for inference.

### `generate_sql(model, tokenizer, db_schema, question, ...) -> dict`
Generates SQL from natural language. Returns `{"sql": ..., "explanation": ...}`.

### `parse_response(response) -> dict`
Splits raw model output into SQL and explanation parts.

---

## `src/evaluate.py`

### `load_eval_dataset(dataset_name, num_samples, seed) -> dataset`
Loads the saved validation set, or creates a split if not available.

### `evaluate(model, tokenizer, dataset) -> (results, accuracy)`
Runs exact match evaluation and prints progress.

---

## CLI Usage

```bash
python src/fine_tune.py --config config.yaml
python src/inference.py --hf --schema "..." --question "..."
python src/evaluate.py --hf --num-samples 100 --output results.json
```
