# SQL Translator – Fine-Tuning LLaMA 3 with Unsloth

Fine-tuning pipeline for SQL generation from natural language using LLaMA 3 8B.
Built by [Happy Hacking Space](https://github.com/HappyHackingSpace).

## Overview

| | |
|---|---|
| Base Model | `unsloth/Meta-Llama-3.1-8B` |
| Dataset | `gretelai/synthetic_text_to_sql` |
| Trained Model | [happyhackingspace/sql-translator-llama3](https://huggingface.co/happyhackingspace/sql-translator-llama3) |
| Tuning | LoRA (4-bit) via `unsloth` + `trl` |

## Setup

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
pip install -r requirements.txt
```

## Usage

**Train:**

```bash
python src/fine_tune.py                        # default config
python src/fine_tune.py --config my_config.yaml # custom config
```

**Inference:**

```bash
python src/inference.py --hf \
  --schema "employees(id, name, salary, dept_id)" \
  --question "Show employees with salary above 50000"

python src/inference.py --hf  # interactive mode
```

**Evaluate:**

```bash
python src/evaluate.py --hf
python src/evaluate.py --hf --num-samples 200
python src/evaluate.py --hf --output results.json
```

## Configuration

All hyperparameters live in [`config.yaml`](config.yaml). Key options:

```yaml
dataset:
  max_samples: null       # limit dataset size (e.g. 1000)
  validation_split: 0.05  # held out for evaluation
```

## Hardware

| Setup | RAM |
|---|---|
| LLaMA 3 8B (4-bit) | ~12–16 GB |
| LLaMA 3 8B (fp16) | ~32 GB |

Recommended GPUs: A100, H100, 2x3090, 2x4090

## Docs

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Development](docs/development.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE) — © 2025 [Happy Hacking Space](https://github.com/HappyHackingSpace)
