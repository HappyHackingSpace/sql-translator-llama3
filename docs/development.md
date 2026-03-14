# Development Notes

## Setup

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
pip install -r requirements.txt
```

## Workflow

1. Edit `config.yaml` for your experiment
2. Train: `python src/fine_tune.py`
3. Evaluate: `python src/evaluate.py`
4. Inference: `python src/inference.py`

The training script splits the dataset into train/validation sets automatically. The validation set is saved to `outputs/val_dataset/` so that `evaluate.py` can use the same held-out data.

## Contribution Tips

- Keep PRs focused and small
- Explain your reasoning in the PR description
- Use meaningful commit messages
