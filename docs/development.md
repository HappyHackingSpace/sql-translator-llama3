# Development Notes

This document is intended for developers contributing to the SQL Translator fine-tuning project by [HappyHackingSpace](https://github.com/HappyHackingSpace).

---

## Project Structure

```text
sql-translator-llama3/
├── src/
│   └── fine_tune.py
├── outputs/
├── models/
├── data/
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
└── docs/
    ├── getting-started.md
    ├── api-reference.md
    ├── development.md
    └── examples/
```

---

## Development Environment

We recommend using **conda** for environment management.

Create and activate the environment:

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
pip install -r requirements.txt
```

---

## Testing

While the current implementation doesn’t include unit tests, contributors are encouraged to:

- Modularize new functions
- Write simple test scripts in a `tests/` folder (if added)
- Validate end-to-end runs of `src/fine_tune.py`

---

## Design Philosophy

- Clear, reusable components
- No business logic inside notebooks
- Model loading, training, data prep = separate functions
- Community-readable, well-commented code

---

## Upload to Hugging Face

When pushing a trained model:

```python
from huggingface_hub import create_repo, upload_folder

create_repo("HappyHackingSpace/sql-translator-llama3", private=False, exist_ok=True)
upload_folder(
    repo_id="HappyHackingSpace/sql-translator-llama3",
    folder_path="outputs/sql_translator_model",
    commit_message="Push from dev machine"
)
```

---

## Contribution Tips

- Keep PRs focused and small
- Explain your reasoning in the PR description
- Use meaningful commit messages

Thank you for improving this project!