# Contributing

We welcome contributions — bug fixes, features, docs, examples.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/HappyHackingSpace/sql-translator-llama3.git
   ```
3. Create a branch:
   ```bash
   git checkout -b feature/my-feature-name
   ```
4. Set up the environment:
   ```bash
   conda create -n sqlft python=3.10 -y && conda activate sqlft
   pip install -r requirements.txt
   ```

## Guidelines

- Follow PEP8
- Keep code modular
- Avoid hardcoded paths or credentials
- Test your changes with `python src/fine_tune.py` and `python src/evaluate.py`

## Pull Requests

1. Ensure your code runs without errors
2. Include a meaningful title and description
3. Link related issues if applicable

## Issues

Use the GitHub Issues tab with the provided templates.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
