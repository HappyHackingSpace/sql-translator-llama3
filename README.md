# 🦙 SQL Translator – Fine-Tuning LLaMA 3 with Unsloth

A fully open-source, fine-tuning pipeline for SQL generation from natural language using the LLaMA 3 8B model.  
This project was built by the [Happy Hacking Space](https://github.com/HappyHackingSpace) community to serve as both:

- A production-ready fine-tuning implementation  
- A reusable **template** for instruction tuning projects  
- A launching pad for your own LLM-based workflows in SQL, data science, and beyond

---

## Project Overview

- Base Model: `unsloth/Meta-Llama-3.1-8B`
- Dataset: `gretelai/synthetic_text_to_sql`
- Output Model: [`happyhackingspace/sql-translator-llama3`](https://huggingface.co/happyhackingspace/sql-translator-llama3)
- Format: Alpaca-style prompt generation
- Tuning Method: PEFT (LoRA) via `unsloth` + `trl`
- Quantization: 4-bit (low RAM usage)

---

## Installation

We recommend using `conda`:

```bash
conda create -n sqlft python=3.10 -y
conda activate sqlft
pip install -r requirements.txt
```

---

## Training the Model

Run the following command:

```bash
python src/fine_tune.py
```

This performs:
- Dataset loading and formatting
- LoRA-based fine-tuning on LLaMA 3
- Saving trained model to: `outputs/sql_translator_model/`

---

## Push to Hugging Face (Optional)

If you want to share your model:

```python
from huggingface_hub import create_repo, upload_folder

create_repo("your-huggingface-username/sql-translator-llama3", private=False, exist_ok=True)
upload_folder(
    repo_id="your-huggingface-username/sql-translator-llama3",
    folder_path="outputs/sql_translator_model",
    commit_message="Initial model push"
)
```

**The model trained in this repository is available here:**  
[`https://huggingface.co/happyhackingspace/sql-translator-llama3`](https://huggingface.co/happyhackingspace/sql-translator-llama3)
---

## Tips for Fine-Tuning & Extension

### Prompt Engineering

- You can enrich the prompt format with:
  - Schema diagrams
  - Example tables
  - System/user role identifiers
  - Task difficulty hints (e.g. joins, nested queries)

### LoRA Parameters and Tuning

| Parameter           | Default | Effect                                                   |
|---------------------|---------|-----------------------------------------------------------|
| `lora_alpha`        | 16      | Controls update strength — larger = stronger adaptation  |
| `r` (rank)          | 16      | LoRA rank — higher = more trainable capacity             |
| `gradient_checkpointing` | on | Reduces RAM usage during training                        |

### Dataset Design

- Start with ~1000 samples for testing  
- Use 5000–20,000+ for decent adaptation  
- Mix in complex and edge case queries  
- Include natural explanations (`sql_explanation`) where possible

---

## Model & Hardware Recommendations

| Model Name           | RAM Needed | Best Use Case                          |
|----------------------|------------|----------------------------------------|
| LLaMA 3 8B (4-bit)   | ~12–16 GB  | Everyday fine-tuning or testing        |
| LLaMA 3 8B (fp16)    | ~32 GB     | Precision-sensitive adaptation         |
| LLaMA 3 70B          | >100 GB    | Research-grade training only           |

- Not suitable for inference on pure CPU setups  
- Recommended GPUs: A100 (40GB), H100, 2×3090, 2×4090, or better  
- Can be used with Colab Pro+ or Paperspace for small-scale tuning

---

## Example

Check: [`docs/examples/sql_prompt_example.md`](docs/examples/sql_prompt_example.md)

---

## Full Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Developer Notes](docs/development.md)

---

## Contributing

We welcome ideas, suggestions, improvements, and fixes!

- [CONTRIBUTING.md](CONTRIBUTING.md)
+ [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## License

MIT License — see [LICENSE](LICENSE)  
© 2025 [Happy Hacking Space](https://github.com/HappyHackingSpace)

---

## About This Project

This project is a **reusable fine-tuning blueprint**, not just a one-off repo.

You can:

- Plug in your own dataset (e.g., chatbot logs, structured instructions)
- Swap the base model (e.g., LLaMA 3 → Mistral, Phi-2, TinyLlama)
- Add multi-turn formatting for conversational SQL generation
- Reuse the structure in enterprise NLP tasks or research pipelines

---

## Model Output Location

> The final trained model from this project is publicly available on the Hugging Face Hub:  
**[happyhackingspace/sql-translator-llama3](https://huggingface.co/happyhackingspace/sql-translator-llama3)**

You can load and use it directly in your Python code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("happyhackingspace/sql-translator-llama3")
model = AutoModelForCausalLM.from_pretrained("happyhackingspace/sql-translator-llama3")
```

### Example Inference

You can test the model with a sample prompt using Transformers:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("happyhackingspace/sql-translator-llama3")
model = AutoModelForCausalLM.from_pretrained("happyhackingspace/sql-translator-llama3")

prompt = """
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction
Company database: The `employees` table contains employee_id, first_name, last_name, salary, department_id.

### Input:
SQL Prompt: Retrieve the first and last name of employees earning over 100000.

### Response:
SQL :
"""

inputs = tokenizer(prompt, return_tensors="pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Contact

For questions, suggestions, or support, feel free to open an issue or reach out via [team@happyhacking.space](mailto:team@happyhacking.space)
