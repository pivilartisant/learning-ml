# PyTorch Project

This is a minimal PyTorch setup using the [uv](https://github.com/astral-sh/uv) package manager for fast, reproducible Python environments.

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
````

### 2. Add dependencies to `pyproject.toml`

Dependencies are managed using `uv` and listed under the `[project]` table:

```toml
[project]
dependencies = [
  "torch",
  "torchvision",
  "torchaudio"
]
```

You can create this file manually or copy from this repo.

### 3. Install dependencies

```bash
uv sync
```

### 4. Verify PyTorch installation

```bash
uv run python -c "import torch; print(torch.__version__)"
```

Expected output (example):

```
2.7.1
```

## Quick Test

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
```

## Project Structure

```
pytorch/
├── .venv/                  # Virtual environment (auto-created by uv)
├── pyproject.toml          # Dependency list
├── README.md               # This file
└── your_code.py            # Your experiments go here
```

## Notes

* Don't use `pip install` directly — use `uv pip install` or define dependencies in `pyproject.toml`.
* `torch` is a Python module, not a CLI tool. Use Python to run it.


