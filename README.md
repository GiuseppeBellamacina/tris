# Align a Small LLM with GRPO for Strict Code/JSON Generation

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Group and Project Information

- **Project ID**: 23
- **Track**: Align a Small LLM with GRPO for Strict Code or JSON Generation
- **Reference Module**: Reinforcement Learning
- **Suggested Size**: Large

## Project Description

This project applies **Group Relative Policy Optimization (GRPO)** to fine-tune a small open-weight LLM (~0.5B–1.5B parameters) so that it generates **syntactically valid JSON and Python code**. Instead of using a neural reward model, rewards are computed programmatically via `json.loads` and `ast.parse`, providing a strict binary signal. The model is trained with LoRA/PEFT for memory efficiency and evaluated on Pass@k metrics comparing pre- and post-training syntactic adherence.

> **Official Report**: For all theoretical details, performance analysis, the architecture used, and contributions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

## Technical Reproducibility

### 1. Environment Setup

**Prerequisites**: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Install uv (if not already installed)
pip install uv

# Create virtual environment and install all dependencies
uv sync

# Include dev tools (ruff, pytest, ipykernel)
uv sync --extra dev
```

**Dataset**: The synthetic dataset is generated programmatically. No external download is needed.

```bash
uv run python -m src.datasets.synthetic_dataset --output data/synthetic --num_samples 5000
```

### 2. Network Training

**Baseline Evaluation** (no training, off-the-shelf model):

```bash
uv run python -m src.evaluation.baseline_eval --config experiments/configs/baseline.yaml
```

**GRPO Training**:

```bash
uv run python -m src.training.grpo_train --config experiments/configs/grpo.yaml
```

**SFT Training** (for comparison):

```bash
uv run python -m src.training.sft_train --config experiments/configs/sft.yaml
```

### 3. Evaluation

```bash
uv run python -m src.evaluation.evaluate --config experiments/configs/grpo.yaml
```

---

_For the declaration of individual tasks and the use of AI, refer to `docs/REPORT.md`._
