# Align a Small LLM with GRPO for Strict JSON Generation

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![References](https://img.shields.io/badge/References-REFERENCES.md-green)](docs/REFERENCES.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Information

| Field | Value |
|---|---|
| **Project ID** | 23 |
| **Track** | Align a Small LLM with GRPO for Strict Code or JSON Generation |
| **Module** | Reinforcement Learning |

## Overview

This project applies **Group Relative Policy Optimization (GRPO)** to fine-tune
five small LLMs (135M–2B parameters) so that they generate **syntactically valid,
schema-conformant JSON**. Instead of a neural reward model, five rule-based reward
components score each completion (format, validity, schema, truncation, reasoning),
providing a dense additive signal.

Training uses a **3-stage curriculum** that progressively shifts difficulty from
simple to hard prompts across 2 500 training steps, with **4-bit NF4 quantization**
and **LoRA** (r=16) on a single NVIDIA L40S GPU via the DMI UniCT cluster.

### Models

| Model | Parameters | Architecture |
|---|---|---|
| SmolLM2-135M-Instruct | 135M | LLaMA-like |
| SmolLM2-360M-Instruct | 360M | LLaMA-like |
| Qwen2.5-0.5B-Instruct | 0.5B | Qwen2.5 |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | LLaMA 2 |
| Gemma-2-2B-it | 2B | Gemma 2 |

> For theoretical details, ablations, and results see **[REPORT.md](docs/REPORT.md)**.

## Repository Structure

```
src/
  datasets/          Synthetic dataset generation and prompt formatting
  models/            Model/tokenizer loading (HuggingFace + LoRA backends)
  training/          GRPO and SFT training loops, reward functions, callbacks
  evaluation/        Baseline and post-training evaluation
  utils/             Config, metrics, visualization, pipeline monitor
experiments/
  configs/           Per-model YAML configs (GRPO, baseline, SFT)
  logs/              Training logs, wandb offline runs, eval figures
  checkpoints/       Saved LoRA adapters and trainer state
notebooks/           Colab notebooks (full pipeline + reference implementations)
docs/                Final report, references, and cluster guide
  papers/            Reference papers (PDF)
cluster/             SLURM scripts, aliases, cleanup tools
data/                Synthetic dataset (generated, not committed)
```

## Setup

**Prerequisites**: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GiuseppeBellamacina/grpo-strict-generation.git
cd grpo-strict-generation

pip install uv          # if not already installed
uv sync                 # core dependencies
uv sync --extra dev     # + ruff, pytest, black
```

**Cluster setup**: see [docs/QUICK_SETUP.md](docs/QUICK_SETUP.md) for
step-by-step instructions or [docs/CLUSTER.md](docs/CLUSTER.md) for the
full guide.

## Usage

### 1. Generate the Synthetic Dataset

```bash
uv run python -m src.datasets.synthetic_dataset \
    --output data/synthetic \
    --num_samples 5000 \
    --test_ratio 0.2
```

### 2. Baseline Evaluation

Evaluate off-the-shelf models without any fine-tuning:

```bash
uv run python -m src.evaluation \
    --config experiments/configs/baseline.yaml
```

### 3. GRPO Training (Curriculum)

Each model has its own config. Training runs a 3-stage curriculum automatically:

```bash
# Single model
uv run python -m src.training \
    --config experiments/configs/grpo_smollm2_135m.yaml

# Resume from checkpoint
uv run python -m src.training \
    --config experiments/configs/grpo_smollm2_135m.yaml --resume
```

On the cluster, use the multi-model chain pipeline:
```bash
run-all                    # train + eval all 5 models sequentially
run-all --models=1,2,3     # specific models only
monitor                    # live dashboard (compact)
monitor --tab              # full job table
```

### 4. Post-Training Evaluation

```bash
uv run python -m src.evaluation \
    --config experiments/configs/grpo_smollm2_135m.yaml --compare
```

### 5. Sync with Cluster (Windows)

```powershell
.\sync_cluster.ps1 -Action upload              # upload project files
.\sync_cluster.ps1 -Action download             # download all results
.\sync_cluster.ps1 -Action download-logs        # logs + figures only
.\sync_cluster.ps1 -Action download-checkpoints # LoRA adapters
.\sync_cluster.ps1 -Action download-wandb       # wandb offline runs
```

## Reward Function

Five **additive** reward components score each completion; weights sum to 1.0.
Reasoning is disabled by default (`thinking: false`) and its weight is
redistributed to the other components.

| Component | Weight | Description |
|---|---|---|
| **Format** | 0.25 | Presence of a ` ```json ... ``` ` code block (partial credit for generic ` ``` `) |
| **Validity** | 0.30 | JSON parseable by `json.loads` (graded score) |
| **Schema** | 0.30 | Structural conformance to prompt constraints (keys, types, counts) |
| **Truncation** | 0.15 | Penalises completions that hit `max_completion_length` mid-token |
| **Reasoning** | 0.00 | `<think>…</think>` block with real content (disabled, weight = 0) |

## Configs

Each model has a dedicated GRPO config specifying its HuggingFace ID, chat
template, and per-model hyperparameters. Curriculum stages and reward weights
are shared across all configs.

| Config | Purpose |
|---|---|
| [`grpo_smollm2_135m.yaml`](experiments/configs/grpo_smollm2_135m.yaml) | GRPO — SmolLM2-135M-Instruct |
| [`grpo_smollm2_360m.yaml`](experiments/configs/grpo_smollm2_360m.yaml) | GRPO — SmolLM2-360M-Instruct |
| [`grpo_qwen05.yaml`](experiments/configs/grpo_qwen05.yaml) | GRPO — Qwen2.5-0.5B-Instruct |
| [`grpo_tinyllama.yaml`](experiments/configs/grpo_tinyllama.yaml) | GRPO — TinyLlama-1.1B-Chat-v1.0 |
| [`grpo_gemma2.yaml`](experiments/configs/grpo_gemma2.yaml) | GRPO — Gemma-2-2B-it |
| [`baseline.yaml`](experiments/configs/baseline.yaml) | Off-the-shelf evaluation (all models) |
| [`sft.yaml`](experiments/configs/sft.yaml) | Supervised fine-tuning (comparison) |

## License

[MIT](LICENSE)
