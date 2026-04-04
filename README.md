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

### Key Results

After 2 500 GRPO training steps with curriculum learning, all five models converge to the 86–97% Pass@1 range:

| Model | Baseline | Post-GRPO | Δ |
|:---|:---:|:---:|:---:|
| SmolLM2-135M | 38.67% | 86.00% | **+47.33 pp** |
| SmolLM2-360M | 77.33% | 94.67% | **+17.33 pp** |
| Qwen2.5-0.5B | 93.00% | 96.33% | **+3.33 pp** |
| TinyLlama-1.1B | 73.00% | 96.33% | **+23.33 pp** |
| Gemma-2-2B | 96.00% | 97.33% | **+1.33 pp** |

## Repository Structure

```text
├── 📁 .devcontainer
│   └── ⚙️ devcontainer.json
├── 📁 .githooks
│   ├── 📝 README.md
│   └── 📄 pre-push
├── 📁 cluster
│   ├── 📄 aliases.sh
│   ├── 📄 chain_next.sh
│   ├── 📄 clean.sh
│   ├── 📄 clean_model.sh
│   ├── 📄 eval.sh
│   ├── 📄 run_all.sh
│   ├── 📄 setup.sh
│   └── 📄 train.sh
├── 📁 data
│   └── 📁 syntethic
├── 📁 docs
│   ├── 📁 papers
│   │   ├── 📕 2502.14905v1.pdf
│   │   ├── 📕 2504.13958v1.pdf
│   │   ├── 📕 2506.11027v2.pdf
│   │   └── 📕 2512.00319v2.pdf
│   ├── 📝 CLUSTER.md
│   ├── 📝 MODELS.md
│   ├── 📝 QUICK_SETUP.md
│   ├── 📝 REFERENCES.md
│   ├── 📝 REPORT.md
│   └── 📝 SLURM_COMMANDS.md
├── 📁 experiments
│   ├── 📁 configs
│   │   ├── ⚙️ baseline.yaml
│   │   ├── ⚙️ grpo_colab.yaml
│   │   ├── ⚙️ grpo_gemma2.yaml
│   │   ├── ⚙️ grpo_qwen05.yaml
│   │   ├── ⚙️ grpo_smollm2_135m.yaml
│   │   ├── ⚙️ grpo_smollm2_360m.yaml
│   │   ├── ⚙️ grpo_tinyllama.yaml
│   │   └── ⚙️ sft.yaml
│   └── 📁 logs
│       └── 📁 grpo
│           ├── 📁 gemma2-2b
│           │   ├── 📁 eval_20260404_195549
│           │   │   ├── 📁 figures
│           │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   ├── ⚙️ comparison.json
│           │   │   ├── ⚙️ completions_baseline.json
│           │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   ├── 📁 train_20260404_095349
│           │   └── ⚙️ baseline_results.json
│           ├── 📁 qwen25-05b
│           │   ├── 📁 eval_20260404_045440
│           │   │   ├── 📁 figures
│           │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   ├── ⚙️ comparison.json
│           │   │   ├── ⚙️ completions_baseline.json
│           │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   ├── 📁 train_20260404_023024
│           │   └── ⚙️ baseline_results.json
│           ├── 📁 smollm2-135m
│           │   ├── 📁 eval_20260403_213246
│           │   │   ├── 📁 figures
│           │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   ├── ⚙️ comparison.json
│           │   │   ├── ⚙️ completions_baseline.json
│           │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   ├── 📁 train_20260403_182533
│           │   └── ⚙️ baseline_results.json
│           ├── 📁 smollm2-360m
│           │   ├── 📁 eval_20260404_014114
│           │   │   ├── 📁 figures
│           │   │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│           │   │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ curriculum_progression.png
│           │   │   │   ├── 🖼️ error_evolution.png
│           │   │   │   ├── 🖼️ errors_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ errors_stage_2_progressive.png
│           │   │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ lengths_stage_2_progressive.png
│           │   │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│           │   │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│           │   │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│           │   │   │   ├── 🖼️ rescued_vs_regressed.png
│           │   │   │   └── 🖼️ stage_difficulty_heatmap.png
│           │   │   ├── ⚙️ comparison.json
│           │   │   ├── ⚙️ completions_baseline.json
│           │   │   ├── ⚙️ completions_stage_1_format_basics.json
│           │   │   ├── ⚙️ completions_stage_2_progressive.json
│           │   │   ├── ⚙️ completions_stage_3_full_difficulty.json
│           │   │   ├── ⚙️ eval_stage_1_format_basics.json
│           │   │   ├── ⚙️ eval_stage_2_progressive.json
│           │   │   └── ⚙️ eval_stage_3_full_difficulty.json
│           │   ├── 📁 train_20260403_222900
│           │   └── ⚙️ baseline_results.json
│           └── 📁 tinyllama-11b
│               ├── 📁 eval_20260404_081506
│               │   ├── 📁 figures
│               │   │   ├── 🖼️ baseline_vs_stage_1_format_basics.png
│               │   │   ├── 🖼️ baseline_vs_stage_2_progressive.png
│               │   │   ├── 🖼️ baseline_vs_stage_3_full_difficulty.png
│               │   │   ├── 🖼️ curriculum_progression.png
│               │   │   ├── 🖼️ error_evolution.png
│               │   │   ├── 🖼️ errors_stage_1_format_basics.png
│               │   │   ├── 🖼️ errors_stage_2_progressive.png
│               │   │   ├── 🖼️ errors_stage_3_full_difficulty.png
│               │   │   ├── 🖼️ lengths_stage_1_format_basics.png
│               │   │   ├── 🖼️ lengths_stage_2_progressive.png
│               │   │   ├── 🖼️ lengths_stage_3_full_difficulty.png
│               │   │   ├── 🖼️ pass_rates_stage_1_format_basics.png
│               │   │   ├── 🖼️ pass_rates_stage_2_progressive.png
│               │   │   ├── 🖼️ pass_rates_stage_3_full_difficulty.png
│               │   │   ├── 🖼️ rescued_vs_regressed.png
│               │   │   └── 🖼️ stage_difficulty_heatmap.png
│               │   ├── ⚙️ comparison.json
│               │   ├── ⚙️ completions_baseline.json
│               │   ├── ⚙️ completions_stage_1_format_basics.json
│               │   ├── ⚙️ completions_stage_2_progressive.json
│               │   ├── ⚙️ completions_stage_3_full_difficulty.json
│               │   ├── ⚙️ eval_stage_1_format_basics.json
│               │   ├── ⚙️ eval_stage_2_progressive.json
│               │   └── ⚙️ eval_stage_3_full_difficulty.json
│               ├── 📁 train_20260404_051851
│               └── ⚙️ baseline_results.json
├── 📁 notebooks
│   ├── 📁 reference
│   │   ├── 📄 Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
│   │   └── 📄 Llama3_1_(8B)_GRPO.ipynb
│   ├── 📄 01_test_config_and_train.ipynb
│   ├── 📄 02_test_config_and_train_fast.ipynb
│   └── 📄 03_full_pipeline.ipynb
├── 📁 src
│   ├── 📁 datasets
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 dataloader.py
│   │   ├── 🐍 synthetic_dataset.py
│   │   └── 🐍 templates.py
│   ├── 📁 evaluation
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 __main__.py
│   │   ├── 🐍 eval_baseline.py
│   │   ├── 🐍 eval_dataset.py
│   │   └── 🐍 eval_grpo.py
│   ├── 📁 models
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 model_loader.py
│   ├── 📁 training
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 __main__.py
│   │   ├── 🐍 callbacks.py
│   │   ├── 🐍 grpo_train.py
│   │   ├── 🐍 grpo_vanilla.py
│   │   ├── 🐍 rewards.py
│   │   └── 🐍 sft_train.py
│   ├── 📁 utils
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 chain_monitor.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 distributed.py
│   │   ├── 🐍 live_training_table.py
│   │   ├── 🐍 metrics.py
│   │   ├── 🐍 show_training_log.py
│   │   └── 🐍 visualization.py
│   └── 🐍 __init__.py
├── 📁 tests
│   ├── 🐍 __init__.py
│   └── 🐍 test_rewards.py
├── ⚙️ .dockerignore
├── ⚙️ .env.example
├── ⚙️ .gitattributes
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── 📄 LICENSE
├── 📝 README.md
├── ⚙️ docker-compose.yml
├── 📄 format.ps1
├── 📄 format.sh
├── ⚙️ pyproject.toml
├──  setup.sh
└── 📄 sync_cluster.ps1
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

### 1. Generate the Synthetic Dataset (optional)

The training and evaluation pipelines generate the dataset automatically from
the YAML config. You only need this if you want to pre-generate or inspect the
dataset independently:

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

Alternatively, the baseline is evaluated automatically when running post-training
evaluation with `--compare` (see §4).

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
# Evaluate final checkpoint vs baseline
uv run python -m src.evaluation \
    --config experiments/configs/grpo_smollm2_135m.yaml --compare

# Evaluate all curriculum stages + baseline (full analysis)
uv run python -m src.evaluation \
    --config experiments/configs/grpo_smollm2_135m.yaml --curriculum
```

### 5. Sync with Cluster (Windows)

```powershell
.\sync_cluster.ps1 -Action upload               # upload project files
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
| [`baseline.yaml`](experiments/configs/baseline.yaml) | Off-the-shelf baseline evaluation |
| [`sft.yaml`](experiments/configs/sft.yaml) | Supervised fine-tuning (experimental, not used in final results) |

## License

[MIT](LICENSE)
