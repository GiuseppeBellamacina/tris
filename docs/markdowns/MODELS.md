# Models

## Selected Models

All five models are trained under identical settings: 4-bit NF4 quantisation,
bfloat16 compute, LoRA r=16 / α=32 on all linear projections, 3-stage
curriculum (2 500 steps total), single L40S GPU.

| Model | Params | Architecture | Config |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | LLaMA-like | `grpo_smollm2_135m.yaml` |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | LLaMA-like | `grpo_smollm2_360m.yaml` |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Qwen 2.5 | `grpo_qwen05.yaml` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | LLaMA 2 | `grpo_tinyllama.yaml` |
| `google/gemma-2-2b-it` | 2B | Gemma 2 | `grpo_gemma2.yaml` |

## Results

| Model | Baseline Pass@1 | GRPO Pass@1 | Δ |
|---|---|---|---|
| SmolLM2-135M | — | — | — |
| SmolLM2-360M | — | — | — |
| Qwen2.5-0.5B | — | — | — |
| TinyLlama-1.1B | — | — | — |
| Gemma-2-2B | — | — | — |

> Results will be filled as training runs complete on the cluster.

## Selection Rationale

Models were chosen to span the 135M–2B parameter range across four distinct
architectures, enabling a systematic study of how model capacity affects GRPO
alignment for structured output:

- **SmolLM2 135M/360M**: Smallest available instruct models — tests the lower
  bound of what GRPO can teach.
- **Qwen2.5-0.5B**: Modern architecture with a strong tokenizer; intermediate
  point between SmolLM2-360M and TinyLlama.
- **TinyLlama-1.1B**: Well-known baseline with LLaMA 2 architecture,
  widely used in RL fine-tuning experiments.
- **Gemma-2-2B**: Largest model in the set; tests whether larger capacity
  yields diminishing returns with the same curriculum.

## Notes

- All models support 4-bit NF4 quantisation and LoRA
- Gemma-2 lacks a system role in its chat template — the pipeline automatically
  merges the system prompt into the user message
- With a 22.5 GB GPU shard (L40S gpu-xlarge), models up to ~3B in 4-bit fit
  comfortably
- Gated models (Llama 3.2) were excluded to keep the pipeline fully
  reproducible without HuggingFace licence gates
