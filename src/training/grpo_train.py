"""GRPO training script using trl GRPOTrainer.

Usage:
    python -m src.training.grpo_train --config experiments/configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer

from datasets import Dataset
from src.datasets.dataloader import format_prompt_for_model, load_synthetic_dataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.rewards import build_reward_function
from src.utils.config import load_config

load_dotenv()


def _build_grpo_config(training_cfg: dict[str, Any], grpo_cfg: dict[str, Any]) -> GRPOConfig:
    """Build a ``GRPOConfig`` from separated training and GRPO config dicts."""
    output_dir = training_cfg.get("output_dir", "experiments/checkpoints/grpo")
    log_dir = training_cfg.get("log_dir", "experiments/logs/grpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Warmup: supports both warmup_steps and warmup_ratio
    warmup_kwargs: dict[str, Any] = {}
    if "warmup_ratio" in training_cfg:
        warmup_kwargs["warmup_ratio"] = training_cfg["warmup_ratio"]
    else:
        warmup_kwargs["warmup_steps"] = training_cfg.get("warmup_steps", 50)

    return GRPOConfig(
        output_dir=output_dir,
        max_steps=training_cfg.get("max_steps", 1000),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=training_cfg.get("weight_decay", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.1),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        logging_dir=log_dir,
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 512),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 256),
        beta=grpo_cfg.get("beta", 0.04),
        temperature=grpo_cfg.get("temperature", 0.7),
        report_to="wandb",
    )


def _prepare_prompt_dataset(config: dict[str, Any], tokenizer: Any) -> Dataset:
    """Load the synthetic dataset and format prompts for the model."""
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "train"),
        max_samples=config["dataset"].get("max_samples"),
    )
    train_ds = ds[config["dataset"].get("split", "train")]

    formatted = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        formatted.append({"prompt": prompt})

    return Dataset.from_list(formatted)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for strict code/JSON generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # Enable vLLM standby if fast_inference is requested
    if config.get("model", {}).get("fast_inference", False):
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

    # Load model and tokenizer
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Prepare dataset
    print("Loading dataset...")
    prompt_dataset = _prepare_prompt_dataset(config, tokenizer)

    # Build reward function
    thinking = config.get("dataset", {}).get("thinking", True)
    reward_fn = build_reward_function(config["reward"], thinking=thinking)

    # Build GRPO config
    grpo_config = _build_grpo_config(config["training"], config["grpo"])

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "grpo-strict-generation")
    os.environ["WANDB_PROJECT"] = wandb_project
    wandb.init(
        project=wandb_project,
        name=wandb_cfg.get("run_name", "grpo-train"),
        config=config,
        tags=wandb_cfg.get("tags", ["grpo", config["model"]["name"].split("/")[-1]]),
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,  # type: ignore[arg-type]
        args=grpo_config,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fn,  # type: ignore[arg-type]
        processing_class=tokenizer,  # type: ignore[arg-type]
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    final_path = Path(grpo_config.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))  # type: ignore[union-attr]
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
