"""GRPO training script using trl GRPOTrainer.

Usage:
    python -m src.training.grpo_train --config experiments/configs/grpo.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

from trl import GRPOConfig, GRPOTrainer

from datasets import Dataset
from src.datasets.dataloader import (
    format_prompt_for_model,
    load_config,
    load_synthetic_dataset,
)
from src.models.model_loader import load_model_and_tokenizer
from src.training.rewards import combined_reward


def build_reward_fn(reward_cfg: dict):
    """Build a reward function from config, compatible with GRPOTrainer.

    GRPOTrainer calls: reward_fn(prompts, completions, **kwargs) -> list[float]
    """
    partial_credit = reward_cfg.get("partial_credit", False)
    reasoning_bonus = reward_cfg.get("reasoning_bonus", 0.0)

    def reward_fn(completions, **kwargs):
        """Reward function for GRPOTrainer.

        Args:
            completions: list of completion strings.
            **kwargs: additional keyword arguments (e.g., prompts).

        Returns:
            list of float rewards.
        """
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            r = combined_reward(
                text,
                task_type="json",
                partial_credit=partial_credit,
                reasoning_bonus=reasoning_bonus,
            )
            rewards.append(r)
        return rewards

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for strict code/JSON generation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    grpo_cfg = config["grpo"]
    reward_cfg = config["reward"]

    # Load model and tokenizer (Unsloth viene gestito da model_loader)
    print(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)

    # Load and prepare dataset
    print("Loading dataset...")
    ds = load_synthetic_dataset(
        path=config["dataset"]["path"],
        split=config["dataset"].get("split", "train"),
        max_samples=config["dataset"].get("max_samples"),
    )
    train_ds = ds[config["dataset"].get("split", "train")]

    # Format prompts for the model
    formatted_prompts = []
    for i in range(len(train_ds)):
        sample = train_ds[i]
        prompt = format_prompt_for_model(sample, tokenizer)
        formatted_prompts.append({"prompt": prompt, "task_type": sample["task_type"]})

    prompt_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in formatted_prompts])

    # Build reward function
    reward_fn = build_reward_fn(reward_cfg)

    # Configure GRPO
    output_dir = training_cfg.get("output_dir", "experiments/checkpoints/grpo")
    log_dir = training_cfg.get("log_dir", "experiments/logs/grpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "grpo-strict-generation"),
        name=wandb_cfg.get("run_name", "grpo-train"),
        config=config,
        tags=wandb_cfg.get("tags", ["grpo", config["model"]["name"].split("/")[-1]]),
    )

    # Warmup: supporta sia warmup_steps che warmup_ratio
    warmup_kwargs = {}
    if "warmup_ratio" in training_cfg:
        warmup_kwargs["warmup_ratio"] = training_cfg["warmup_ratio"]
    else:
        warmup_kwargs["warmup_steps"] = training_cfg.get("warmup_steps", 50)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=training_cfg.get("max_steps", 1000),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        **warmup_kwargs,
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
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
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))  # type: ignore[union-attr]
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
