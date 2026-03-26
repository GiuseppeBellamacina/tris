"""Reward functions for GRPO training.

All rewards are rule-based (no neural reward model):
  - JSON: validated via json.loads
  - Reasoning bonus: checks for <think>...</think> tags

Security: NEVER uses exec() or eval(). Only static parsing.
"""

from __future__ import annotations

import json
import re


def extract_code_block(text: str, language: str) -> str | None:
    """Extract the first fenced code block of the given language from text.

    Looks for ```language ... ``` patterns. Falls back to the first
    generic ``` ... ``` block if no language-specific block is found.
    """
    # Try language-specific block first
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: generic fenced block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: if the entire output looks like raw JSON without fences
    stripped = text.strip()
    if language == "json" and (stripped.startswith("{") or stripped.startswith("[")):
        return stripped

    return None


def json_reward(completion: str, partial_credit: bool = False) -> float:
    """Compute reward for JSON generation.

    Args:
        completion: The model's full completion text.
        partial_credit: If True, use a graduated scale instead of binary 0/1.

    Returns:
        If partial_credit is False: 1.0 (valid) or 0.0 (invalid).
        If partial_credit is True, graduated scale:
          1.00 — valid JSON that parses successfully
          0.75 — has code block, looks like JSON structure, error near end (>70%)
          0.50 — has code block, looks like JSON structure, error in middle
          0.25 — has code block but content doesn't look like JSON at all
          0.00 — no code block found
    """
    code = extract_code_block(completion, "json")
    if code is None:
        return 0.0

    try:
        json.loads(code)
        return 1.0
    except json.JSONDecodeError as e:
        if not partial_credit:
            return 0.0

        # Check if the content has JSON-like structure (braces/brackets, colons, quotes)
        has_braces = "{" in code or "[" in code
        has_colons = ":" in code
        has_quotes = '"' in code
        looks_like_json = has_braces and (has_colons or has_quotes)

        if not looks_like_json:
            return 0.25

        # Error position ratio: how far into the text the first error occurs
        error_pos_ratio = e.pos / max(len(code), 1) if hasattr(e, "pos") and e.pos else 0.0

        if error_pos_ratio > 0.7:
            return 0.75  # almost valid — likely a trailing comma or missing brace
        return 0.5  # structural but broken earlier


def reasoning_reward(completion: str) -> float:
    """Bonus reward for including reasoning before the code block.

    Checks for <think>...</think> tags containing non-trivial content.

    Returns:
        0.2 if reasoning tags found with content, 0.0 otherwise.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Require at least 20 chars of reasoning to count
        if len(content) >= 20:
            return 0.2
    return 0.0


def combined_reward(
    completion: str,
    task_type: str = "json",
    partial_credit: bool = False,
    reasoning_bonus: float = 0.0,
) -> float:
    """Compute the combined reward for a completion.

    Args:
        completion: The model's full completion text.
        task_type: Must be "json".
        partial_credit: Whether to use partial credit scoring.
        reasoning_bonus: If > 0, add reasoning_reward scaled by this factor.

    Returns:
        Total reward in [0.0, 1.0 + reasoning_bonus].
    """
    if task_type != "json":
        raise ValueError(f"Unknown task_type: {task_type}. Only 'json' is supported.")

    base = json_reward(completion, partial_credit=partial_credit)

    if reasoning_bonus > 0:
        base += reasoning_reward(completion) * (reasoning_bonus / 0.2)

    return base


def build_reward_function(
    partial_credit: bool = False,
    reasoning_bonus: float = 0.0,
):
    """Build a reward function compatible with trl GRPOTrainer.

    GRPOTrainer expects: reward_fn(completions: list[str], **kwargs) -> list[float]
    All tasks are JSON-only.
    """

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            r = combined_reward(
                completion,
                task_type="json",
                partial_credit=partial_credit,
                reasoning_bonus=reasoning_bonus,
            )
            rewards.append(r)
        return rewards

    return reward_fn
