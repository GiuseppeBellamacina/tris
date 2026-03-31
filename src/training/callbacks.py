"""Custom TrainerCallbacks for GRPO training."""

from __future__ import annotations

from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class HighPrecisionLogCallback(TrainerCallback):
    """Print training metrics with higher float precision (8 decimal places).

    The default HuggingFace Trainer formats floats to 6 decimal places, which
    causes very small loss values (e.g. GRPO policy gradient loss) to appear
    as ``-0.000000``.  This callback reprints every ``on_log`` event to stdout
    with enough precision to see the actual values.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero or not logs:
            return
        parts = [f"step={state.global_step}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.8f}" if isinstance(v, float) else f"{k}={v}")
        print("  " + "  ".join(parts))
