"""Display training log from trainer_state.json as a formatted table.

Usage:
    python -m src.utils.show_training_log experiments/checkpoints/grpo/stage_1_format_basics/checkpoint-720
    python -m src.utils.show_training_log experiments/checkpoints/grpo/stage_1_format_basics/checkpoint-720 --cols step,loss,reward,learning_rate
    python -m src.utils.show_training_log experiments/checkpoints/grpo/ --last
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default columns to show (most useful for GRPO training)
_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/format_reward/mean",
    "rewards/validity_reward/mean",
    "rewards/schema_reward/mean",
    "rewards/truncation_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]


def _find_trainer_state(path: str) -> Path | None:
    """Find trainer_state.json from a checkpoint or output dir path."""
    p = Path(path)

    # Direct file
    if p.name == "trainer_state.json" and p.exists():
        return p

    # Inside a checkpoint dir
    ts = p / "trainer_state.json"
    if ts.exists():
        return ts

    # --last: find the latest checkpoint-* in a directory
    ckpts = sorted(p.glob("checkpoint-*"))
    if ckpts:
        ts = ckpts[-1] / "trainer_state.json"
        if ts.exists():
            return ts

    # Search in stage subdirs — use the LAST stage's latest checkpoint
    stage_dirs = sorted(p.glob("stage_*"))
    if stage_dirs:
        # Iterate in reverse to find the latest stage with a checkpoint
        for stage_dir in reversed(stage_dirs):
            ckpts = sorted(stage_dir.glob("checkpoint-*"))
            if ckpts:
                ts = ckpts[-1] / "trainer_state.json"
                if ts.exists():
                    return ts

    return None


def _format_value(val: object) -> str:
    """Format a value for display."""
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def show_log(
    path: str,
    columns: list[str] | None = None,
    tail: int | None = None,
) -> None:
    ts_path = _find_trainer_state(path)
    if ts_path is None:
        print(f"No trainer_state.json found in {path}")
        return

    print(f"Source: {ts_path.parent.name}/{ts_path.name}")

    data = json.loads(ts_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])
    if not log_history:
        print("No log entries found.")
        return

    # Filter to training logs only (skip eval entries)
    train_logs = [
        e for e in log_history if "loss" in e or "reward" in e
    ]
    if not train_logs:
        print("No training log entries found.")
        return

    if tail:
        train_logs = train_logs[-tail:]

    cols = columns or _DEFAULT_COLS
    # Filter columns to those that actually exist
    available = set()
    for entry in train_logs:
        available.update(entry.keys())
    cols = [c for c in cols if c in available]

    if not cols:
        print("No matching columns found. Available:")
        for k in sorted(available):
            print(f"  {k}")
        return

    # Shorten column headers for display
    short_names = []
    for c in cols:
        # "rewards/format_reward/mean" → "format"
        if c.startswith("rewards/") and c.endswith("/mean"):
            short_names.append(c.split("/")[1].replace("_reward", ""))
        elif c == "completion_length":
            short_names.append("comp_len")
        elif c == "learning_rate":
            short_names.append("lr")
        else:
            short_names.append(c)

    # Compute column widths
    rows = []
    for entry in train_logs:
        row = [_format_value(entry.get(c)) for c in cols]
        rows.append(row)

    widths = [
        max(len(h), max(len(r[i]) for r in rows))
        for i, h in enumerate(short_names)
    ]

    # Print header
    header = " │ ".join(
        h.rjust(w) for h, w in zip(short_names, widths)
    )
    sep = "─┼─".join("─" * w for w in widths)
    print(f" {header}")
    print(f" {sep}")

    # Print rows
    for row in rows:
        line = " │ ".join(v.rjust(w) for v, w in zip(row, widths))
        print(f" {line}")

    print(
        f"\n{len(rows)} entries, global_step={data.get('global_step', '?')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show training log as table"
    )
    parser.add_argument(
        "path", help="Path to checkpoint dir or output dir"
    )
    parser.add_argument(
        "--cols",
        type=str,
        default=None,
        help="Comma-separated column names",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Show only last N entries",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="List all available columns",
    )
    args = parser.parse_args()

    if args.all_cols:
        ts_path = _find_trainer_state(args.path)
        if ts_path:
            data = json.loads(ts_path.read_text(encoding="utf-8"))
            cols = set()
            for e in data.get("log_history", []):
                cols.update(e.keys())
            print("Available columns:")
            for c in sorted(cols):
                print(f"  {c}")
        return

    columns = args.cols.split(",") if args.cols else None
    show_log(args.path, columns=columns, tail=args.tail)


if __name__ == "__main__":
    main()
