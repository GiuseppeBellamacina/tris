#!/usr/bin/env python3
"""Parse HuggingFace Trainer log lines from stdin and display as a live table.

Usage:
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table --cols step,reward,loss
    tail -f logs/slurm-train-1234.log | python -u -m src.utils.live_training_table --rows 30
"""

from __future__ import annotations

import ast
import os
import re
import sys
from collections import deque

_DEFAULT_COLS = [
    "step",
    "loss",
    "reward",
    "reward_std",
    "rewards/format_reward/mean",
    "rewards/validity_reward/mean",
    "rewards/schema_reward/mean",
    "rewards/reasoning_reward/mean",
    "rewards/truncation_reward/mean",
    "completion_length",
    "learning_rate",
    "grad_norm",
]

_SHORT_NAMES = {
    "rewards/format_reward/mean": "format",
    "rewards/validity_reward/mean": "validity",
    "rewards/schema_reward/mean": "schema",
    "rewards/reasoning_reward/mean": "reasoning",
    "rewards/truncation_reward/mean": "truncation",
    "completion_length": "comp_len",
    "learning_rate": "lr",
}

_DICT_PATTERN = re.compile(r"\{.*\}")
# Matches "step=240  loss=0.001  reward=0.55 ..." key=value lines
_KV_PATTERN = re.compile(r"step=\d+")


def _parse_kv_line(line: str) -> dict | None:
    """Parse 'key=value  key=value ...' formatted log lines."""
    if not _KV_PATTERN.search(line):
        return None
    # Extract the key=value portion (after any prefix like timestamps/progress bars)
    m = re.search(r"(step=\d+.*)", line)
    if not m:
        return None
    kv_part = m.group(1)
    entry: dict = {}
    for pair in re.finditer(
        r"([\w/]+)=([-+]?\d+\.?\d*(?:e[+-]?\d+)?)", kv_part
    ):
        key, val = pair.group(1), pair.group(2)
        try:
            entry[key] = float(val)
        except ValueError:
            entry[key] = val
    return entry if "step" in entry else None


def _format_val(key: str, val: object) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if key == "step":
            return str(int(val))
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def _clear() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def _redraw(
    header: str,
    separator: str,
    rows: deque[str],
) -> None:
    """Clear and redraw the full display."""
    _clear()
    print(f" {header}")
    print(f" {separator}")
    for row in rows:
        print(f" {row}")
    sys.stdout.flush()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=str, default=None)
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of metric rows to keep visible (default: 20)",
    )
    args = parser.parse_args()

    cols = args.cols.split(",") if args.cols else _DEFAULT_COLS
    max_rows = args.rows

    # State
    header = ""
    separator = ""
    metric_rows: deque[str] = deque(maxlen=max_rows)
    widths: list[int] = []
    active_cols: list[str] = []
    header_ready = False

    # Skip completion sample blocks printed by CompletionSampleCallback
    in_sample_block = False
    pending_separator = False

    try:
        for line in sys.stdin:
            line = line.rstrip("\n\r")
            stripped = line.strip()

            # ── Capture completion sample blocks ──────────────────────
            # Block structure printed by CompletionSampleCallback:
            #   ══════...══════   (open)
            #   COMPLETION SAMPLES
            #   ══════...══════
            #   ... sample lines ...
            #   ══════...══════   (close)

            is_separator = stripped.startswith("═" * 10)

            if in_sample_block:
                # Inside a sample block — skip all lines until closing ═══
                if is_separator:
                    in_sample_block = False
                    pending_separator = False
                continue

            if is_separator and not in_sample_block:
                pending_separator = True
                continue

            if pending_separator and "COMPLETION SAMPLES" in stripped:
                in_sample_block = True
                pending_separator = False
                continue

            # It was a stray separator, ignore
            pending_separator = False

            # ── Parse metric lines ────────────────────────────────────
            entry = None

            # Try key=value format first (never truncated by SLURM)
            entry = _parse_kv_line(stripped)

            # Fall back to dict format
            if entry is None:
                m = _DICT_PATTERN.search(stripped)
                if m:
                    try:
                        entry = ast.literal_eval(m.group(0))
                    except (ValueError, SyntaxError):
                        pass

            if not entry or "step" not in entry:
                continue

            # Filter to requested columns that exist
            current_active = [c for c in cols if c in entry]
            if not current_active:
                continue

            # Build header on first metric line (or if columns changed)
            if not header_ready or current_active != active_cols:
                active_cols = current_active
                short_names = [
                    _SHORT_NAMES.get(c, c) for c in active_cols
                ]
                widths = [max(8, len(s)) for s in short_names]
                header = " │ ".join(
                    s.rjust(w) for s, w in zip(short_names, widths)
                )
                separator = "─┼─".join("─" * w for w in widths)
                header_ready = True

            vals = [_format_val(c, entry.get(c)) for c in active_cols]
            # Update widths if needed
            for i, v in enumerate(vals):
                if len(v) > widths[i]:
                    widths[i] = len(v)
            row = " │ ".join(v.rjust(w) for v, w in zip(vals, widths))
            metric_rows.append(row)

            # Redraw
            _redraw(header, separator, metric_rows)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
