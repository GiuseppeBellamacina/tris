from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.utils.visualization import (
    plot_baseline_vs_grpo_comparison,
    plot_completion_length_distribution,
    plot_completions_error_breakdown,
    plot_curriculum_progression,
    plot_error_evolution,
    plot_per_category_breakdown,
    plot_rescued_vs_regressed,
    plot_stage_difficulty_heatmap,
)

LOGS_ROOT = Path("experiments/logs/grpo")

STAGE_KEYS = [
    ("stage_1_format_basics", "Stage 1: format_basics"),
    ("stage_2_progressive", "Stage 2: progressive"),
    ("stage_3_full_difficulty", "Stage 3: full_difficulty"),
]


def process_eval_dir(eval_dir: Path, model_name: str) -> None:
    figures_dir = eval_dir / "figures"
    if not figures_dir.exists():
        return

    # --- Error breakdown per stage (pie chart) ---
    for stage_key, stage_label in STAGE_KEYS:
        compl_path = eval_dir / f"completions_{stage_key}.json"
        if not compl_path.exists():
            continue
        completions = json.loads(
            compl_path.read_text(encoding="utf-8")
        )
        fig_path = str(figures_dir / f"errors_{stage_key}.png")
        plot_completions_error_breakdown(
            completions,
            title=f"Error Breakdown — {stage_label}",
            output_path=fig_path,
        )
    # --- Pass rates per stage ---
    for stage_key, stage_label in STAGE_KEYS:
        eval_json = eval_dir / f"eval_{stage_key}.json"
        if not eval_json.exists():
            continue
        ed = json.loads(eval_json.read_text(encoding="utf-8"))
        fig_path = str(figures_dir / f"pass_rates_{stage_key}.png")
        plot_per_category_breakdown(
            ed["metrics"],
            title=f"Pass Rate \u2014 {stage_label}",
            output_path=fig_path,
        )
    # --- Completion lengths per stage ---
    for stage_key, stage_label in STAGE_KEYS:
        compl_path = eval_dir / f"completions_{stage_key}.json"
        if not compl_path.exists():
            continue
        completions = json.loads(
            compl_path.read_text(encoding="utf-8")
        )
        fig_path = str(figures_dir / f"lengths_{stage_key}.png")
        plot_completion_length_distribution(
            completions,
            title=f"Completion Lengths \u2014 {stage_label}",
            output_path=fig_path,
        )
    # --- Rescued vs Regressed (pie chart) ---
    bl_compl_path = eval_dir / "completions_baseline.json"
    # Use last stage as GRPO completions
    grpo_compl_path = None
    for stage_key, _ in reversed(STAGE_KEYS):
        candidate = eval_dir / f"completions_{stage_key}.json"
        if candidate.exists():
            grpo_compl_path = candidate
            break

    if bl_compl_path.exists() and grpo_compl_path:
        bl_compl = json.loads(
            bl_compl_path.read_text(encoding="utf-8")
        )
        grpo_compl = json.loads(
            grpo_compl_path.read_text(encoding="utf-8")
        )
        rr_path = str(figures_dir / "rescued_vs_regressed.png")
        plot_rescued_vs_regressed(
            bl_compl,
            grpo_compl,
            model_name=model_name,
            output_path=rr_path,
        )

    # --- Stage x Difficulty heatmap ---
    stage_completions: list[tuple[str, list[dict]]] = []
    if bl_compl_path.exists():
        stage_completions.append(
            (
                "Baseline",
                json.loads(bl_compl_path.read_text(encoding="utf-8")),
            )
        )
    for stage_key, stage_label in STAGE_KEYS:
        compl_path = eval_dir / f"completions_{stage_key}.json"
        if compl_path.exists():
            stage_completions.append(
                (
                    stage_label,
                    json.loads(
                        compl_path.read_text(encoding="utf-8")
                    ),
                )
            )
    if len(stage_completions) >= 2:
        hm_path = str(figures_dir / "stage_difficulty_heatmap.png")
        plot_stage_difficulty_heatmap(
            stage_completions,
            model_name=model_name,
            output_path=hm_path,
        )

        # --- Error evolution ---
        evo_path = str(figures_dir / "error_evolution.png")
        plot_error_evolution(
            stage_completions,
            model_name=model_name,
            output_path=evo_path,
        )

    # --- Curriculum progression (grouped bar + delta) ---
    # Build stage_results from eval JSON files
    baseline_path = eval_dir.parent / "baseline_results.json"
    if baseline_path.exists() and bl_compl_path.exists():
        bl_data = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        baseline_metrics = bl_data["detailed_metrics"]
        stage_results = [
            {"label": "Baseline", "metrics": baseline_metrics}
        ]

        for stage_key, stage_label in STAGE_KEYS:
            eval_json = eval_dir / f"eval_{stage_key}.json"
            if eval_json.exists():
                ed = json.loads(eval_json.read_text(encoding="utf-8"))
                stage_results.append(
                    {"label": stage_label, "metrics": ed["metrics"]}
                )

        if len(stage_results) >= 2:
            cp_path = str(figures_dir / "curriculum_progression.png")
            plot_curriculum_progression(
                stage_results,
                model_name=model_name,
                output_path=cp_path,
            )

            # Individual baseline-vs-stage charts
            for r in stage_results[1:]:
                safe = (
                    r["label"]
                    .lower()
                    .replace(" ", "_")
                    .replace(":", "")
                )
                bvs_path = str(
                    figures_dir / f"baseline_vs_{safe}.png"
                )
                plot_baseline_vs_grpo_comparison(
                    baseline_metrics=baseline_metrics,
                    grpo_metrics=r["metrics"],
                    model_name=f"{model_name} — {r['label']}",
                    output_path=bvs_path,
                )


def main() -> None:
    for model_dir in sorted(LOGS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        for child in sorted(model_dir.iterdir()):
            if child.is_dir() and (child / "figures").exists():
                print(f"\n  Processing {child.name}/")
                process_eval_dir(child, model_name)


if __name__ == "__main__":
    main()
