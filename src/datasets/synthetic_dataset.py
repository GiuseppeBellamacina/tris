"""Synthetic dataset generator for strict JSON generation tasks.

Generates prompt-instruction pairs at three difficulty levels (simple, medium, hard)
for JSON tasks. Used as the training/eval dataset for GRPO alignment.

Usage:
    python -m src.datasets.synthetic_dataset --output data/synthetic --num_samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# JSON prompt templates — Simple
# ---------------------------------------------------------------------------

_JSON_SIMPLE: list[dict] = [
    {
        "instruction": (
            "Generate a valid JSON object with the following keys: "
            '"{k1}" (string), "{k2}" (integer), "{k3}" (boolean).'
        ),
        "params": lambda: {
            "k1": random.choice(["name", "title", "label", "city", "color", "brand", "category"]),
            "k2": random.choice(["age", "count", "score", "year", "quantity", "rating", "price"]),
            "k3": random.choice(["active", "verified", "enabled", "visible", "published", "premium", "available"]),
        },
    },
    {
        "instruction": "Generate a JSON array containing exactly {n} strings representing {topic}.",
        "params": lambda: {
            "n": random.randint(3, 7),
            "topic": random.choice(
                [
                    "fruit names",
                    "country names",
                    "programming languages",
                    "animal species",
                    "planet names",
                    "car brands",
                    "musical instruments",
                    "European capital cities",
                    "file extensions",
                    "color names",
                ]
            ),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a key "{k1}" (string) and a key ' '"{k2}" which is an array of {n} integers.'
        ),
        "params": lambda: {
            "k1": random.choice(["id", "name", "label", "code", "tag"]),
            "k2": random.choice(["values", "scores", "data", "items", "measurements"]),
            "n": random.randint(3, 6),
        },
    },
    {
        "instruction": (
            "Generate a JSON object with exactly {n} key-value pairs where all "
            "values are of type {vtype}. Use descriptive key names related to {topic}."
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
            "vtype": random.choice(["string", "integer", "boolean"]),
            "topic": random.choice(["weather", "food", "sports", "music", "travel", "health"]),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a simple {entity} "
            'with keys "{k1}" (string), "{k2}" ({t2}), and "{k3}" ({t3}).'
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "contact card",
                    "book entry",
                    "to-do item",
                    "event",
                    "bookmark",
                    "notification",
                    "tag",
                    "setting",
                    "preference",
                ]
            ),
            "k1": random.choice(["name", "title", "description", "label"]),
            "k2": random.choice(["priority", "order", "level", "index"]),
            "t2": random.choice(["integer", "number"]),
            "k3": random.choice(["done", "read", "starred", "archived", "pinned"]),
            "t3": "boolean",
        },
    },
    {
        "instruction": (
            "Generate a JSON array of exactly {n} objects, each containing "
            'only a "{k1}" (string) and a "{k2}" (integer).'
        ),
        "params": lambda: {
            "n": random.randint(2, 5),
            "k1": random.choice(["name", "item", "label", "title", "city"]),
            "k2": random.choice(["value", "count", "score", "amount", "population"]),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string) and a "{k2}" key '
            "containing a flat array of {n} {elem_type}."
        ),
        "params": lambda: {
            "k1": random.choice(["category", "group", "type", "section"]),
            "k2": random.choice(["items", "elements", "entries", "members"]),
            "n": random.randint(3, 7),
            "elem_type": random.choice(["strings", "integers", "booleans"]),
        },
    },
    {
        "instruction": (
            "Generate a valid JSON object representing a key-value mapping "
            "of {n} {domain} abbreviations to their full names."
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
            "domain": random.choice(
                [
                    "country",
                    "US state",
                    "HTTP status code",
                    "chemical element",
                    "currency",
                    "time zone",
                    "unit of measurement",
                ]
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# JSON prompt templates — Medium
# ---------------------------------------------------------------------------

_JSON_MEDIUM: list[dict] = [
    {
        "instruction": (
            "Generate a JSON array of {n} objects, each with keys "
            '"{k1}" (string), "{k2}" (integer), and a nested object '
            '"{k3}" containing "{nk1}" (string) and "{nk2}" (string).'
        ),
        "params": lambda: {
            "n": random.randint(2, 4),
            "k1": random.choice(["name", "title", "username", "product", "label"]),
            "k2": random.choice(["age", "id", "score", "quantity", "price"]),
            "k3": random.choice(["address", "contact", "location", "details", "metadata"]),
            "nk1": random.choice(["street", "city", "email", "line1", "type"]),
            "nk2": random.choice(["zip", "country", "phone", "region", "code"]),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with at least {n} fields, "
            "including one nested object and one array field."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "user profile",
                    "product listing",
                    "blog post",
                    "movie record",
                    "employee record",
                    "restaurant menu item",
                    "flight booking",
                    "music album",
                    "course syllabus",
                    "recipe",
                ]
            ),
            "n": random.randint(5, 8),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a configuration file for a {app_type} "
            "with keys for {k1}, {k2}, and a nested {k3} section."
        ),
        "params": lambda: {
            "app_type": random.choice(
                [
                    "web server",
                    "database",
                    "logging service",
                    "cache layer",
                    "message queue",
                    "API gateway",
                    "monitoring agent",
                ]
            ),
            "k1": random.choice(["host", "port", "name", "endpoint"]),
            "k2": random.choice(["timeout", "retries", "max_connections", "buffer_size"]),
            "k3": random.choice(["auth", "ssl", "logging", "metrics", "cors"]),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} that includes "
            "a list of {n} {sub_entity}, each with at least 3 fields."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "classroom",
                    "shopping cart",
                    "playlist",
                    "project board",
                    "team roster",
                    "menu",
                    "itinerary",
                    "inventory",
                ]
            ),
            "n": random.randint(2, 5),
            "sub_entity": random.choice(
                ["students", "items", "tracks", "tasks", "members", "dishes", "stops", "products"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {doc_type} with the following "
            'sections: "{s1}", "{s2}", and "{s3}". Each section should be a nested '
            "object with at least 2 fields."
        ),
        "params": lambda: {
            "doc_type": random.choice(
                ["report", "invoice", "contract", "specification", "manifest", "log entry", "audit record"]
            ),
            "s1": random.choice(["header", "metadata", "summary", "info"]),
            "s2": random.choice(["body", "content", "details", "payload"]),
            "s3": random.choice(["footer", "signature", "status", "notes"]),
        },
    },
    {
        "instruction": (
            'Generate a JSON object with a "{k1}" key (string), a "{k2}" key '
            '(ISO 8601 date string), and a "{k3}" key containing an array of '
            'objects with "{nk1}" (string) and "{nk2}" (number) fields.'
        ),
        "params": lambda: {
            "k1": random.choice(["title", "name", "event", "project"]),
            "k2": random.choice(["created_at", "due_date", "start_date", "timestamp"]),
            "k3": random.choice(["entries", "line_items", "records", "milestones"]),
            "nk1": random.choice(["description", "label", "name", "note"]),
            "nk2": random.choice(["amount", "hours", "progress", "value"]),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {form_type} form schema with {n} fields. "
            'Each field should have "label" (string), "type" (one of "text", "number", '
            '"email", "select"), and "required" (boolean).'
        ),
        "params": lambda: {
            "form_type": random.choice(
                ["registration", "contact", "survey", "feedback", "checkout", "onboarding", "support ticket"]
            ),
            "n": random.randint(4, 7),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {entity} with an "
            '"id" (integer), "name" (string), "tags" (array of strings), '
            'and a "metadata" nested object with at least 3 key-value pairs.'
        ),
        "params": lambda: {
            "entity": random.choice(
                ["document", "asset", "resource", "artifact", "dataset", "model", "deployment", "experiment"]
            ),
        },
    },
]

# ---------------------------------------------------------------------------
# JSON prompt templates — Hard
# ---------------------------------------------------------------------------

_JSON_HARD: list[dict] = [
    {
        "instruction": (
            "Generate a JSON Schema (draft-07) that validates objects representing "
            "a {entity}. The schema must include required fields, type constraints, "
            "a nested object property, and an array property with item validation."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "REST API error response",
                    "e-commerce order",
                    "weather forecast",
                    "user registration form",
                    "CI/CD pipeline definition",
                    "IoT sensor reading",
                    "GraphQL query result",
                    "OAuth2 token response",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a paginated API response for a "
            "list of {entity}. Include metadata fields (page, per_page, total, "
            "total_pages) and a results array with {n} items, each having at "
            "least {f} fields including one nested object."
        ),
        "params": lambda: {
            "entity": random.choice(
                [
                    "users",
                    "products",
                    "articles",
                    "transactions",
                    "repositories",
                    "notifications",
                    "search results",
                ]
            ),
            "n": random.randint(2, 4),
            "f": random.randint(4, 6),
        },
    },
    {
        "instruction": (
            "Generate a deeply nested JSON object (at least 4 levels of nesting) "
            "representing a {domain} hierarchy. Each level should have a "
            '"name" (string), "id" (integer), and "children" (array of sub-objects).'
        ),
        "params": lambda: {
            "domain": random.choice(
                [
                    "company organizational chart",
                    "file system directory tree",
                    "category taxonomy",
                    "geographical region breakdown",
                    "menu navigation structure",
                    "permission role hierarchy",
                ]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a full {api_type} API endpoint "
            "specification (OpenAPI-style) including path, method, parameters "
            "(with types and required flags), request body schema, and "
            "response schema with example values."
        ),
        "params": lambda: {
            "api_type": random.choice(
                ["user management", "payment processing", "search", "file upload", "authentication", "notification"]
            ),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {workflow} workflow definition "
            'with at least {n} steps. Each step must have an "id", "name", '
            '"type" (one of "action", "condition", "loop"), "config" '
            '(nested object), and "next" (string or null).'
        ),
        "params": lambda: {
            "workflow": random.choice(
                [
                    "data pipeline",
                    "CI/CD build",
                    "order fulfillment",
                    "user onboarding",
                    "content moderation",
                    "ETL process",
                ]
            ),
            "n": random.randint(4, 6),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a dashboard configuration with "
            '{n} widgets. Each widget has "id", "type" (chart, table, metric, '
            'or map), "title", "position" (object with x, y, w, h), and a '
            '"data_source" object with "endpoint" (string), "params" (object), '
            'and "refresh_interval" (integer).'
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a {domain} event conforming "
            "to the CloudEvents specification (version 1.0). Include specversion, "
            "id, source, type, subject, time, datacontenttype, and a nested data "
            "object with at least {n} domain-specific fields."
        ),
        "params": lambda: {
            "domain": random.choice(
                [
                    "e-commerce purchase",
                    "IoT temperature alert",
                    "user authentication",
                    "deployment completed",
                    "payment failed",
                    "inventory low",
                ]
            ),
            "n": random.randint(4, 7),
        },
    },
    {
        "instruction": (
            "Generate a JSON object representing a database migration plan with "
            '{n} operations. Each operation should be an object with "order" '
            '(integer), "type" (one of "create_table", "add_column", '
            '"create_index", "add_constraint"), "table" (string), and '
            '"definition" (nested object describing the change in detail).'
        ),
        "params": lambda: {
            "n": random.randint(3, 6),
        },
    },
]

# ---------------------------------------------------------------------------
# All pools with metadata
# ---------------------------------------------------------------------------

TASK_POOLS = {
    "json_simple": {"templates": _JSON_SIMPLE, "task_type": "json", "difficulty": "simple"},
    "json_medium": {"templates": _JSON_MEDIUM, "task_type": "json", "difficulty": "medium"},
    "json_hard": {"templates": _JSON_HARD, "task_type": "json", "difficulty": "hard"},
}

# Difficulty distribution weights: more simple/medium, fewer hard
DIFFICULTY_WEIGHTS = {"simple": 0.40, "medium": 0.35, "hard": 0.25}


def _build_system_prompt(task_type: str) -> str:
    """Build the system prompt instructing the model to output valid JSON."""
    return (
        "You are a helpful assistant that generates valid JSON. "
        "Respond ONLY with a JSON code block. Do not include any explanation "
        "before or after the JSON. Wrap your output in ```json and ``` markers."
    )


def generate_sample(rng: random.Random | None = None) -> dict:
    """Generate a single prompt sample with metadata."""
    if rng is None:
        rng = random.Random()

    # All tasks are JSON — pick difficulty only
    task_type = "json"
    difficulty = rng.choices(list(DIFFICULTY_WEIGHTS.keys()), weights=list(DIFFICULTY_WEIGHTS.values()), k=1)[0]

    pool_key = f"{task_type}_{difficulty}"
    pool = TASK_POOLS[pool_key]
    template = rng.choice(pool["templates"])

    params = template["params"]()
    instruction = template["instruction"].format(**params)
    system_prompt = _build_system_prompt(task_type)

    return {
        "system_prompt": system_prompt,
        "prompt": instruction,
        "task_type": task_type,
        "difficulty": difficulty,
    }


def generate_dataset(
    num_samples: int = 5000,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> DatasetDict:
    """Generate the full synthetic dataset as a HuggingFace DatasetDict."""
    rng = random.Random(seed)
    samples = [generate_sample(rng) for _ in range(num_samples)]

    # Deterministic split
    rng.shuffle(samples)
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    def _to_columnar(rows: list[dict]) -> dict:
        return {k: [r[k] for r in rows] for k in rows[0]}

    return DatasetDict(
        {
            "train": Dataset.from_dict(_to_columnar(train_samples)),
            "test": Dataset.from_dict(_to_columnar(test_samples)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic JSON prompt dataset")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction for test split")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples (seed={args.seed})...")
    ds = generate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    # Also save a human-readable preview
    preview_path = out_path / "preview.json"
    preview = [ds["train"][i] for i in range(min(10, len(ds["train"])))]
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset saved to {out_path}")
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Test:  {len(ds['test'])} samples")

    # Print distribution stats
    for split_name in ["train", "test"]:
        split = ds[split_name]
        diffs = split["difficulty"]
        print(f"\n  {split_name} distribution:")
        for dd in ["simple", "medium", "hard"]:
            count = sum(1 for d in diffs if d == dd)
            print(f"    json/{dd}: {count}")


if __name__ == "__main__":
    main()
