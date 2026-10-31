"""Run an experiment config over one or more datasets and write one result row per
(model, dataset, example). Models are the outer loop so each stays loaded in VRAM.

Usage:
    python -m lab.runner configs/smoke.toml
    python -m lab.runner configs/gsm_contamination.toml --limit 20   # quick pilot
"""
import argparse
import hashlib
import json
import shutil
import statistics
import tomllib
from datetime import datetime
from pathlib import Path

from .client import chat
from .scorers import SCORERS
from .store import Store, request_key

ROOT = Path(__file__).resolve().parent.parent
# Example fields that aren't copied into a row's metadata
EXAMPLE_FIELDS = {"id", "question", "answer"}


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def build_messages(prompt: dict, example: dict) -> list[dict]:
    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"].format(**example)})
    return messages


def run(config_path: Path, limit: int | None = None) -> Path:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    datasets = {Path(p).stem: load_dataset(ROOT / p, limit) for p in config["datasets"]}
    scorer = SCORERS[config["scorer"]]
    options = config.get("options", {})
    store = Store(ROOT / "results" / "cache.db")

    run_id = f"{config['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.toml")

    groups: dict[tuple[str, str], list[dict]] = {}
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for model in config["models"]:
            for dataset_name, dataset in datasets.items():
                rows = groups.setdefault((model, dataset_name), [])
                for i, example in enumerate(dataset, 1):
                    messages = build_messages(config["prompt"], example)
                    key = request_key(model, messages, options)
                    result = store.get(key)
                    cached = result is not None
                    if not cached:
                        result = chat(model, messages, options)
                        store.put(key, model, {"messages": messages, "options": options}, result)

                    row = {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "model": model,
                        "dataset": dataset_name,
                        "example_id": example["id"],
                        "expected": example["answer"],
                        "output": result.content,
                        **scorer(result.content, example["answer"]),
                        "cached": cached,
                        "tokens_per_s": result.tokens_per_s,
                        "meta": {k: v for k, v in example.items() if k not in EXAMPLE_FIELDS},
                        **result.to_dict(),
                    }
                    del row["content"]
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
                    print(f"[{model} | {dataset_name}] {i}/{len(dataset)} {example['id']} "
                          f"score={row['score']:.0f} {result.total_s:.2f}s{' (cached)' if cached else ''}")

    print_summary(groups)
    print(f"\nResults: {run_dir}")
    return run_dir


def print_summary(groups: dict[tuple[str, str], list[dict]]) -> None:
    print(f"\n{'model':<16}{'dataset':<22}{'n':>6}{'accuracy':>10}{'strict':>8}"
          f"{'latency_s':>11}{'tok/s':>8}{'cached':>8}")
    for (model, dataset), rows in groups.items():
        strict = (f"{statistics.mean(r['strict_score'] for r in rows):.1%}"
                  if "strict_score" in rows[0] else "-")
        print(f"{model:<16}{dataset:<22}{len(rows):>6}"
              f"{statistics.mean(r['score'] for r in rows):>10.1%}{strict:>8}"
              f"{statistics.mean(r['total_s'] for r in rows):>11.2f}"
              f"{statistics.mean(r['tokens_per_s'] for r in rows):>8.1f}"
              f"{sum(r['cached'] for r in rows):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int, help="only run the first N examples of each dataset")
    args = parser.parse_args()
    run(args.config, args.limit)


if __name__ == "__main__":
    main()
