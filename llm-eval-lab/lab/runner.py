"""Run an experiment config over a dataset and write one result row per (model, example).

Usage:
    python -m lab.runner configs/smoke.toml
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


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(prompt: dict, example: dict) -> list[dict]:
    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"].format(**example)})
    return messages


def run(config_path: Path) -> Path:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    dataset = load_dataset(ROOT / config["dataset"])
    scorer = SCORERS[config["scorer"]]
    options = config.get("options", {})
    store = Store(ROOT / "results" / "cache.db")

    run_id = f"{config['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.toml")

    rows_by_model: dict[str, list[dict]] = {}
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for model in config["models"]:
            rows = rows_by_model.setdefault(model, [])
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
                    "example_id": example["id"],
                    "expected": example["answer"],
                    "output": result.content,
                    "score": scorer(result.content, example["answer"]),
                    "cached": cached,
                    "tokens_per_s": result.tokens_per_s,
                    **result.to_dict(),
                }
                del row["content"]
                out.write(json.dumps(row) + "\n")
                rows.append(row)
                print(f"[{model}] {i}/{len(dataset)} {example['id']} "
                      f"score={row['score']:.0f} {result.total_s:.2f}s{' (cached)' if cached else ''}")

    print_summary(rows_by_model)
    print(f"\nResults: {run_dir}")
    return run_dir


def print_summary(rows_by_model: dict[str, list[dict]]) -> None:
    print(f"\n{'model':<24}{'n':>5}{'accuracy':>10}{'latency_s':>11}{'tok/s':>8}{'cached':>8}")
    for model, rows in rows_by_model.items():
        print(f"{model:<24}{len(rows):>5}"
              f"{statistics.mean(r['score'] for r in rows):>10.1%}"
              f"{statistics.mean(r['total_s'] for r in rows):>11.2f}"
              f"{statistics.mean(r['tokens_per_s'] for r in rows):>8.1f}"
              f"{sum(r['cached'] for r in rows):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
