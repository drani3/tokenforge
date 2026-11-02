"""Read model responses from a run.

Usage:
    python -m lab.show results/runs/<run_id> --model qwen2.5:7b --failed
    python -m lab.show results/runs/<run_id> --model qwen2.5:7b --regressions --dataset gsm_symbolic_main

--regressions shows variants the model got wrong although it solved the GSM8K original,
with both questions and both responses side by side.
"""
import argparse
import json
import sys
from pathlib import Path

from .analyze import ORIGINALS, load_rows

ROOT = Path(__file__).resolve().parent.parent
RULE = "=" * 100


def load_questions(datasets: set[str]) -> dict[str, str]:
    questions = {}
    for name in datasets:
        with (ROOT / "data" / f"{name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                questions[r["id"]] = r["question"]
    return questions


def print_response(label: str, row: dict, question: str, full: bool) -> None:
    verdict = "CORRECT" if row["score"] else "WRONG"
    output = row["output"] if full else "\n".join(row["output"].splitlines()[-12:])
    print(f"--- {label}: {row['example_id']} | expected {row['expected']} | got {row['extracted']} "
          f"({row['extraction']}) | {verdict}")
    print(f"Q: {question}\n")
    print(output if full else f"[...last 12 lines; --full for all]\n{output}")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # questions contain ₺, ₣, ...
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--failed", action="store_true", help="only wrong answers")
    parser.add_argument("--regressions", action="store_true",
                        help="variants answered wrong whose GSM8K original was answered right")
    parser.add_argument("--full", action="store_true", help="print whole responses")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = [r for r in load_rows(args.run_dir) if not args.model or r["model"] == args.model]
    questions = load_questions({r["dataset"] for r in rows})

    if args.regressions:
        originals = {(r["model"], r["meta"]["original_id"]): r for r in rows if r["dataset"] == ORIGINALS}
        selected = [
            r for r in rows
            if r["dataset"] != ORIGINALS and (not args.dataset or r["dataset"] == args.dataset)
            and r["score"] == 0 and originals.get((r["model"], r["meta"]["original_id"]), {}).get("score") == 1
        ]
        print(f"{len(selected)} regressions (showing up to {args.limit})\n")
        for r in selected[:args.limit]:
            orig = originals[(r["model"], r["meta"]["original_id"])]
            print(RULE)
            print(f"{r['model']} | original_id {r['meta']['original_id']}\n")
            print_response("GSM8K original", orig, questions[orig["example_id"]], args.full)
            print_response(r["dataset"], r, questions[r["example_id"]], args.full)
        return

    selected = [r for r in rows
                if (not args.dataset or r["dataset"] == args.dataset) and (not args.failed or r["score"] == 0)]
    print(f"{len(selected)} rows (showing up to {args.limit})\n")
    for r in selected[:args.limit]:
        print(RULE)
        print_response(f"{r['model']} | {r['dataset']}", r, questions[r["example_id"]], args.full)


if __name__ == "__main__":
    main()
