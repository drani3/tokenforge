"""Memorization probe: can a model continue the first half of a question verbatim?

For each GSM8K original (likely seen in training) and its GSM-Symbolic variant (published
Oct 2024, after these models, so a clean control), feed the first half of the question in raw
completion mode and compare the continuation to the true second half:

    overlap      longest common subsequence of words / words in the true second half
    num_recall   share of the true second half's numbers that appear in the continuation
    orig_nums    (variants only) share of the *original's* second-half numbers that appear,
                 i.e. the model continuing a variant with numbers it remembers from GSM8K

Usage:
    python -m lab.memprobe --models gemma2:2b llama3.2:3b qwen2.5:7b
    python -m lab.memprobe --summarize results/runs/memprobe-<ts>
"""
import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .analyze import ci
from .client import complete
from .store import Store, request_key

ROOT = Path(__file__).resolve().parent.parent
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 128}
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace(",", ""))


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(text.replace(",", "")))


def split_half(question: str) -> tuple[str, str]:
    tokens = question.split(" ")
    mid = len(tokens) // 2
    return " ".join(tokens[:mid]), " ".join(tokens[mid:])


def lcs_ratio(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return 0.0
    prev = [0] * (len(candidate) + 1)
    for r in reference:
        cur = [0]
        for j, c in enumerate(candidate, 1):
            cur.append(prev[j - 1] + 1 if r == c else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] / len(reference)


def recall(expected: set[str], text: str) -> float | None:
    return len(expected & numbers(text)) / len(expected) if expected else None


def load(name: str) -> list[dict]:
    with (ROOT / "data" / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def probe_items() -> list[dict]:
    originals = {r["original_id"]: r for r in load("gsm8k_originals")}
    variants = [r for r in load("gsm_symbolic_main") if r["instance"] == 0]
    items = []
    for v in variants:
        o = originals[v["original_id"]]
        o_prefix, o_rest = split_half(o["question"])
        v_prefix, v_rest = split_half(v["question"])
        items.append({"kind": "gsm8k", "id": o["id"], "original_id": o["original_id"],
                      "prefix": o_prefix, "rest": o_rest})
        items.append({"kind": "variant", "id": v["id"], "original_id": o["original_id"],
                      "prefix": v_prefix, "rest": v_rest, "original_rest": o_rest})
    return items


def run(models: list[str]) -> Path:
    store = Store(ROOT / "results" / "cache.db")
    run_dir = ROOT / "results" / "runs" / f"memprobe-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    items = probe_items()
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for model in models:
            for i, item in enumerate(items, 1):
                key = request_key(model, [{"role": "raw", "content": item["prefix"]}], OPTIONS)
                result = store.get(key)
                if result is None:
                    result = complete(model, item["prefix"], OPTIONS)
                    store.put(key, model, {"raw_prompt": item["prefix"], "options": OPTIONS}, result)
                # Compare only as many words as the true remainder has (models often run on)
                cont_words = words(result.content)[: len(words(item["rest"])) + 5]
                cont = " ".join(cont_words)
                row = {
                    "model": model, **item, "continuation": result.content,
                    "overlap": lcs_ratio(words(item["rest"]), cont_words),
                    "num_recall": recall(numbers(item["rest"]), cont),
                    "orig_nums": (recall(numbers(item["original_rest"]) - numbers(item["rest"]), cont)
                                  if item["kind"] == "variant" else None),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"[{model}] {i}/{len(items)} {item['id']} overlap={row['overlap']:.2f}", flush=True)
    summarize(run_dir)
    return run_dir


def mean_ci(values: list[float], rng: random.Random) -> str:
    if not values:
        return "-"
    boots = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    lo, hi = ci(boots)
    return f"{statistics.mean(values):.2f} [{lo:.2f}, {hi:.2f}]"


def leading_match(reference: str, continuation: str) -> int:
    """Number of words at the start of the continuation that exactly match the true remainder.
    Stricter than LCS overlap, which common words like 'how many' inflate even for unrelated text."""
    n = 0
    for r, c in zip(words(reference), words(continuation)):
        if r != c:
            break
        n += 1
    return n


def summarize(run_dir: Path, exact_words: int = 8) -> None:
    rng = random.Random(0)
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["kind"])].append(r)

    print(f"\n{'model':<14}{'set':<9}{'overlap [95% CI]':>22}{'overlap>=0.8':>14}"
          f"{f'first {exact_words} exact':>16}{'num recall':>12}{'orig nums in variant':>22}")
    for (model, kind), rs in groups.items():
        overlap = [r["overlap"] for r in rs]
        num = [r["num_recall"] for r in rs if r["num_recall"] is not None]
        orig = [r["orig_nums"] for r in rs if r["orig_nums"] is not None]
        exact = [leading_match(r["rest"], r["continuation"]) >= min(exact_words, len(words(r["rest"])))
                 for r in rs]
        print(f"{model:<14}{kind:<9}{mean_ci(overlap, rng):>22}"
              f"{sum(o >= 0.8 for o in overlap) / len(overlap):>14.0%}"
              f"{sum(exact) / len(exact):>16.0%}"
              f"{statistics.mean(num) if num else 0:>12.0%}"
              f"{(f'{statistics.mean(orig):.0%}' if orig else '-'):>22}")
    print(f"\noverlap = LCS of words vs true 2nd half (1.0 = verbatim; common words give a ~0.3 floor)"
          f"\nfirst {exact_words} exact = continuation starts with the true next {exact_words} words"
          f"\nnum recall = share of the true 2nd half's numbers reproduced"
          f"\norig nums in variant = share of the GSM8K original's numbers produced when continuing the variant")
    print(f"\nResults: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["gemma2:2b", "llama3.2:3b", "qwen2.5:7b"])
    parser.add_argument("--summarize", type=Path, help="re-print the summary of an existing run")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args.models)


if __name__ == "__main__":
    main()
