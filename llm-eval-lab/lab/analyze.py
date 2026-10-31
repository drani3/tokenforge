"""Analyze a GSM8K vs GSM-Symbolic run.

Prints, per model:
  - accuracy per dataset with a 95% cluster-bootstrap CI (resampling GSM8K questions,
    since variants of the same question are correlated)
  - the gap vs GSM8K on the same original questions (paired bootstrap)
  - spread of accuracy across variant sets (instance 0, 1, ...), as in the GSM-Symbolic paper

Usage:
    python -m lab.analyze results/runs/<run_id>
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ORIGINALS = "gsm8k_originals"
N_BOOT = 2000


def load_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_original(rows: list[dict]) -> dict[int, list[float]]:
    """Scores grouped by the GSM8K question they derive from (the bootstrap cluster)."""
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["meta"]["original_id"]].append(r["score"])
    return clusters


def ci(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]


def mean_of(clusters: dict[int, list[float]], ids: list[int]) -> float:
    scores = [x for i in ids for x in clusters[i]]
    return sum(scores) / len(scores)


def bootstrap_acc(clusters: dict[int, list[float]], rng: random.Random) -> tuple[float, float, float]:
    ids = list(clusters)
    boots = [mean_of(clusters, rng.choices(ids, k=len(ids))) for _ in range(N_BOOT)]
    return (mean_of(clusters, ids), *ci(boots))


def bootstrap_gap(orig: dict[int, list[float]], var: dict[int, list[float]],
                  rng: random.Random) -> tuple[float, float, float] | None:
    """GSM8K accuracy minus variant accuracy, restricted to shared originals and resampled jointly."""
    ids = sorted(orig.keys() & var.keys())
    if not ids:
        return None
    gap = lambda sample: mean_of(orig, sample) - mean_of(var, sample)
    boots = [gap(rng.choices(ids, k=len(ids))) for _ in range(N_BOOT)]
    return (gap(ids), *ci(boots))


def variant_set_accs(rows: list[dict]) -> list[float]:
    """Accuracy of each variant set (all templates at one instance index)."""
    sets = defaultdict(list)
    for r in rows:
        sets[r["meta"]["instance"]].append(r["score"])
    return [statistics.mean(sets[i]) for i in sorted(sets)]


def summarize(run_dir: Path, seed: int = 0) -> dict[str, dict[str, dict]]:
    """model -> dataset -> stats. Shared by the text report and the plots."""
    rng = random.Random(seed)
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in load_rows(run_dir):
        groups[r["model"]][r["dataset"]].append(r)

    summary: dict[str, dict[str, dict]] = {}
    for model, datasets in groups.items():
        orig = by_original(datasets[ORIGINALS]) if ORIGINALS in datasets else None
        summary[model] = {}
        for name, rows in datasets.items():
            clusters = by_original(rows)
            is_variant = name != ORIGINALS
            has_strict = all("strict_score" in r for r in rows)  # older runs predate it
            summary[model][name] = {
                "n": len(rows),
                "acc": bootstrap_acc(clusters, rng),
                "strict_acc": statistics.mean(r["strict_score"] for r in rows) if has_strict else None,
                "format_rate": (sum(r["extraction"] == "format" for r in rows) / len(rows)
                                if has_strict else None),
                "gap": bootstrap_gap(orig, clusters, rng) if orig and is_variant else None,
                "set_accs": variant_set_accs(rows) if is_variant else [],
                "latency_s": statistics.mean(r["total_s"] for r in rows),
                "tokens_per_s": statistics.mean(r["tokens_per_s"] for r in rows),
            }
    return summary


def format_spread(accs: list[float]) -> str:
    if len(accs) < 2:
        return "-"
    return (f"{len(accs)} sets: mean {statistics.mean(accs):.1%} sd {statistics.stdev(accs):.1%} "
            f"range {min(accs):.1%}-{max(accs):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for model, datasets in summarize(args.run_dir, args.seed).items():
        print(f"\n=== {model} ===")
        print(f"{'dataset':<20}{'n':>6}{'accuracy [95% CI]':>24}{'strict':>8}{'format':>8}"
              f"{'gap vs GSM8K [95% CI]':>28}   variant-set spread")
        for name, s in datasets.items():
            acc, lo, hi = s["acc"]
            gap = "-"
            if s["gap"]:
                g, glo, ghi = s["gap"]
                gap = f"{g:+.1%} [{glo:+.1%}, {ghi:+.1%}]"
            strict = f"{s['strict_acc']:.1%}" if s["strict_acc"] is not None else "-"
            fmt = f"{s['format_rate']:.0%}" if s["format_rate"] is not None else "-"
            print(f"{name:<20}{s['n']:>6}{f'{acc:.1%} [{lo:.1%}, {hi:.1%}]':>24}{strict:>8}{fmt:>8}"
                  f"{gap:>28}   {format_spread(s['set_accs'])}")
    print("\naccuracy = flexible (falls back to last number), strict = 'Answer: n' format required, "
          "format = share of responses that used the format")


if __name__ == "__main__":
    main()
