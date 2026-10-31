"""Build eval sets from GSM-Symbolic (https://huggingface.co/datasets/apple/GSM-Symbolic).

Downloads the main/p1/p2 splits into data/raw/gsm_symbolic/ and writes:
    data/gsm8k_originals.jsonl            the GSM8K test questions the templates are built from
    data/gsm_symbolic_{main,p1,p2}.jsonl  generated variants, first N instances per template

Usage:
    python -m lab.prepare_gsm --instances 10
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "gsm_symbolic"
URL = "https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/main/{split}/test.jsonl"
SPLITS = ["main", "p1", "p2"]


def download(split: str, retries: int = 5) -> Path:
    path = RAW_DIR / f"{split}.jsonl"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(URL.format(split=split), timeout=60) as resp:
                path.write_bytes(resp.read())
            return path
        except OSError as e:
            print(f"download {split} failed ({e}), attempt {attempt}/{retries}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {split}")


def final_answer(gsm_answer: str) -> float | int:
    """GSM8K answers end with '#### <number>'."""
    value = float(gsm_answer.split("####")[-1].strip().replace(",", ""))
    return int(value) if value.is_integer() else value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows):>5} rows -> {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", type=int, default=10,
                        help="variants to keep per template (max 50)")
    args = parser.parse_args()

    originals: dict[int, dict] = {}
    for split in SPLITS:
        with download(split).open(encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        variants = []
        for r in raw:
            if r["instance"] >= args.instances:
                continue
            variants.append({
                "id": f"{split}-t{r['id']}-i{r['instance']}",
                "question": r["question"],
                "answer": final_answer(r["answer"]),
                "template_id": r["id"],
                "instance": r["instance"],
                "original_id": r["original_id"],
            })
            originals.setdefault(r["original_id"], {
                "id": f"gsm8k-{r['original_id']}",
                "question": r["original_question"],
                "answer": final_answer(r["original_answer"]),
                "original_id": r["original_id"],
                # main's template order, so --limit picks the same questions across datasets
                "template_id": r["id"],
            })
        # Sort by variant set first: --limit 100 then covers one complete set of all templates
        variants.sort(key=lambda v: (v["instance"], v["template_id"]))
        write_jsonl(ROOT / "data" / f"gsm_symbolic_{split}.jsonl", variants)

    write_jsonl(ROOT / "data" / "gsm8k_originals.jsonl",
                sorted(originals.values(), key=lambda r: r["template_id"]))


if __name__ == "__main__":
    main()
