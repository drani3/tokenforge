# Findings

Running log of what the experiments in this lab have shown, with the evidence for each claim and the
commands to reproduce it. Last updated: 2026-11-02.

**TL;DR**
1. **qwen2.5:7b's GSM8K score is inflated.** It scores 98% on GSM8K but 90% on the same questions with new
   names and numbers. All 8 questions that flipped went the same direction (p = 0.008).
2. **gemma2:2b shows no contamination signal** (61% vs 60%, flips balanced 12 vs 11). **llama3.2:3b is
   borderline** (77% vs 68%, p = 0.09).
3. **All three models break down as problems get longer.** From GSM-Symbolic to P2, gemma drops 60% → 14%,
   llama 68% → 29% and qwen 90% → 67%.
4. **The failures are reasoning failures, not arithmetic.** None of qwen's 8 regressions had a calculation
   error. Several read identical wording differently from the original, and one followed the original's
   item order.
5. **The memorization is structural, not verbatim.** None of the models can recite GSM8K questions any better
   than the unseen variants (prefix-completion probe).
6. Along the way: **two grader bugs**, **one broken dataset item**, **greedy decoding that isn't deterministic
   across runs**, and **thermal throttling** that invalidates the speed numbers.

---

## Setup

| | |
|---|---|
| Models | `gemma2:2b`, `llama3.2:3b`, `qwen2.5:7b`, all instruct versions with default Ollama quantization |
| Hardware | RTX 3060 Laptop GPU, 6 GB VRAM |
| Data | 100 GSM8K test questions and their [GSM-Symbolic](https://arxiv.org/abs/2410.05229) variants (main, P1 = +1 clause, P2 = +2 clauses). Variant set 0 for main and P1; sets 0–1 for P2, which has only 50 templates |
| Prompt | Zero-shot chain of thought, ending with "`Answer: <number>`" (`configs/gsm_contamination.toml`) |
| Decoding | Greedy: `temperature = 0`, `seed = 42`, `num_predict = 1024` |
| Grading | Last `Answer: <n>` in the response, falling back to the last number; must equal the expected answer. Reported as **flexible** and **strict** (format required), as in lm-evaluation-harness |
| Stats | 95% CIs from a cluster bootstrap over GSM8K questions (2,000 resamples); gaps are paired on the same original questions |

**Runs**
- Main eval: `results/runs/gsm_contamination-20261101-173851`. This is a regrade of `…-20261101-154330` after the currency fix; the model outputs are identical.
- Memorization probe: `results/runs/memprobe-20261102-182141`

**Reproduce**
```
python -m lab.prepare_gsm --instances 10
python -m lab.runner configs/gsm_contamination.toml --limit 100
python -m lab.analyze results/runs/<run_id>
python -m lab.plot results/runs/<run_id>
python -m lab.show results/runs/<run_id> --model qwen2.5:7b --regressions --dataset gsm_symbolic_main
python -m lab.memprobe
```
`results/` and the GSM-Symbolic data are gitignored. Re-running rebuilds them, with model calls served from `results/cache.db` when present.

---

## Finding 1: GSM8K vs GSM-Symbolic (contamination)

**Question:** do the models score lower on GSM8K questions once the names and numbers change?

| Model | GSM8K | GSM-Symbolic | Gap [95% CI] | Flips: worse / better | Sign test |
|---|---|---|---|---|---|
| gemma2:2b | 61% [51, 71] | 60% [50, 70] | **+1%** [−8, +10] | 12 / 11 | p = 1.00 |
| llama3.2:3b | 77% [69, 85] | 68% [59, 77] | **+9%** [0, +18] | 16 / 7 | p = 0.09 |
| qwen2.5:7b | 98% [95, 100] | 90% [84, 95] | **+8%** [+3, +14] | **8 / 0** | **p = 0.008** |

*"Flips" are original/variant pairs where exactly one was answered correctly. Without contamination, they
should split about evenly in both directions.*

**Evidence**
- **qwen:** the CI for the gap excludes zero, and **every one of the 8 flips goes the same direction**. Not a
  single variant was solved when its original wasn't. The chance of 8/0 if direction were random is 0.8%.
  Excluding the broken trout question (Finding 4) leaves the gap at **+7.1%** (n = 99).
- **gemma:** flips split 12/11, which is what noise looks like. Its GSM8K score appears to reflect its actual
  ability.
- **llama:** leans the same way (16/7) but isn't significant at 100 questions. The full 10-set run should
  settle it.

**Conclusion:** qwen2.5:7b's GSM8K accuracy overstates its math reasoning by roughly 7–8 points. This is
consistent with GSM8K being in its training data. (It isn't *proof*: we can't inspect the training data.)

## Finding 2: Accuracy falls sharply as problems get longer

| Model | GSM8K | Symbolic | P1 (+1 clause) | P2 (+2 clauses) |
|---|---|---|---|---|
| gemma2:2b | 61% | 60% | 40% [31, 49] | **14%** [6, 23] |
| llama3.2:3b | 77% | 68% | 66% [57, 75] | **29%** [20, 39] |
| qwen2.5:7b | 98% | 90% | 80% [72, 87] | **67%** [56, 77] |

**Evidence:** `plots/difficulty.png` and `plots/gap.png` in the main run. All P2 gaps against GSM8K exclude
zero: gemma +44% [+27, +60], llama +51% [+35, +66], qwen +31% [+21, +42].

**Conclusion:** adding two reasoning steps cuts small-model accuracy by half or more. qwen degrades most
gracefully, but even it loses about a quarter of its accuracy from Symbolic to P2.

## Finding 3: The failures are reasoning failures, not arithmetic

**Method:** I read all 8 qwen2.5:7b regressions (original right, variant wrong):
`python -m lab.show <run> --model qwen2.5:7b --regressions --dataset gsm_symbolic_main --full`

| Question (template) | Failure | Category |
|---|---|---|
| Ice cubes (t11) | The variant lists "8 giant, **15 small, 13 medium**". qwen assigned 15 to *medium* and 13 to *small*, the item order of the **original** question | Followed the original |
| Beach walk (t17) | Assumed equal distance means equal time (60 + 60 min). It computed that the sidewalk speed is double, then ignored it | Reasoning |
| TV episodes (t21) | Counted Tue + Wed (8 h) twice (8 + 8 + 6.25 + 8), then forced 9.67 to "9" | Bookkeeping |
| Cotton candy (t40) | "Get **5** pounds 1/4 off" while buying 22: discounted all 7 extra pounds. The original (buy 2) had no leftover case | Reasoning, about a structure the original lacked |
| Trout (t76) | See Finding 4. qwen's answer is reasonable | **Dataset flaw** |
| Vending machine (t80) | Read "$5" as starting money in the original but "$1200" as *amount spent* in the variant, with the same wording | Inconsistent reading |
| Apples & bananas (t91) | "36 fewer apples than the bananas **bought**" (60): used the new banana total (113) | Wrong reference |
| Birthday party (t96) | "She **and her friends**… ride seven times": charged rides per person in the original, only for Mia here | Inconsistent reading |

**Tally (8):** 0 arithmetic errors · 6 reasoning or reading errors (one of them on a structurally harder
variant) · 1 followed the original · 1 dataset flaw.

**Proof sample (ice cubes):**
> **Variant:** "…If he makes 8 giant cubes, **15 small cubes, and 13 medium cubes**…"
> **qwen:** "Pavel makes **15 medium cubes**… Pavel makes **13 small cubes**…" → 64.4 (expected 86)
> **Original:** "…If he makes 3 giant cubes, **7 medium cubes, and 8 small cubes**…"

**Conclusion:** the earlier worry that variants are just "harder arithmetic" (e.g. ratio 97:37 vs 7:13) is
**not supported**. qwen computed 7 × 53, 1200 / 0.93 and 11 × 21 correctly. It fails at *understanding*
problems it can no longer pattern-match.

## Finding 4: GSM-Symbolic has flaws too

- **Broken item: template 76 (trout).** "Wei caught **4** trouts… the first weighs 70 kg, the second 33 kg,
  and the last 32 kg." Only 3 weights are given, yet the expected answer (167) implies a 4th trout of 32 kg.
  The original (gsm8k-1207) has 3 fish and 3 weights. Any model that answers 135 is penalized unfairly.
- **Some variants change structure, not just numbers.** In t40 (cotton candy), the original deal covers the
  whole purchase, while the variant leaves leftover pounds at full price. That case doesn't exist in the
  original.

**Conclusion:** part of every model's measured gap is benchmark noise, not contamination. Benchmarks need
error analysis too.

## Finding 5: The memorization is structural, not verbatim

**Method:** raw completion (no chat template, greedy). Each model gets the first half of each GSM8K question,
and of its variant as a control, published Oct 2024 after the models' release. The continuation is compared
with the true second half. Run: `results/runs/memprobe-20261102-182141` (600 completions).

| Model | Set | Word overlap [95% CI] | Starts with true next 8 words | Number recall |
|---|---|---|---|---|
| gemma2:2b | GSM8K | 0.34 [0.31, 0.37] | 2% | 15% |
| | variant | 0.34 [0.31, 0.37] | 1% | 10% |
| llama3.2:3b | GSM8K | 0.37 [0.34, 0.40] | 1% | 21% |
| | variant | 0.35 [0.32, 0.38] | 0% | 9% |
| qwen2.5:7b | GSM8K | 0.38 [0.34, 0.41] | 1% | 16% |
| | variant | 0.39 [0.36, 0.42] | 0% | 12% |

When continuing a *variant*, how often does the model produce the **original's** numbers, compared with a
random other question's numbers (chance)?

| Model | Original's numbers | Chance baseline |
|---|---|---|
| gemma2:2b | 5.9% | 7.4% |
| llama3.2:3b | 13.4% | 10.4% |
| qwen2.5:7b | 7.4% | 4.7% |

**Evidence**
- Overlap and exact-start rates are the same for GSM8K and the control. **No model recites GSM8K.**
- The near-verbatim hits (9 of 600) are **predictable text**, and they occur as often for variants. Example:
  gemma continued "her sister ate 1 more than her," with "and her brother ate 1 more than her sister" (15/18
  words), while llama continued the same prefix with "2 more".
- Number recall is higher on GSM8K, but GSM8K uses small, guessable numbers (1, 2, 3, 12) and the variants
  don't. **This metric is confounded.**
- Original-number intrusion is at chance for gemma and slightly above for llama and qwen. That's the same
  ordering as Finding 1, but too small to call evidence. One notable case: given the start of Pavel's
  ice cube variant, qwen continued with "**3 giant cubes**", the original's number, on the same question
  where it followed the original's item order in Finding 3.

**Conclusion:** the contamination shown in Finding 1 is **familiarity with the problems' structure**, not
word-for-word recall. A prefix probe on instruct models can't detect it. Base models would be the proper
test.

---

## Methodology findings (bugs and pitfalls caught)

### M1. Grader took the *first* `Answer:`, not the last
A self-correcting response ("Answer: 12 … wait … Answer: 15") was graded on 12. Fixed and covered by
`tests/test_scorers.py::test_last_answer_wins_when_model_corrects_itself`.

### M2. Grader only recognized `$`, but GSM-Symbolic uses ¥ € ₺ ₣
The variants contain `$` 587 times, `¥` 302, `€` 277, `₺` 272 and `₣` 270. Answers like `Answer: ₺206` fell
through to the last-number fallback.
- **Effect before the fix:** the format-following rate on variants looked like **82–85%** against 99% on
  GSM8K, an apparent "models follow instructions less on unfamiliar questions" effect that was **entirely a
  grader artifact**. After the fix: **92–100%**. Strict accuracy rose by up to 8 points. Flexible accuracy was
  unchanged.
- Fixed and covered by `tests/test_scorers.py::test_other_currencies`.
- **Lesson:** a surprising result is a reason to check the grader first.

### M3. Greedy decoding is not deterministic across runs
The same request (identical prompt, `temperature 0`, `seed 42`) for llama3.2:3b on the hobbies question
(t3, expected 150):
- In the eval run: **136** (cached; the same answer appears in the main run)
- Replayed 4 times a day later: **150**, identical to each other

The two responses diverge **around the 5th word** ("we need to find…" vs "let's start by identifying…"). The
136 response subtracted the hikers twice:
> "19 (paint) + 7 (football) + 14 (hike) = 40 … 190 − 40 = 150 students like to either bake or hike … 150 − 14 = 136"

Sampling the same question at temperature 0.8 gave **4/10** correct with the raw prompt and **5/10** with
the eval prompt.

**Likely cause** (unverified): floating-point differences when two candidate tokens are nearly tied, from
model loading or prompt-prefix caching. **Implication:** single-run scores carry hidden noise on near-50/50
questions, and the cache freezes one sample. This is why confidence intervals and multiple variant sets
matter.

### M4. Thermal throttling invalidates speed numbers
`results/gpu_log.csv` for the main run: GPU temperature peaked at **89°C**, with the SM clock **below
1,000 MHz in 88 of 229 samples (~38%)**, against a 2,100 MHz maximum.

| Model | Mean latency (s) | tok/s |
|---|---|---|
| gemma2:2b | 1.9–4.3 | 85–110 |
| llama3.2:3b | 4.3–7.1 | 61–63 |
| qwen2.5:7b | 8.5–14.8 | 28 |

The relative ordering holds; the absolute numbers don't. Separately, llama3.2:3b ran at **103 tok/s** in the
cool smoke test and **~22 tok/s** throughout an earlier test run. That isn't fully explained by throttling
and is still open.

### M5. `--limit` originally picked unrepresentative questions
The first pilot's datasets were sorted by template, so `--limit 10` took 10 variants of **one** question while
the GSM8K set took 10 **different** questions. The numbers weren't comparable. Fixed: variants are sorted by
(variant set, template), and the originals follow the same template order.

### M6. Metrics need a chance baseline
Memorization-probe number recall looked like a signal (GSM8K 15–21% vs variants 9–12%) until we considered
that GSM8K numbers are small and easy to guess. The "original numbers in variants" metric only became
interpretable with a random-question baseline.

---

## Limitations
- **n = 100 per dataset, one variant set** (two for P2). The CIs are wide, llama's gap is undecided, and
  variance across variant sets is unmeasured.
- **One prompt** (zero-shot CoT). The paper uses 8-shot CoT, and results may differ.
- **Instruct models only**, at default quantization.
- **Exact-match grading:** gemma answered 49.9 to a question with answer 50 (fog bank, t1) after rounding
  mid-computation, and was marked wrong.
- Training data isn't public, so contamination is *inferred*, not observed.

## Open questions / next steps
- [ ] Run all 10 variant sets: tighter CIs, variance across sets, settle llama
- [ ] Repeat Finding 1 with 8-shot CoT, as in the paper
- [ ] Memorization probe on base models, with longer prefixes (75%)
- [ ] Determinism check: re-run with the cache off and count flipped answers (M3)
- [ ] Explain llama3.2:3b's tok/s anomaly (M4)
