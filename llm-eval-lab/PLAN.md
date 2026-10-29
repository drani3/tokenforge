# Local LLM Evaluation Lab — Plan

Goal: build a harness that answers *"which local model + quantization + prompt + settings gives the best accuracy per second on my machine?"*, then use it to optimize a RAG pipeline — with evaluation rigorous enough to trust the conclusions.

Stack: Python 3.11+, [Ollama](https://ollama.com) (local, `localhost:11434`), `ollama` Python package, pandas, matplotlib, SQLite/JSONL for results.

---

## Phase 0 — Setup
- [x] Install Ollama, verify with `ollama run llama3.2:3b "hi"`
- [x] Note hardware: RTX 3060 Laptop, 6 GB VRAM → up to ~7–8B at q4 fits on GPU
- [ ] Pull 3–4 models from different families sized for your hardware, e.g. `llama3.2:3b`, `qwen2.5:7b`, `gemma2:9b`, `phi3`
- [x] Create Python venv (`py -3.12 -m venv .venv`), `pip install -r requirements.txt`

## Phase 1 — Harness
- [x] `client.py`: thin wrapper around `/api/chat` that returns output + metrics (`eval_count`, `eval_duration`, `total_duration`, `prompt_eval_count`)
- [x] `runner.py`: run `(model, prompt_template, options, dataset)` → one result row per example
- [x] Persist every run to JSONL/SQLite: run_id, config hash, model, options, input, output, score, latency, tokens/sec, timestamp
- [x] Caching: skip calls whose `(config, input)` already exist
- [x] Config files (TOML) per experiment so every run is reproducible

**Done when:** one command runs a config over a dataset and writes a results file. ✅ `.venv/Scripts/python -m lab.runner configs/smoke.toml`

## Phase 2 — Model comparison on checkable tasks
Datasets (start with ~100–200 examples each):
- [ ] **Math**: GSM8K subset — exact match on final number (write an answer extractor)
- [ ] **Classification**: sentiment or topic — accuracy, macro-F1
- [ ] **Structured extraction**: text → JSON fields using Ollama's `format` (JSON schema) — validity rate + per-field accuracy

Analysis:
- [ ] Table: model × task → score, mean latency, tokens/sec
- [ ] Plot accuracy vs. latency, mark the **Pareto frontier**

## Phase 3 — Optimization experiments (one variable at a time)
- [ ] **Prompting**: zero-shot vs few-shot (1/3/5) vs chain-of-thought vs "answer only"
- [ ] **Quantization**: same model at `q4_K_M` vs `q8_0` vs `fp16` — accuracy lost vs speed gained
- [ ] **Sampling**: temperature 0 vs 0.7; **self-consistency** (5 samples, majority vote) — gain vs 5× cost
- [ ] **Context size** (`num_ctx`): effect on speed/memory
- [ ] Write a short conclusion per experiment

## Phase 4 — RAG pipeline
- [ ] Choose a corpus (own notes, docs, Wikipedia subset)
- [ ] Hand-write ~100 Q&A pairs, each tagged with the source chunk/document → eval set
- [ ] Embeddings via `/api/embed` (`nomic-embed-text`, `mxbai-embed-large`)
- [ ] Vector store: start with NumPy cosine similarity (no DB needed)
- [ ] Variables to compare:
  - chunk size (256 / 512 / 1024 tokens) and overlap
  - retrieval: BM25 vs embeddings vs hybrid (reciprocal rank fusion)
  - top-k (1 / 3 / 5 / 10)
  - embedding model
- [ ] Metrics, evaluated **separately**:
  - Retrieval: recall@k, MRR
  - Generation: correctness, faithfulness (answer supported by retrieved context)

## Phase 5 — LLM-as-judge (and how far to trust it)
- [ ] Judge prompt with a clear rubric, using the largest local model
- [ ] Hand-grade ~50 answers yourself
- [ ] Measure judge–human agreement (accuracy, Cohen's kappa)
- [ ] Bias probes: length bias, position bias (swap A/B order), self-preference
- [ ] Decide where the judge is trustworthy enough to use

## Evaluation rules (apply throughout)
- Fixed seeds (`options.seed`) and fixed, versioned eval sets
- Split dev/test — tune on dev, report test **once** at the end
- Report **bootstrap 95% confidence intervals**; treat overlapping CIs as "no clear difference"
- Fair comparisons: same examples, same budget per config
- Log everything; never trust a number you can't regenerate

## Stretch goals
- [ ] Automatic prompt optimization (generate prompt variants, keep the best on dev)
- [ ] Simple web dashboard for browsing results
- [ ] Concurrency (`OLLAMA_NUM_PARALLEL`) and throughput benchmarking
- [ ] Fine-tune a small model (LoRA) and add it to the comparison
- [ ] Final write-up: "what I learned about local LLMs", like a mini paper

## Suggested layout
```
llm-eval-lab/
  PLAN.md
  configs/          # experiment configs
  data/             # eval sets (gsm8k_subset.jsonl, rag_qa.jsonl, ...)
  lab/
    client.py       # Ollama wrapper + metrics
    runner.py       # run config over dataset
    scorers.py      # exact match, F1, JSON validity, judge
    rag/            # chunking, embedding, retrieval
    stats.py        # bootstrap CIs, kappa
  results/          # JSONL / SQLite (gitignored)
  notebooks/        # analysis + plots
```
