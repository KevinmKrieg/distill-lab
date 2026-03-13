# distill-lab

PyTorch project for teacher-student language model distillation.

This repo is built to answer a simple question: how much of a larger language model can you preserve in a smaller student when you distill logits, hard labels, and optional hidden states? The codebase is intentionally small, config-driven, and easy to run locally.

## Usage examples

### 1. Fast smoke test

Run the full teacher vs student-from-scratch vs distilled-student workflow on a synthetic corpus:

```bash
python -m src.train --config configs/teacher_toy.yaml
python -m src.train --config configs/student_scratch_toy.yaml
python -m src.train --config configs/student_distill_toy.yaml
python scripts/summarize_runs.py \
  results/teacher_toy \
  results/student_scratch_toy \
  results/student_distill_toy \
  --labels teacher scratch distill \
  --output-dir results/reports/toy
```

### 2. Open-source teacher distillation with GPT-2

Distill a frozen `gpt2` teacher into a smaller custom student on the checked-in Tiny Shakespeare corpus. The student uses the GPT-2 tokenizer so teacher and student logits share the same vocabulary.

```bash
python -m src.train --config configs/student_scratch_shakespeare_gpt2.yaml
python -m src.train --config configs/student_distill_shakespeare_gpt2.yaml
python scripts/benchmark_runs.py \
  results/student_scratch_shakespeare_gpt2 \
  results/student_distill_shakespeare_gpt2 \
  --labels scratch distill \
  --output-dir results/reports/shakespeare_gpt2_bench
```

The first run will download the GPT-2 tokenizer and model unless they are already cached locally.

### 3. Real-text distillation with local subword tokenization

Train a larger local teacher and smaller student on the checked-in Tiny Shakespeare corpus using the repo's byte-level BPE tokenizer:

```bash
python -m src.train --config configs/teacher_shakespeare_bpe.yaml
python -m src.train --config configs/student_scratch_shakespeare_bpe.yaml
python -m src.train --config configs/student_distill_shakespeare_bpe.yaml
python -m src.eval --config configs/student_distill_shakespeare_bpe.yaml
python scripts/summarize_runs.py \
  results/teacher_shakespeare_bpe \
  results/student_scratch_shakespeare_bpe \
  results/student_distill_shakespeare_bpe \
  --labels teacher scratch distill \
  --output-dir results/reports/shakespeare_bpe
```

### 4. Open-data workflow for larger experiments

Prepare a standard open corpus, then run the same teacher/student/distillation pipeline:

```bash
python scripts/prepare_wikitext2.py
python -m src.train --config configs/teacher_wikitext2_bpe.yaml
python -m src.train --config configs/student_scratch_wikitext2_bpe.yaml
python -m src.train --config configs/student_distill_wikitext2_bpe.yaml
```

Or normalize local TinyStories shards into `train.txt` and `val.txt` first:

```bash
python scripts/prepare_tinystories.py --source-dir /path/to/TinyStories
python -m src.train --config configs/teacher_tinystories_bpe.yaml
python -m src.train --config configs/student_scratch_tinystories_bpe.yaml
python -m src.train --config configs/student_distill_tinystories_bpe.yaml
```

### 5. Add latency and memory to the comparison table

After you have trained one or more runs, benchmark their saved `best.pt` checkpoints:

```bash
python scripts/benchmark_runs.py \
  results/teacher_shakespeare_bpe \
  results/student_scratch_shakespeare_bpe \
  results/student_distill_shakespeare_bpe \
  --labels teacher scratch distill \
  --output-dir results/reports/shakespeare_bpe_bench
```

This writes a `benchmark.json` file into each run directory and a comparison table with latency, throughput, and memory columns.

## Open-source model use case

The repo now supports a real Hugging Face teacher path for causal language modeling distillation. In practice, that means you can:

- tokenize your corpus with a pretrained tokenizer such as `gpt2`
- freeze a pretrained causal LM teacher loaded through `transformers`
- train a smaller custom student transformer against the teacher logits
- optionally match the teacher's last hidden states through a learned projection
- summarize accuracy, perplexity, latency, and memory in the same reporting pipeline

The current Hugging Face adapter is focused on causal LM teachers. It does not yet support encoder-only teachers or sequence-classification heads.

## Example output

Below is a real validation-perplexity plot produced by the reporting script from a verified local smoke run. All three runs improve over training, and the teacher improves fastest on this tiny setup.

![Validation perplexity curve](docs/assets/report_smoke_val_perplexity.svg)

Example summary from that run:

| run | best val ppl | best val acc | params |
| --- | ---: | ---: | ---: |
| teacher | 26.0278 | 0.2137 | 12053 |
| scratch | 39.5372 | 0.1183 | 4005 |
| distill | 41.8707 | 0.0878 | 4005 |

## What the repo includes

- teacher training, student baseline training, and student distillation
- synthetic, char-tokenized, word-tokenized, BPE-tokenized, and Hugging Face-tokenized data paths
- local-teacher and Hugging Face-teacher distillation flows for causal LM
- dataset prep scripts for WikiText-2 and TinyStories
- reporting tools that write `summary.csv`, `summary.md`, benchmark tables, and SVG plots
- YAML configs for toy, Shakespeare, WikiText-2, TinyStories, and GPT-2-based experiments

## Minimal project map

```text
configs/   experiment definitions
scripts/   dataset prep, reporting, and benchmarking CLIs
src/       training, evaluation, tokenization, teachers, and losses
data/      local corpora
artifacts/ saved tokenizer models
results/   checkpoints and metrics
```

## Notes

- The first BPE run will create a tokenizer model under `artifacts/tokenizers/` and later runs will reuse it.
- The GPT-2 example depends on Hugging Face model files being downloadable or already cached locally.
- `scripts/summarize_runs.py` turns `metrics.jsonl` files into CSV, Markdown, and SVG artifacts.
- `scripts/benchmark_runs.py` adds latency, throughput, and memory data to the same comparison flow.
