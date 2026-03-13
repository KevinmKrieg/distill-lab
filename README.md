# distill-lab

PyTorch project for teacher-student language model distillation.

The repo now supports three data/tokenization paths:

- a synthetic toy corpus for fast smoke tests
- local text corpora with character or word tokenization
- local text corpora with a native byte-level BPE tokenizer that can be trained and persisted from your training split

## What is implemented

- Teacher training with standard next-token cross-entropy
- Student baseline training from scratch
- Student distillation with:
  - softened teacher logits via KL divergence
  - hard-label cross-entropy on ground-truth next tokens
  - optional hidden-state matching through a learned projection
- Dataset backends for:
  - synthetic template data
  - line-based text corpora from local files
- Tokenizer support for:
  - character-level tokenization
  - word-level tokenization with frequency filtering
  - byte-level BPE tokenization with saved tokenizer models
- Dataset preparation scripts for:
  - WikiText-2
  - TinyStories
- YAML configs for toy, Shakespeare, WikiText-2, and TinyStories experiments
- Checkpointing and JSONL metric logging
- Standalone evaluation entrypoint

## Project layout

```text
distill-lab/
  artifacts/
    tokenizers/
  configs/
    teacher_toy.yaml
    teacher_shakespeare_char.yaml
    teacher_shakespeare_bpe.yaml
    teacher_wikitext2_bpe.yaml
    teacher_tinystories_bpe.yaml
    ...
  data/
    tiny_shakespeare/
      train.txt
      val.txt
  scripts/
    run_train.sh
    run_eval.sh
    prepare_wikitext2.py
    prepare_tinystories.py
  src/
    data.py
    eval.py
    losses.py
    models.py
    train.py
    utils.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Synthetic smoke test:

```bash
python -m src.train --config configs/teacher_toy.yaml
python -m src.train --config configs/student_scratch_toy.yaml
python -m src.train --config configs/student_distill_toy.yaml
```

Real-text char tokenizer run on the checked-in Shakespeare sample:

```bash
python -m src.train --config configs/teacher_shakespeare_char.yaml
python -m src.train --config configs/student_scratch_shakespeare_char.yaml
python -m src.train --config configs/student_distill_shakespeare_char.yaml
```

Real-text subword run on the same corpus:

```bash
python -m src.train --config configs/teacher_shakespeare_bpe.yaml
python -m src.train --config configs/student_scratch_shakespeare_bpe.yaml
python -m src.train --config configs/student_distill_shakespeare_bpe.yaml
python -m src.eval --config configs/student_distill_shakespeare_bpe.yaml
```

The first BPE run will create `artifacts/tokenizers/tiny_shakespeare_bpe.json`. Later runs reuse it.

## Preparing WikiText-2

```bash
python scripts/prepare_wikitext2.py
python -m src.train --config configs/teacher_wikitext2_bpe.yaml
python -m src.train --config configs/student_scratch_wikitext2_bpe.yaml
python -m src.train --config configs/student_distill_wikitext2_bpe.yaml
```

If you already have the extracted dataset locally:

```bash
python scripts/prepare_wikitext2.py --source-dir /path/to/wikitext-2
```

## Preparing TinyStories

If you have local TinyStories JSON or JSONL shards:

```bash
python scripts/prepare_tinystories.py --source-dir /path/to/TinyStories
python -m src.train --config configs/teacher_tinystories_bpe.yaml
python -m src.train --config configs/student_scratch_tinystories_bpe.yaml
python -m src.train --config configs/student_distill_tinystories_bpe.yaml
```

If you have a local archive instead:

```bash
python scripts/prepare_tinystories.py --archive /path/to/TinyStories.tar.gz
```

The TinyStories prep script can also download from a URL if you supply `--download-url`, but I did not exercise that path in this environment.

## Data configuration

`data.type: toy` uses the deterministic synthetic corpus.

`data.type: text` reads `train_path` and `val_path`, builds the tokenizer on the training split, and chunks tokenized lines into causal language-model sequences.

Example BPE config:

```yaml
data:
  type: text
  name: wikitext2
  train_path: data/wikitext2/train.txt
  val_path: data/wikitext2/val.txt
  batch_size: 32
  max_seq_len: 128
  stride: 64
  num_workers: 0
  tokenizer:
    level: bpe
    vocab_size: 512
    min_frequency: 2
    model_path: artifacts/tokenizers/wikitext2_bpe.json
```

Supported tokenizer configs:

- `level: char`
- `level: word`
- `level: bpe`
- `lowercase`, `min_freq`, and `max_vocab_size` for word tokenization
- `vocab_size`, `min_frequency`, and `model_path` for BPE tokenization

## Training objective

```text
L = alpha * T^2 * KL(softmax(z_t / T), softmax(z_s / T))
  + (1 - alpha) * CE(y, z_s)
  + hidden_loss_weight * MSE(W h_s, h_t)
```

Where `W` projects student hidden states into the teacher hidden size when hidden matching is enabled.

## Outputs

Each run writes into its configured `output_dir`:

- `best.pt`: best checkpoint by validation perplexity
- `last.pt`: final epoch checkpoint
- `metrics.jsonl`: per-epoch training and validation metrics

## Notes

- The BPE tokenizer is implemented locally in `src/data.py`; no external tokenization library is required.
- WikiText-2 and TinyStories configs assume you have prepared `train.txt` and `val.txt` with the provided scripts.
- Only the Shakespeare-based char and BPE paths were exercised locally in this environment. The remote dataset preparation paths were added but not executed because network access is restricted here.
