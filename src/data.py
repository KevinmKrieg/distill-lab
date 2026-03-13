from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
BYTE_TOKEN_OFFSET = 256
LEXICON: Dict[str, Sequence[str]] = {
    "adjectives": ["curious", "silent", "brisk", "eager", "bright", "steady"],
    "animals": ["otter", "falcon", "panda", "badger", "fox", "heron"],
    "professions": ["artist", "pilot", "teacher", "engineer", "doctor", "writer"],
    "verbs": ["studies", "builds", "finds", "carries", "observes", "sketches"],
    "places": ["forest", "harbor", "library", "station", "garden", "workshop"],
    "objects": ["lantern", "map", "notebook", "engine", "camera", "compass"],
    "times": ["today", "nightly", "weekly", "early", "soon", "again"],
}

TEMPLATES: Sequence[Sequence[str]] = (
    ("the", "{adjectives}", "{animals}", "{verbs}", "near", "the", "{places}"),
    ("a", "{professions}", "{verbs}", "with", "the", "{objects}", "{times}"),
    ("the", "{animals}", "{verbs}", "beside", "a", "{adjectives}", "{places}"),
    ("each", "{professions}", "{verbs}", "a", "{objects}", "inside", "the", "{places}"),
)


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class BaseTokenizer:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError


class WordTokenizer(BaseTokenizer):
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        lowercase: bool = True,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "WordTokenizer":
        counts: Dict[str, int] = {}
        for text in texts:
            tokens = cls._tokenize(text, lowercase)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        kept = [token for token, freq in sorted_tokens if freq >= min_freq]
        if max_vocab_size is not None:
            limit = max(max_vocab_size - len(SPECIAL_TOKENS), 0)
            kept = kept[:limit]
        ordered = SPECIAL_TOKENS + kept
        return cls({token: idx for idx, token in enumerate(ordered)}, lowercase=lowercase)

    def __init__(self, vocab: Dict[str, int], lowercase: bool = True) -> None:
        super().__init__(vocab)
        self.lowercase = lowercase

    @classmethod
    def _tokenize(cls, text: str, lowercase: bool) -> List[str]:
        normalized = text.lower() if lowercase else text
        return cls.TOKEN_PATTERN.findall(normalized)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(token, self.unk_id) for token in self._tokenize(text, self.lowercase)]


class CharTokenizer(BaseTokenizer):
    @classmethod
    def from_texts(cls, texts: Sequence[str], max_vocab_size: int | None = None) -> "CharTokenizer":
        charset = sorted({char for text in texts for char in text})
        if max_vocab_size is not None:
            limit = max(max_vocab_size - len(SPECIAL_TOKENS), 0)
            charset = charset[:limit]
        ordered = SPECIAL_TOKENS + charset
        return cls({token: idx for idx, token in enumerate(ordered)})

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.unk_id) for char in text]


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab: Dict[str, int], merges: Sequence[Sequence[int]], token_bytes: Dict[int, bytes]) -> None:
        super().__init__(vocab)
        self.merges = [tuple(pair) for pair in merges]
        self.merge_ranks = {tuple(pair): rank for rank, pair in enumerate(self.merges)}
        self.token_bytes = token_bytes

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int,
        min_frequency: int = 2,
        model_path: str | None = None,
    ) -> "BPETokenizer":
        token_bytes = {idx: bytes([idx]) for idx in range(BYTE_TOKEN_OFFSET)}
        sequences = [list(text.encode("utf-8")) for text in texts if text]
        merges: List[tuple[int, int]] = []
        next_token_id = BYTE_TOKEN_OFFSET
        target_merges = max(vocab_size - len(SPECIAL_TOKENS) - BYTE_TOKEN_OFFSET, 0)

        for _ in range(target_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            merged_token = next_token_id
            next_token_id += 1
            token_bytes[merged_token] = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
            merges.append(best_pair)
            sequences = [cls._merge_sequence(sequence, best_pair, merged_token) for sequence in sequences]

        vocab = cls._build_vocab(token_bytes)
        tokenizer = cls(vocab=vocab, merges=merges, token_bytes=token_bytes)
        if model_path is not None:
            tokenizer.save(model_path)
        return tokenizer

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = payload["vocab"]
        merges = payload["merges"]
        token_bytes = {int(key): bytes.fromhex(value) for key, value in payload["token_bytes"].items()}
        return cls(vocab=vocab, merges=merges, token_bytes=token_bytes)

    @staticmethod
    def _build_vocab(token_bytes: Dict[int, bytes]) -> Dict[str, int]:
        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for token_id in sorted(token_bytes):
            vocab[f"bpe:{token_id}"] = len(vocab)
        return vocab

    @staticmethod
    def _merge_sequence(sequence: Sequence[int], pair: tuple[int, int], merged_token: int) -> List[int]:
        merged: List[int] = []
        index = 0
        while index < len(sequence):
            if index < len(sequence) - 1 and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
                merged.append(merged_token)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def save(self, path: str) -> None:
        payload = {
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "token_bytes": {str(key): value.hex() for key, value in self.token_bytes.items()},
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def encode(self, text: str) -> List[int]:
        sequence = list(text.encode("utf-8"))
        while len(sequence) > 1:
            candidate_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(sequence, sequence[1:])
                if pair in self.merge_ranks
            ]
            if not candidate_pairs:
                break
            _, best_pair = min(candidate_pairs, key=lambda item: item[0])
            merged_id = BYTE_TOKEN_OFFSET + self.merge_ranks[best_pair]
            sequence = self._merge_sequence(sequence, best_pair, merged_id)
        return [self.vocab.get(f"bpe:{token_id}", self.unk_id) for token_id in sequence]


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sequence = list(self.sequences[index])
        return {"input_ids": sequence[:-1], "labels": sequence[1:]}


def build_toy_vocab() -> Dict[str, int]:
    tokens = set(SPECIAL_TOKENS)
    for words in LEXICON.values():
        tokens.update(words)
    for template in TEMPLATES:
        for token in template:
            if not token.startswith("{"):
                tokens.add(token)
    ordered = SPECIAL_TOKENS + sorted(token for token in tokens if token not in SPECIAL_TOKENS)
    return {token: idx for idx, token in enumerate(ordered)}


def _render_template(rng: random.Random, template: Sequence[str]) -> List[str]:
    rendered: List[str] = []
    for token in template:
        if token.startswith("{") and token.endswith("}"):
            key = token[1:-1]
            rendered.append(rng.choice(list(LEXICON[key])))
        else:
            rendered.append(token)
    return rendered


def generate_sentences(num_samples: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    sentences: List[List[str]] = []
    for _ in range(num_samples):
        template = rng.choice(list(TEMPLATES))
        sentences.append(_render_template(rng, template))
    return sentences


def build_toy_sequences(config: Dict) -> Dict[str, object]:
    data_cfg = config["data"]
    vocab = build_toy_vocab()
    seed = config["seed"]
    train_sentences = generate_sentences(data_cfg["train_samples"], seed)
    val_sentences = generate_sentences(data_cfg["val_samples"], seed + 1)

    def encode_sentence(sentence: Sequence[str]) -> List[int]:
        sequence = [vocab["<bos>"]] + [vocab[token] for token in sentence] + [vocab["<eos>"]]
        sequence = sequence[: data_cfg["max_seq_len"]]
        if sequence[-1] != vocab["<eos>"]:
            sequence[-1] = vocab["<eos>"]
        return sequence

    return {
        "train_sequences": [encode_sentence(sentence) for sentence in train_sentences],
        "val_sequences": [encode_sentence(sentence) for sentence in val_sentences],
        "vocab": vocab,
        "pad_id": vocab["<pad>"],
        "tokenizer_name": "toy-word",
        "dataset_name": "synthetic_templates",
    }


def read_text_lines(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_tokenizer(tokenizer_cfg: Dict, train_texts: Sequence[str]) -> BaseTokenizer:
    level = tokenizer_cfg.get("level", "char")
    if level == "char":
        return CharTokenizer.from_texts(train_texts, max_vocab_size=tokenizer_cfg.get("max_vocab_size"))
    if level == "word":
        return WordTokenizer.from_texts(
            train_texts,
            lowercase=tokenizer_cfg.get("lowercase", True),
            min_freq=tokenizer_cfg.get("min_freq", 1),
            max_vocab_size=tokenizer_cfg.get("max_vocab_size"),
        )
    if level == "bpe":
        model_path = tokenizer_cfg.get("model_path")
        if model_path is not None and Path(model_path).exists():
            return BPETokenizer.from_file(model_path)
        vocab_size = tokenizer_cfg.get("vocab_size", 512)
        min_frequency = tokenizer_cfg.get("min_frequency", 2)
        return BPETokenizer.train(
            train_texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            model_path=model_path,
        )
    raise ValueError(f"Unsupported tokenizer level: {level}")


def segment_tokens(token_ids: Sequence[int], max_seq_len: int, stride: int | None) -> List[List[int]]:
    if max_seq_len < 2:
        raise ValueError("data.max_seq_len must be at least 2")
    chunk_size = max_seq_len
    step = stride or (chunk_size - 1)
    step = max(step, 1)
    if len(token_ids) < 2:
        return []

    sequences: List[List[int]] = []
    start = 0
    while start + 1 < len(token_ids):
        chunk = list(token_ids[start : start + chunk_size])
        if len(chunk) < 2:
            break
        sequences.append(chunk)
        if start + chunk_size >= len(token_ids):
            break
        start += step
    return sequences


def build_text_sequences(config: Dict) -> Dict[str, object]:
    data_cfg = config["data"]
    train_texts = read_text_lines(data_cfg["train_path"])
    val_texts = read_text_lines(data_cfg["val_path"])
    tokenizer_cfg = data_cfg.get("tokenizer", {"level": "char"})
    tokenizer = build_tokenizer(tokenizer_cfg, train_texts)
    stride = data_cfg.get("stride")
    max_seq_len = data_cfg["max_seq_len"]

    def encode_document(text: str) -> List[int]:
        return [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]

    train_sequences: List[List[int]] = []
    for text in train_texts:
        train_sequences.extend(segment_tokens(encode_document(text), max_seq_len, stride))

    val_sequences: List[List[int]] = []
    for text in val_texts:
        val_sequences.extend(segment_tokens(encode_document(text), max_seq_len, stride))

    if not train_sequences or not val_sequences:
        raise ValueError("Text dataset produced no training or validation sequences. Check paths and max_seq_len.")

    return {
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "vocab": tokenizer.vocab,
        "pad_id": tokenizer.pad_id,
        "tokenizer_name": tokenizer_cfg.get("level", "char"),
        "dataset_name": data_cfg.get("name", Path(data_cfg["train_path"]).parent.name or "text_corpus"),
    }


def collate_batch(features: Sequence[Dict[str, List[int]]], pad_id: int) -> Batch:
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(features), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(features), max_len), dtype=torch.bool)

    for row, feature in enumerate(features):
        length = len(feature["input_ids"])
        input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        attention_mask[row, :length] = True

    return Batch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def create_dataloaders(config: Dict) -> Dict[str, object]:
    data_cfg = config["data"]
    dataset_type = data_cfg.get("type", "toy")
    if dataset_type == "toy":
        dataset_bundle = build_toy_sequences(config)
    elif dataset_type == "text":
        dataset_bundle = build_text_sequences(config)
    else:
        raise ValueError(f"Unsupported data.type: {dataset_type}")

    collate = lambda batch: collate_batch(batch, dataset_bundle["pad_id"])
    train_dataset = SequenceDataset(dataset_bundle["train_sequences"])
    val_dataset = SequenceDataset(dataset_bundle["val_sequences"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "vocab": dataset_bundle["vocab"],
        "vocab_size": len(dataset_bundle["vocab"]),
        "pad_id": dataset_bundle["pad_id"],
        "dataset_name": dataset_bundle["dataset_name"],
        "tokenizer_name": dataset_bundle["tokenizer_name"],
    }
