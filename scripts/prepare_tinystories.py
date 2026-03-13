#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Iterator


TEXT_KEYS = ("story", "text", "content")


def normalize_story(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def iter_story_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.json", "*.jsonl"):
        yield from root.rglob(pattern)


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    elif archive_path.suffix in {".gz", ".tar", ".tgz", ".xz"} or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path) as archive:
            archive.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    return target_dir


def read_stories(root: Path) -> list[str]:
    stories: list[str] = []
    for file_path in iter_story_files(root):
        if file_path.suffix == ".jsonl":
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                for key in TEXT_KEYS:
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        stories.append(normalize_story(value))
                        break
        else:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            rows: Iterable[dict] = payload if isinstance(payload, list) else [payload]
            for row in rows:
                for key in TEXT_KEYS:
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        stories.append(normalize_story(value))
                        break
    if not stories:
        raise ValueError(f"No stories found under {root}")
    return stories


def write_split(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize TinyStories JSON shards into train/val/test text files.")
    parser.add_argument("--output-dir", default="data/tinystories", help="Destination directory for train/val/test text files.")
    parser.add_argument("--source-dir", default=None, help="Directory containing TinyStories JSON or JSONL shards.")
    parser.add_argument("--archive", default=None, help="Local archive containing TinyStories shards.")
    parser.add_argument("--download-url", default=None, help="Optional URL to download a TinyStories archive before extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic shuffling before the split.")
    parser.add_argument("--val-size", type=float, default=0.01, help="Validation split fraction.")
    parser.add_argument("--test-size", type=float, default=0.01, help="Test split fraction.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        if args.source_dir:
            source_root = Path(args.source_dir)
        elif args.archive:
            source_root = extract_archive(Path(args.archive), tmp_root / "tinystories")
        elif args.download_url:
            archive_name = args.download_url.rsplit("/", 1)[-1] or "tinystories.tar.gz"
            archive_path = tmp_root / archive_name
            print(f"downloading {args.download_url} -> {archive_path}")
            urllib.request.urlretrieve(args.download_url, archive_path)
            source_root = extract_archive(archive_path, tmp_root / "tinystories")
        else:
            raise ValueError("Provide --source-dir, --archive, or --download-url.")

        stories = read_stories(source_root)
        rng = random.Random(args.seed)
        rng.shuffle(stories)

        total = len(stories)
        test_count = max(int(total * args.test_size), 1)
        val_count = max(int(total * args.val_size), 1)
        train_count = max(total - val_count - test_count, 1)
        train_rows = stories[:train_count]
        val_rows = stories[train_count:train_count + val_count]
        test_rows = stories[train_count + val_count:]

        write_split(output_dir / "train.txt", train_rows)
        write_split(output_dir / "val.txt", val_rows)
        write_split(output_dir / "test.txt", test_rows)

    print(f"wrote {output_dir / 'train.txt'}")
    print(f"wrote {output_dir / 'val.txt'}")
    print(f"wrote {output_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
