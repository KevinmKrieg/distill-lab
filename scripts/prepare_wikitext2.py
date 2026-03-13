#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def normalize_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


def find_split_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or normalize WikiText-2 into train/val/test text files.")
    parser.add_argument("--output-dir", default="data/wikitext2", help="Destination directory for train/val/test text files.")
    parser.add_argument("--source-dir", default=None, help="Existing extracted WikiText-2 directory. Skips download when provided.")
    parser.add_argument("--download-url", default="https://wikitext.s3.amazonaws.com/wikitext-2-v1.zip", help="URL for the WikiText-2 zip archive.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        working_root = Path(tmpdir)
        search_root = Path(args.source_dir) if args.source_dir else working_root

        if args.source_dir is None:
            archive_path = working_root / "wikitext-2-v1.zip"
            print(f"downloading {args.download_url} -> {archive_path}")
            urllib.request.urlretrieve(args.download_url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(working_root)

        train_text = normalize_lines(find_split_file(search_root, "wiki.train.tokens").read_text(encoding="utf-8"))
        val_text = normalize_lines(find_split_file(search_root, "wiki.valid.tokens").read_text(encoding="utf-8"))
        test_text = normalize_lines(find_split_file(search_root, "wiki.test.tokens").read_text(encoding="utf-8"))

        (output_dir / "train.txt").write_text(train_text, encoding="utf-8")
        (output_dir / "val.txt").write_text(val_text, encoding="utf-8")
        (output_dir / "test.txt").write_text(test_text, encoding="utf-8")

    print(f"wrote {output_dir / 'train.txt'}")
    print(f"wrote {output_dir / 'val.txt'}")
    print(f"wrote {output_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
