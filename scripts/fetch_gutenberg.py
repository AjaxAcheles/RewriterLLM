"""fetch_gutenberg.py — download ~500 Project Gutenberg books for the M2 corpus.

Downloads the sedthh/gutenberg_english HuggingFace dataset (streaming) and writes
each book as a UTF-8 .txt file to data/raw_texts/.

Usage:
    python scripts/fetch_gutenberg.py [--n-books 500] [--out-dir data/raw_texts]
"""
import argparse
from pathlib import Path
from datasets import load_dataset


def _text_field(row):
    for k in ("TEXT", "text", "content", "body"):
        if isinstance(row.get(k), str):
            return k
    str_keys = [k for k, v in row.items() if isinstance(v, str)]
    return max(str_keys, key=lambda k: len(row[k])) if str_keys else None


def main(n_books=500, out_dir="data/raw_texts", dataset="sedthh/gutenberg_english"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset, split="train", streaming=True)
    field, i = None, 0
    for i, row in enumerate(ds):
        if i >= n_books:
            break
        field = field or _text_field(row)
        if not field:
            raise SystemExit(f"No text field found; row keys = {list(row)}")
        Path(f"{out_dir}/{i:05d}.txt").write_text(row[field], errors="ignore")
        if (i + 1) % 50 == 0:
            print(f"  Downloaded {i + 1} books...")
    print(f"Downloaded {min(i + 1, n_books)} books to {out_dir} (field='{field}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-books", type=int, default=500)
    parser.add_argument("--out-dir", default="data/raw_texts")
    args = parser.parse_args()
    main(args.n_books, args.out_dir)
