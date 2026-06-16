#!/usr/bin/env python3
"""Prepare STAR-1 training data locally from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save STAR-1 dataset as JSON")
    parser.add_argument("--dataset", default="UCSC-VLAA/STAR-1")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        default="defenses/star1/data/STAR-1.json",
        help="Output JSON file (list[dict])",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(x) for x in ds]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} rows to {args.output}")
    if rows:
        print(f"Example keys: {list(rows[0].keys())}")


if __name__ == "__main__":
    main()

