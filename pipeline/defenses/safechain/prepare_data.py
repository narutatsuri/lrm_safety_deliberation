#!/usr/bin/env python3
"""Prepare SafeChain training data locally from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize SafeChain dataset")
    parser.add_argument("--dataset", default="UWNSL/SafeChain")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        default="defenses/safechain/data/SafeChain-train.json",
        help="Output JSON file (list[dict])",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help=(
            "Optional label filter. Example: --labels adversarial_harmful vanilla_harmful. "
            "Default keeps all labels."
        ),
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rows = [dict(x) for x in ds]

    if args.labels:
        wanted = set(args.labels)
        rows = [r for r in rows if r.get("label") in wanted]

    normalized = []
    for i, r in enumerate(rows):
        normalized.append(
            {
                "id": r.get("id", i),
                "question": r.get("instruction", ""),
                "response": r.get("response", ""),
                "label": r.get("label", ""),
                "source": "SafeChain",
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(normalized)} rows to {args.output}")
    if normalized:
        labels = sorted({r["label"] for r in normalized})
        print(f"Labels: {labels}")
        print(f"Example keys: {list(normalized[0].keys())}")


if __name__ == "__main__":
    main()

