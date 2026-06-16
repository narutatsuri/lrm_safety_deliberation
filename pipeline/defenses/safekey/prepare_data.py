#!/usr/bin/env python3
"""Prepare SafeKey training data by downloading the canonical mix2k file."""

from __future__ import annotations

import argparse
import json
import os

import requests


DEFAULT_URL = "https://raw.githubusercontent.com/eric-ai-lab/SafeKey/main/data/train/sft_mix_2k.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SafeKey sft_mix_2k training data")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        default="defenses/safekey/data/sft_mix_2k.json",
        help="Output JSON file (list[dict])",
    )
    args = parser.parse_args()

    resp = requests.get(args.url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON from {args.url}, got {type(data)}")

    required = {"question", "response", "source", "summary_end_idx", "next_sentence_end_idx"}
    missing = sorted(required - set(data[0].keys())) if data else sorted(required)
    if missing:
        raise ValueError(f"Downloaded data missing required fields: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} rows to {args.output}")
    if data:
        print(f"Example keys: {list(data[0].keys())}")


if __name__ == "__main__":
    main()

