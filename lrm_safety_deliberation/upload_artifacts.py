"""Upload the intermediate-artifact bundle to the HuggingFace Hub.

Companion to :mod:`lrm_safety_deliberation.fetch_artifacts`: pushes the artifact subtrees
listed in that module's ``MANIFEST`` from a local source tree (the research repo
or any tree laid out the same way) to the dataset repo, preserving the relative
paths so :func:`fetch_artifacts.fetch` can pull them straight back.

Requires authentication (``huggingface-cli login``) and write access to the repo.

Usage::

    # upload the cheap "figures" tier (a few GB) from the research tree:
    python -m lrm_safety_deliberation.upload_artifacts --source /path/to/research_repo --tier figures
    # everything (hundreds of GB):
    python -m lrm_safety_deliberation.upload_artifacts --source /path/to/research_repo --tier full
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fetch_artifacts import DEFAULT_REPO, MANIFEST, patterns_for


def _source_root(source: str | None) -> Path:
    from .paths import ARTIFACTS_ROOT
    root = Path(source).expanduser().resolve() if source else ARTIFACTS_ROOT
    if not root.exists():
        raise SystemExit(f"source tree does not exist: {root}")
    return root


def upload(tier: str = "figures", repo: str | None = None, source: str | None = None,
           dry_run: bool = False, workers: int = 4) -> None:
    repo = repo or DEFAULT_REPO
    root = _source_root(source)
    patterns = patterns_for(tier)

    # Report what will be sent (and surface any empty patterns up front).
    print(f"Source: {root}\nRepo:   {repo} (dataset)\nTier:   {tier}\n")
    total_files = total_bytes = 0
    for b in MANIFEST:
        if b["tier"] not in (("figures",) if tier == "figures" else ("figures", "full")):
            continue
        files = [f for pat in b["patterns"] for f in root.glob(pat) if f.is_file()]
        nbytes = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_bytes += nbytes
        flag = "" if files else "  <-- NO MATCHES on this source tree"
        print(f"  {b['name']:22s} {len(files):6d} files  {nbytes/1e9:7.2f} GB{flag}")
    print(f"\n  TOTAL {total_files} files, {total_bytes/1e9:.2f} GB")

    if dry_run:
        print("\n[dry-run] nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required: pip install huggingface-hub") from exc

    api = HfApi()
    print(f"\nUploading -> {repo} ...")
    # For large volumes (the full tier is hundreds of GB) use the resumable,
    # multi-threaded, batched-commit uploader recommended by the Hub.
    if total_bytes > 20e9 and hasattr(api, "upload_large_folder"):
        # Low worker count keeps memory bounded for hundreds-of-GB uploads on a
        # shared/login node (resumable: re-run to continue after any interruption).
        api.upload_large_folder(
            repo_id=repo, repo_type="dataset",
            folder_path=str(root), allow_patterns=patterns,
            num_workers=workers, print_report=False,
        )
    else:
        api.upload_folder(
            folder_path=str(root), repo_id=repo, repo_type="dataset",
            allow_patterns=patterns,
            commit_message=f"Upload {tier}-tier intermediate artifacts",
        )
    print("done. Consumers can now run `python -m lrm_safety_deliberation.fetch_artifacts --tier " + tier + "`.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["figures", "full"], default="figures")
    ap.add_argument("--repo", default=None, help="dataset repo id (overrides default)")
    ap.add_argument("--source", default=None,
                    help="local tree containing the artifacts (default: ARTIFACTS_ROOT)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, then stop")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel workers for the large-folder uploader (lower = less memory)")
    args = ap.parse_args()
    upload(args.tier, args.repo, args.source, args.dry_run, args.workers)


if __name__ == "__main__":
    main()
