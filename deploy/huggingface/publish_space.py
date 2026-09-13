"""
publish_space.py

Package the QueueIQ Vision API as a Hugging Face Space (Docker SDK) and upload
it. Only what the server needs at runtime is sent — no dataset, notebooks or
training scripts — and the model weights go through the Hub's large-file
storage automatically, so Git LFS does not have to be set up by hand.

Use the same Python for all three commands, and log in once with a token that
has "write" access (huggingface.co/settings/tokens). The script can be run from
any folder:

    python -m pip install huggingface_hub
    python -c "from huggingface_hub import login; login()"
    python D:/QueueIQ/QueueIQ-AI/deploy/huggingface/publish_space.py --space <owner>/queueiq-api

Check what would be uploaded without touching the Hub:

    python deploy/huggingface/publish_space.py --space <owner>/queueiq-api --dry-run

The two demo videos are third-party footage and are left out unless
--include-videos is given, because a public Space lets anyone download them.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CODE = ["server.py", "queueiq_engine.py", "requirements.txt", "requirements-server.txt",
        "synthetic_checkout_data.csv"]
WEIGHTS_REQUIRED = ["yolov8n.pt", "fullness_classifier.pt", "class_names.txt"]
WEIGHTS_OPTIONAL = ["basket_world.pt", "basket_detector.pt"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def space_url(space: str) -> str:
    """https://<owner>-<name>.hf.space — Hugging Face lowercases the id and turns
    underscores and dots into hyphens."""
    owner, name = space.split("/", 1)
    slug = re.sub(r"[^a-z0-9-]", "-", f"{owner}-{name}".lower())
    return f"https://{slug}.hf.space"


def stage(dest: Path, include_videos: bool) -> list[tuple[str, int]]:
    files: list[Path] = []
    missing = [n for n in CODE + WEIGHTS_REQUIRED if not (ROOT / n).exists()]
    if missing:
        sys.exit(f"Missing from {ROOT}: {', '.join(missing)}. "
                 "Run python fetch_models.py first.")
    files += [ROOT / n for n in CODE + WEIGHTS_REQUIRED]
    files += [ROOT / n for n in WEIGHTS_OPTIONAL if (ROOT / n).exists()]
    files += [p for p in sorted((ROOT / "examples").glob("*"))
              if p.suffix.lower() in IMAGE_EXT and "_annotated" not in p.name]
    if include_videos:
        files += sorted((ROOT / "videos").glob("*.mp4"))

    listing = []
    for src in files:
        rel = src.relative_to(ROOT)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        listing.append((str(rel).replace("\\", "/"), src.stat().st_size))
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(HERE / name, dest / name)
        listing.append((name, (HERE / name).stat().st_size))
    return listing


def preflight() -> None:
    """Fail before staging anything when an upload cannot possibly work, and say
    exactly what to run - with the interpreter that is actually running, since
    the usual cause is a second Python on PATH without huggingface_hub."""
    py = f'"{sys.executable}"'
    nl = chr(10)
    try:
        from huggingface_hub import get_token
    except ImportError:
        sys.exit(nl.join([
            "huggingface_hub is not installed for this Python:",
            f"  {sys.executable}",
            "",
            "Install it for this interpreter, or run the script with the Python that already has it:",
            f"  {py} -m pip install huggingface_hub",
        ]))
    if not get_token():
        sys.exit(nl.join([
            "Not logged in to Hugging Face.",
            "",
            "Create a token with WRITE access at https://huggingface.co/settings/tokens,",
            "then log in once with this same Python (paste the token when asked):",
            f'  {py} -c "from huggingface_hub import login; login()"',
            "",
            "Then run this script again.",
        ]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish the QueueIQ Vision API to a Hugging Face Space")
    ap.add_argument("--space", required=True, help="<owner>/<space-name>, e.g. dmtech/queueiq-api")
    ap.add_argument("--include-videos", action="store_true",
                    help="also upload videos/*.mp4 (third-party footage)")
    ap.add_argument("--dry-run", action="store_true", help="stage and list files, upload nothing")
    args = ap.parse_args()

    if "/" not in args.space:
        sys.exit("--space must look like <owner>/<space-name>")

    if not args.dry_run:
        preflight()

    with tempfile.TemporaryDirectory(prefix="queueiq-space-") as tmp:
        listing = stage(Path(tmp), args.include_videos)
        total = sum(size for _, size in listing)
        print(f"Space: {args.space}\n")
        for rel, size in listing:
            print(f"  {size / 1048576:8.2f} MB  {rel}")
        print(f"  {total / 1048576:8.2f} MB  total, {len(listing)} files\n")

        if args.dry_run:
            print("Dry run — nothing uploaded.")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space",
                          commit_message="Deploy QueueIQ Vision API")

    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"API URL once the build finishes: {space_url(args.space)}")
    print(f"Check it with: curl {space_url(args.space)}/health")
    print(f"Build the dashboard with: VITE_QUEUEIQ_API={space_url(args.space)} npm run build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
