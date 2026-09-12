"""
fetch_models.py

Put the model weights in place on a machine that does not have them — a server,
a fresh clone, or a laptop where the weights were kept outside the repository.

Weights are resolved at runtime from QUEUEIQ_MODELS_DIR (falling back to this
folder), so on a server they can live on a mounted volume instead of inside the
git checkout:

    export QUEUEIQ_MODELS_DIR=/opt/queueiq/models
    python fetch_models.py

Two of the five files are derived from public Ultralytics weights and are
rebuilt locally with no credentials:

    yolov8n.pt        downloaded by Ultralytics on first use
    basket_world.pt   YOLO-World-S with the vocabulary ["shopping basket",
                      "shopping cart"] baked in and the CLIP text encoder
                      stripped, so no text model is needed at inference time

The other three are trained artifacts of this project and cannot be rebuilt
without the dataset, so they have to come from somewhere you control:

    fullness_classifier.pt, class_names.txt, basket_detector.pt

Point QUEUEIQ_MODELS_URL at a directory that serves them over HTTP — an Oracle
Object Storage pre-authenticated request works well, so does any static host:

    export QUEUEIQ_MODELS_URL=https://objectstorage.<region>.oraclecloud.com/p/<token>/n/<ns>/b/queueiq-models/o
    python fetch_models.py

Without a URL the script reports what is missing and stops; copying the three
files over with scp is a perfectly good alternative.

USAGE:
    python fetch_models.py            # rebuild what can be rebuilt, fetch the rest
    python fetch_models.py --check    # report only, download nothing
    python fetch_models.py --force    # re-create files even when they are present
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("QUEUEIQ_MODELS_DIR") or ROOT).expanduser()
MODELS_URL = os.environ.get("QUEUEIQ_MODELS_URL", "").rstrip("/")

# sha256 and size of the weights this revision was tested with. A file that
# differs is reported but never overwritten silently - a retrained model is a
# deliberate change, not corruption.
MANIFEST: dict[str, tuple[str, int]] = {
    "yolov8n.pt": ("f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36", 6549796),
    "basket_world.pt": ("192a927d8699b08c05e18aef56a66a8603ef03067564243e6f4fc2b48d4d3803", 25761664),
    "basket_detector.pt": ("303085a3a4b7ba02492ecbe13ff318ff1ed2d986947b37ea1b69aad7789f0ef6", 6248874),
    "fullness_classifier.pt": ("76cebaf8bc67c22c1c3d2fb579f892247f59da8d9a6657e3f025b75af08d08e0", 9165259),
    "class_names.txt": ("2513b0b1bc102596814f630316bb8b2b0824077499502b5309a798c9a8f903c1", 48),
}

# Files the server needs for tier "full". basket_detector.pt is the lighter
# fallback detector and is optional once basket_world.pt is present.
REQUIRED = ["yolov8n.pt", "fullness_classifier.pt", "class_names.txt"]
DERIVABLE = ["yolov8n.pt", "basket_world.pt"]   # rebuilt locally from public weights


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(name: str) -> str:
    """present | modified | missing, for the file the server would actually load.

    Rebuilt weights are not byte-identical to the ones this revision was tested
    with (serialisation embeds timestamps), so the checksum is only enforced for
    the trained artifacts, which must arrive unchanged."""
    for folder in (MODELS_DIR, ROOT):
        p = folder / name
        if p.exists():
            want = MANIFEST.get(name)
            if name in DERIVABLE or not want or sha256(p) == want[0]:
                return "present"
            return "modified"
    return "missing"


def rebuild_yolov8n(dest: Path) -> bool:
    from ultralytics import YOLO

    YOLO("yolov8n.pt")  # Ultralytics downloads into the working directory
    src = Path("yolov8n.pt")
    if not src.exists():
        return False
    if src.resolve() != dest.resolve():
        shutil.move(str(src), str(dest))
    return dest.exists()


def rebuild_basket_world(dest: Path) -> bool:
    """YOLO-World-S with our two prompts baked in, minus the CLIP text encoder
    (330 MB -> 26 MB, and nothing has to embed text at inference time)."""
    from ultralytics import YOLO

    world = YOLO("yolov8s-worldv2.pt")
    world.set_classes(["shopping basket", "shopping cart"])
    for key in [k for k in world.model._modules if "clip" in k.lower()]:
        del world.model._modules[key]
    world.save(str(dest))
    # the 330 MB source checkpoint is only needed to build the file above
    Path("yolov8s-worldv2.pt").unlink(missing_ok=True)
    return dest.exists()


REBUILDERS = {"yolov8n.pt": rebuild_yolov8n, "basket_world.pt": rebuild_basket_world}


def download(name: str, dest: Path) -> bool:
    url = f"{MODELS_URL}/{name}"
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  failed: {e}")
        return False
    want = MANIFEST.get(name)
    if want and name not in DERIVABLE and sha256(tmp) != want[0]:
        print("  checksum does not match the manifest — keeping the download as "
              f"{tmp.name} so you can inspect it")
        return False
    tmp.replace(dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch or rebuild the QueueIQ model weights")
    ap.add_argument("--check", action="store_true", help="report only, download nothing")
    ap.add_argument("--force", action="store_true", help="re-create files that are already present")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"models directory : {MODELS_DIR}")
    print(f"download base    : {MODELS_URL or '(not set)'}\n")

    states = {name: status(name) for name in MANIFEST}
    for name, state in states.items():
        mark = {"present": "ok     ", "modified": "differs", "missing": "MISSING"}[state]
        print(f"  {mark}  {name}")
    print()

    if args.check:
        missing = [n for n in REQUIRED if states[n] == "missing"]
        print("tier full is possible" if not missing else f"missing for tier full: {missing}")
        return 1 if missing else 0

    failed = []
    for name in MANIFEST:
        if states[name] != "missing" and not args.force:
            continue
        dest = MODELS_DIR / name
        print(f"{name}:")
        if name in DERIVABLE:
            try:
                if REBUILDERS[name](dest):
                    print("  rebuilt locally")
                    continue
            except Exception as e:
                print(f"  rebuild failed: {e}")
        if MODELS_URL and download(name, dest):
            print("  downloaded")
            continue
        failed.append(name)
        print("  could not be obtained")

    if failed:
        print("\nStill missing: " + ", ".join(failed))
        if not MODELS_URL:
            print("Set QUEUEIQ_MODELS_URL to a directory serving these files, or copy "
                  "them across manually, for example:")
            print(f"  scp {' '.join(failed)} user@server:{MODELS_DIR}/")
        return 1

    print("\nAll weights are in place. Start the server with: python server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
