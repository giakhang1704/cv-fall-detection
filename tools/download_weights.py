"""Download the two pretrained files the pipeline needs into `weights/` and verify their SHA-256.

    python tools/download_weights.py

Both files come from the official Ultralytics assets release v8.4.0.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
FILES = {
    "yolo26n-pose.pt": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    "yolo26n-reid.onnx": "8529c383197ae4c468eda535d1b165f8b4162cf17bf5fbcff49c7cb6455bc0bb",
}


def sha256(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    """Download any missing weight file and verify its SHA-256."""
    target = ROOT / "weights"
    target.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        path = target / name
        if path.exists() and sha256(path) == expected:
            print(f"ok       {path.relative_to(ROOT)}")
            continue
        print(f"download {name} ...")
        urllib.request.urlretrieve(BASE + name, path)
        if sha256(path) != expected:
            path.unlink()
            sys.exit(f"SHA-256 mismatch for {name}; file removed")
        print(f"ok       {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
