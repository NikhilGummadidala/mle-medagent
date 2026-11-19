"""Download the third-party open-source weights MedAgent uses.

The weights are too large for an ordinary Git file (GitHub rejects files over
100 MB), so they are fetched from Hugging Face at pinned revisions and verified
against a recorded SHA-256 before being moved into ``saved_models/``.

    python scripts/download_models.py            # fetch anything missing
    python scripts/download_models.py --verify   # only check what is on disk

Only ``.safetensors`` weights and JSON configs are downloaded. No remote code
is executed: the architectures are implemented in ``models/``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "saved_models"

# (local dir, repo, pinned revision, {filename: sha256 or None for small text files})
SOURCES = [
    (
        "mammoscreen",
        "ianpan/mammoscreen",
        "56b289045f71d26d85b7ecf0d204469ec8817c27",
        {
            "config.json": None,
            "model.safetensors": "8ace0998e2d534b08eaa673fad37df16eaffef6dca3f3b5ac2196857a2949596",
        },
    ),
    (
        "mammo_crop",
        "ianpan/mammo-crop",
        "bec2e302d47176d743a86d276d8402f876ad0f9f",
        {
            "config.json": None,
            "model.safetensors": "57bd1f259c2b1588d96953195665bca3972e0954194d8eaeaf2bd12af47591d2",
        },
    ),
    (
        "skin_vit",
        "Anwarkh1/Skin_Cancer-Image_Classification",
        "e37ebda4a662db26d9221c78f6c72b1cb8736ce0",
        {
            "config.json": None,
            "model.safetensors": "b8136bbbdd085cc97f69bdef01bd0c4c54262cf3ba44f5a94ebc9c8c54e85d60",
        },
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "model-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done >= next_report:
                print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                next_report += max(total // 5, 1)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="check files on disk, download nothing")
    args = parser.parse_args()

    failures = 0
    for local, repo, revision, files in SOURCES:
        target_dir = MODEL_DIR / local
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, expected in files.items():
            path = target_dir / name
            if not path.exists():
                if args.verify:
                    print(f"MISSING  {path.relative_to(ROOT)}")
                    failures += 1
                    continue
                print(f"Downloading {repo}@{revision[:8]} {name}", flush=True)
                fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", path)
            if expected is None:
                print(f"OK       {path.relative_to(ROOT)}")
                continue
            actual = sha256(path)
            if actual != expected:
                print(f"BAD HASH {path.relative_to(ROOT)}\n  expected {expected}\n  actual   {actual}")
                failures += 1
                if not args.verify:
                    path.unlink()
            else:
                print(f"OK       {path.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
