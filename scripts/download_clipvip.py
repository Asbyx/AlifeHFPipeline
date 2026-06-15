"""Download the CLIP-ViP B/32 vision weights (clipvip_32.pt) used by the
particlelife and lenia rewarders.

The file is too large to track in git, so it is hosted on the Hugging Face Hub.
This script fetches it via the public direct-download URL using only the Python
standard library, so it works on a fresh clone before any dependencies are
installed.

Usage:
    uv run scripts/download_clipvip.py
    # or
    python scripts/download_clipvip.py

It is idempotent: if the weights already exist with the expected SHA256, it
exits immediately without re-downloading. Pass --force to download anyway.
"""

import argparse
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Direct-download URL for the file in the public HF repo. `resolve/main` serves
# the raw bytes (git-LFS content), so no huggingface_hub / git-lfs is needed.
URL = "https://huggingface.co/frotaur/clipvip-32-rlhfalife/resolve/main/clipvip_32.pt"

# Destination is the path both the particlelife and lenia rewarders load from.
# Resolved relative to the repository root (this file lives in <root>/scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "profiles" / "particlelife" / "rewarder" / "clipvip_32.pt"

EXPECTED_SHA256 = "933c73557d06cd068556f19364bbc61c5e817d10571a248bf6accb45af89c5b8"
EXPECTED_BYTES = 349942024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, showing a simple progress bar.

    Downloads to a temporary file in the destination directory first, then
    atomically renames, so an interrupted download never leaves a corrupt file
    at the final path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "rlhfalife-downloader"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        tmp = tempfile.NamedTemporaryFile(
            delete=False, dir=dest.parent, prefix=dest.name + ".", suffix=".part"
        )
        try:
            with tmp:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total)
            sys.stdout.write("\n")
            Path(tmp.name).replace(dest)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise


def _print_progress(downloaded: int, total: int) -> None:
    mb = downloaded / (1 << 20)
    if total:
        pct = downloaded / total * 100
        bar_len = 30
        filled = int(bar_len * downloaded / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {mb:7.1f} / {total / (1 << 20):.1f} MB"
        )
    else:
        sys.stdout.write(f"\r  {mb:7.1f} MB downloaded")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid file already exists.",
    )
    args = parser.parse_args()

    if DEST.exists() and not args.force:
        print(f"Found existing weights at {DEST}")
        print("Verifying checksum...")
        actual = sha256_of(DEST)
        if actual == EXPECTED_SHA256:
            print("Checksum OK - nothing to do. (Use --force to re-download.)")
            return 0
        print(
            f"  WARNING: checksum mismatch (got {actual[:12]}..., "
            f"expected {EXPECTED_SHA256[:12]}...). Re-downloading."
        )

    print(f"Downloading clipvip_32.pt (~{EXPECTED_BYTES / (1 << 20):.0f} MB)")
    print(f"  from {URL}")
    print(f"  to   {DEST}")
    try:
        download(URL, DEST)
    except urllib.error.HTTPError as e:
        print(f"\nERROR: download failed with HTTP {e.code} ({e.reason}).", file=sys.stderr)
        if e.code in (401, 403, 404):
            print(
                "  The repo may be private or the URL may have changed. Check that\n"
                "  https://huggingface.co/frotaur/clipvip-32-rlhfalife is public.",
                file=sys.stderr,
            )
        return 1
    except urllib.error.URLError as e:
        print(f"\nERROR: network error: {e.reason}.", file=sys.stderr)
        print("  Check your internet connection and try again.", file=sys.stderr)
        return 1

    print("Verifying checksum...")
    actual = sha256_of(DEST)
    if actual != EXPECTED_SHA256:
        print(
            f"ERROR: checksum mismatch after download.\n"
            f"  got      {actual}\n"
            f"  expected {EXPECTED_SHA256}\n"
            f"  The downloaded file may be corrupt. Delete {DEST} and retry.",
            file=sys.stderr,
        )
        return 1

    print(f"Done. Weights ready at {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
