"""
Downloads the three used sub-datasets from a personally hosted LRZ folder
- avoids unnecessary data being downloaded
- avoids huge space consumption while unpacking data

Files are saved into data/raw/NCMAPSS by default (override with --dest)
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Personal TUM-Hosted LRZ folder
BASE_URL = "https://syncandshare.lrz.de/dl/fiSVapJH4tXupCuLPHZh4Z/"

FILES = [
    "N-CMAPSS_DS01.h5",
    "N-CMAPSS_DS02.h5",
    "N-CMAPSS_DS03.h5",
]

CHUNK = 8 << 20


# Laedt eine gegebene Datei runter und prueft auf Dopplungen
def download(name: str, dest: Path) -> None:
    target = dest / name
    if target.exists():  # check für dopplungen
        print(f"  {name} already present, skipping")
        return

    # erst in eine .part-Datei schreiben und nach vollstaendigem Download umbenennen
    partial = target.with_name(target.name + ".part")
    url = BASE_URL + name
    print(f"  downloading {name} ...", flush=True)
    try:
        with (
            urllib.request.urlopen(url, timeout=300) as response,
            open(partial, "wb") as handle,
        ):  # Download
            written = 0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(target)
    print(f"    wrote {target} ({written / 1e9:.2f} GB)")


# runterladen im gewollten datei pfad
def fetch(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, dest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/NCMAPSS"))
    args = parser.parse_args()
    return fetch(args.dest)


if __name__ == "__main__":
    sys.exit(main())
