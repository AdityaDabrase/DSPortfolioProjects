#!/usr/bin/env python3
"""Download all NA telecom pipeline source data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.crtc_ingest import ingest_crtc_mobile
from src.ingest.fcc_ingest import ingest_fcc_county
from src.ingest.generate_subscriptions import generate_subscriptions


def main() -> int:
    print("Downloading CRTC retail mobile data...")
    crtc = ingest_crtc_mobile()
    for name, path in crtc.items():
        print(f"  ok  crtc/{name}: {path.relative_to(ROOT)}")

    print("\nDownloading FCC county connection data...")
    fcc = ingest_fcc_county()
    print(f"  ok  fcc: {fcc.relative_to(ROOT)}")

    print("\nGenerating synthetic subscription snapshot...")
    subs = generate_subscriptions()
    print(f"  ok  subscriptions: {subs.relative_to(ROOT)}")

    print("\nAll sources ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
