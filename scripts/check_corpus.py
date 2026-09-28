from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from src.config import RAW_DIR
from src.loaders import load_documents


def main() -> None:
    files = [p for p in RAW_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")]
    print(f"Files: {len(files)}")
    print("By extension:", dict(Counter(p.suffix.lower() for p in files)))
    units = load_documents(RAW_DIR)
    chars = sum(len(x["text"]) for x in units)
    print(f"Extracted pages/units: {len(units)}")
    print(f"Extracted characters: {chars:,}")
    if len(files) < 20:
        print("WARNING: capstone requires 20-50 high-quality documents.")


if __name__ == "__main__":
    main()
