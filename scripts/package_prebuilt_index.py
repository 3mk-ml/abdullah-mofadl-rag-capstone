"""Package the precomputed Chroma and BM25 indexes for Streamlit hosting."""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prebuilt_index import ARCHIVE_NAME, _index_ready  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
DIST_DIR = ROOT / "dist"


def main() -> None:
    if not _index_ready(INDEX_DIR):
        raise SystemExit("Missing or incomplete RAG index. Run scripts/ingest.py first.")

    DIST_DIR.mkdir(exist_ok=True)
    archive_path = DIST_DIR / ARCHIVE_NAME
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(INDEX_DIR.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                bundle.write(path, path.relative_to(ROOT).as_posix())

    with archive_path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    checksum_path = archive_path.with_name(archive_path.name + ".sha256")
    checksum_path.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")
    print(f"Index archive: {archive_path} ({archive_path.stat().st_size:,} bytes)")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
