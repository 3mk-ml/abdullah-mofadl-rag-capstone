from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "sources_manifest.csv"
DEFAULT_OUT = ROOT / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/153 Safari/537.36 "
        "AgriRAG-Capstone/1.0"
    )
}


def safe_name(source_id: str, title: str, suffix: str) -> str:
    title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
    return f"{source_id}_{title[:90]}{suffix}"


def get(session: requests.Session, url: str, *, timeout: int = 90) -> requests.Response:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(last)


def extract_readable_text(html: str, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else page_url
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    lines = []
    last = None
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line == last:
            continue
        lines.append(line)
        last = line

    return f"# {title}\n\nSource URL: {page_url}\n\n" + "\n\n".join(lines)


def collect_fao_book(session: requests.Session, index_url: str) -> tuple[str, list[str]]:
    index = get(session, index_url)
    soup = BeautifulSoup(index.text, "html.parser")
    parsed = urlparse(index_url)
    index_name = Path(parsed.path).name.lower()
    prefix = re.sub(r"00\.html?$", "", index_name)
    directory = str(Path(parsed.path).parent).replace("\\", "/").rstrip("/") + "/"

    urls = {index_url}
    for a in soup.find_all("a", href=True):
        absolute = urljoin(index_url, a["href"])
        p = urlparse(absolute)
        if p.netloc.lower() != parsed.netloc.lower():
            continue
        name = Path(p.path).name.lower()
        if not name.endswith((".htm", ".html")):
            continue
        if not p.path.startswith(directory):
            continue
        if prefix and not name.startswith(prefix):
            continue
        urls.add(absolute.split("#", 1)[0])

    ordered = sorted(urls, key=lambda u: Path(urlparse(u).path).name.lower())
    chapters = []
    used = []
    for i, url in enumerate(ordered, start=1):
        try:
            r = get(session, url)
            chapters.append(f"\n\n---\n\n## Document part {i}\n\n" + extract_readable_text(r.text, url))
            used.append(url)
        except Exception as exc:
            print(f"  warning: skipped chapter {url}: {exc}")

    return "".join(chapters).strip(), used


def save_pdf(session: requests.Session, url: str, path: Path) -> tuple[int, str]:
    r = get(session, url)
    content = r.content
    ctype = (r.headers.get("content-type") or "").lower()
    if not content.startswith(b"%PDF") and "application/pdf" not in ctype:
        raise ValueError(f"URL did not return a PDF (content-type={ctype or 'unknown'})")
    path.write_bytes(content)
    return len(content), r.url


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the curated AgriRAG corpus")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sources")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    successes = 0
    failures = []
    inventory = []

    for row in rows:
        sid = row["source_id"].strip()
        title = row["title"].strip()
        mode = row["mode"].strip()
        url = row["download_url"].strip()
        suffix = ".md" if mode == "fao_html_book" else ".pdf"
        out_path = args.out / safe_name(sid, title, suffix)

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"[SKIP] {sid} {out_path.name}")
            successes += 1
            inventory.append((sid, out_path.name, out_path.stat().st_size, sha256(out_path), "existing"))
            continue

        print(f"[GET ] {sid} {title}")
        try:
            if mode == "fao_html_book":
                text, used_urls = collect_fao_book(session, url)
                if len(text) < 3000:
                    raise ValueError("HTML book extraction was unexpectedly small")
                out_path.write_text(text, encoding="utf-8")
                final_url = url
                size = out_path.stat().st_size
                note = f"{len(used_urls)} HTML parts"
            elif mode == "pdf":
                size, final_url = save_pdf(session, url, out_path)
                note = "PDF"
            else:
                raise ValueError(f"unsupported mode: {mode}")

            digest = sha256(out_path)
            inventory.append((sid, out_path.name, size, digest, f"{note}; {final_url}"))
            successes += 1
            print(f"      saved {out_path.name} ({size/1024/1024:.2f} MB)")
        except Exception as exc:
            failures.append((sid, title, str(exc)))
            print(f"[FAIL] {sid}: {exc}")

    inv_path = args.out / "corpus_inventory.csv"
    with inv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "filename", "bytes", "sha256", "note"])
        writer.writerows(inventory)

    print(f"\nSuccessful documents: {successes}/{len(rows)}")
    print(f"Inventory: {inv_path}")
    if failures:
        print("\nFailures:")
        for sid, title, error in failures:
            print(f"  {sid} {title}: {error}")

    if successes < 20:
        raise SystemExit(
            "Fewer than 20 documents were collected. Resolve failed links before indexing."
        )


if __name__ == "__main__":
    main()
