"""
Problem 1 - Task 0: Data Collection (Crawler)
==============================================
Collect English-only text from IIT Jodhpur pages and related documents.

Covered categories:
  - departments
  - academic_programs
  - research
  - announcements
  - academic_regulations
  - newsletters_circulars
  - faculty_profiles
  - course_syllabus

Outputs:
  1) corpus_raw.txt (plain text, one cleaned block per source)
  2) results/source_manifest.json (URL + metadata for traceability)

Usage:
  python task0_data_collection.py
  python task0_data_collection.py --max-pages 120 --max-depth 2

Optional dependency for PDF extraction:
  pip install pypdf
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# Suppress noisy third-party runtime warnings during crawl runs.
warnings.filterwarnings("ignore")


IITJ_DOMAIN = "iitj.ac.in"


# Seed pages for each required category.
SEED_URLS: Dict[str, List[str]] = {
    "departments": [
        "https://iitj.ac.in/departments/",
    ],
    "academic_programs": [
        "https://iitj.ac.in/academics/",
        "https://iitj.ac.in/admissions/",
    ],
    "research": [
        "https://iitj.ac.in/research/",
    ],
    "announcements": [
        "https://iitj.ac.in/announcements/",
        "https://iitj.ac.in/news/",
    ],
    "academic_regulations": [
        "https://iitj.ac.in/office-of-academics/en/academic-regulations",
        "https://iitj.ac.in/office-of-academics/en/academics",
        "https://iitj.ac.in/office-of-academics/en/download-forms",
    ],
    "newsletters_circulars": [
        "https://iitj.ac.in/newsletter/",
        "https://iitj.ac.in/circulars/",
    ],
    "faculty_profiles": [
        "https://iitj.ac.in/faculty/",
    ],
    "course_syllabus": [
        "https://iitj.ac.in/academics/curriculum/",
        "https://iitj.ac.in/academics/syllabus/",
    ],
}


CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "departments": ("department", "school"),
    "academic_regulations": (
        "academic-regulations", "regulation", "ordinance", "rule", "academic-rules", "policy", "office-of-academics"
    ),
    "academic_programs": ("academics", "programme", "program", "admission", "btech", "mtech", "phd"),
    "research": ("research", "project", "center", "centre", "laboratory", "publication"),
    "announcements": ("announcement", "notice", "news", "event"),
    "newsletters_circulars": ("newsletter", "circular", "bulletin"),
    "faculty_profiles": ("faculty", "people", "profile", "staff"),
    "course_syllabus": ("syllabus", "curriculum", "course"),
}


PDF_EXTENSIONS = (".pdf",)


@dataclass
class CrawlRecord:
    category: str
    url: str
    content_type: str
    fetched_at_utc: str
    char_count: int
    token_count_approx: int


def build_seed_frontier() -> deque:
    q = deque()
    for category, urls in SEED_URLS.items():
        for u in urls:
            q.append((u, category, 0))
    return q


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def is_allowed_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    if not p.netloc:
        return False
    return IITJ_DOMAIN in p.netloc.lower()


def infer_category(url: str, fallback: str) -> str:
    lower = url.lower()
    for category, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                return category

    # Avoid labeling generic utility pages as regulations just because
    # they were discovered from a regulation seed page.
    if fallback == "academic_regulations":
        if "academics" in lower or "program" in lower or "admission" in lower:
            return "academic_programs"
        return "departments"

    return fallback


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def english_line_filter(text: str, min_ascii_ratio: float = 0.9, min_alpha: int = 12) -> str:
    """
    Keep only lines that are likely English (ASCII-dominant and alphabetic).
    This is conservative and removes most non-English script content.
    """
    kept: List[str] = []
    for raw_line in re.split(r"[\n\r]+", text):
        line = raw_line.strip()
        if not line:
            continue
        ascii_count = sum(1 for ch in line if ord(ch) < 128)
        ratio = ascii_count / max(1, len(line))
        alpha = sum(1 for ch in line if ch.isalpha() and ord(ch) < 128)
        if ratio >= min_ascii_ratio and alpha >= min_alpha:
            kept.append(line)
    return "\n".join(kept)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form", "nav", "footer"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("section")
        or soup.find("div", attrs={"id": re.compile(r"content|main", re.I)})
        or soup.body
        or soup
    )

    text = main.get_text("\n", strip=True)
    text = clean_text(text)
    return english_line_filter(text)


def extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        import io

        reader = PdfReader(io.BytesIO(content))
        parts: List[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        text = "\n".join(parts)
        text = clean_text(text)
        return english_line_filter(text)
    except Exception:
        return ""


def discover_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = normalize_url(urljoin(base_url, href))
        if is_allowed_url(abs_url):
            links.append(abs_url)
    return links


def fetch_url(session: requests.Session, url: str, timeout: int = 20) -> Tuple[Optional[str], Optional[bytes], str]:
    """
    Returns (html_text, binary_content, content_type).
    One of html_text or binary_content will be set.
    """
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code >= 400:
            return None, None, ""
        ctype = (resp.headers.get("Content-Type") or "").lower()

        if "text/html" in ctype or url.lower().endswith((".html", "/")):
            resp.encoding = resp.encoding or "utf-8"
            return resp.text, None, "html"

        if "application/pdf" in ctype or url.lower().endswith(PDF_EXTENSIONS):
            return None, resp.content, "pdf"

        # Unsupported content type.
        return None, None, ""
    except Exception:
        return None, None, ""


def crawl_iitj(
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
) -> Tuple[List[str], List[CrawlRecord], Dict[str, int]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) IITJ-NLP-Crawler/1.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    frontier = build_seed_frontier()
    visited: Set[str] = set()
    seen_text_hashes: Set[str] = set()

    corpus_blocks: List[str] = []
    manifest: List[CrawlRecord] = []
    per_category_count = collections.Counter()

    while frontier and len(visited) < max_pages:
        url, category, depth = frontier.popleft()
        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        html_text, binary_data, ctype = fetch_url(session, url)
        if not ctype:
            continue

        extracted = ""
        if ctype == "html" and html_text is not None:
            extracted = html_to_text(html_text)
        elif ctype == "pdf" and binary_data is not None:
            extracted = extract_pdf_text(binary_data)

        if extracted:
            digest = hashlib.sha1(extracted.encode("utf-8", errors="ignore")).hexdigest()
            if digest not in seen_text_hashes:
                seen_text_hashes.add(digest)
                inferred = infer_category(url, category)
                corpus_blocks.append(extracted)
                manifest.append(
                    CrawlRecord(
                        category=inferred,
                        url=url,
                        content_type=ctype,
                        fetched_at_utc=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        char_count=len(extracted),
                        token_count_approx=len(extracted.split()),
                    )
                )
                per_category_count[inferred] += 1

        if ctype == "html" and html_text is not None and depth < max_depth:
            for nxt in discover_links(url, html_text):
                if nxt not in visited:
                    frontier.append((nxt, infer_category(nxt, category), depth + 1))

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return corpus_blocks, manifest, dict(per_category_count)


def write_outputs(corpus_blocks: List[str], manifest: List[CrawlRecord], out_corpus: str, out_manifest: str) -> None:
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)

    # One source block separated by blank lines.
    with open(out_corpus, "w", encoding="utf-8") as f:
        f.write("\n\n".join(corpus_blocks).strip() + "\n")

    json_payload = [record.__dict__ for record in manifest]
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl IIT Jodhpur sources and build corpus_raw.txt")
    parser.add_argument("--max-pages", type=int, default=120, help="Maximum URLs to visit")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum link depth from seeds")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between requests (seconds)")
    parser.add_argument("--out-corpus", default="corpus_raw.txt", help="Output corpus file path")
    parser.add_argument(
        "--out-manifest",
        default="results/source_manifest.json",
        help="Output source manifest JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print(" Task 0: IIT Jodhpur Data Collection (Crawler)")
    print("=" * 70)
    print(f"[INFO] max_pages={args.max_pages}, max_depth={args.max_depth}, delay={args.delay}")

    blocks, manifest, per_category = crawl_iitj(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
    )

    write_outputs(blocks, manifest, args.out_corpus, args.out_manifest)

    print(f"[INFO] Collected source blocks: {len(blocks)}")
    print(f"[INFO] Manifest entries      : {len(manifest)}")
    print("[INFO] Coverage by category  :")
    for cat in sorted(SEED_URLS.keys()):
        print(f"  - {cat:<22} {per_category.get(cat, 0)}")

    print(f"[INFO] corpus output   : {args.out_corpus}")
    print(f"[INFO] manifest output : {args.out_manifest}")

    # Requirement reminder without warning-level noise.
    if per_category.get("academic_regulations", 0) == 0:
        print("[INFO] No academic_regulations page captured. Update seed URLs and rerun.")

    print("[DONE] Data collection finished.")


if __name__ == "__main__":
    main()
