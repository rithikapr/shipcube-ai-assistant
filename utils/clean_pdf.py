from collections import Counter
import re
import json
import logging
from pathlib import Path
from typing import Iterable, List, Set, Optional, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----- Default regex patterns -----

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
PHONE_RE = re.compile(r"\+?\d[\d\-\s\(\)]{6,}\d")
TOC_RE = re.compile(r"\.{2,}\s*\d+\s*$")         # "Topic ...... 12"
PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d+\s*$")
COPYRIGHT_RE = re.compile(r"(copyright|all rights reserved|printed in)", re.IGNORECASE)
CLICK_HERE_RE = re.compile(r"\b(click here|request information|click for|visit)\b", re.IGNORECASE)

SHORT_LINE_WORDS = 4
PUNCT_RATIO_THRESHOLD = 0.60
NUMERIC_RATIO_THRESHOLD = 0.60


# ----- Utility helpers -----
def normalize_whitespace_and_hyphenation(text: str) -> str:
    if not text:
        return text
    
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def line_noise_scores(line: str) -> tuple[float, float]:
    if not line:
        return 1.0, 1.0
    
    chars = len(line)
    punct = sum(1 for c in line if not c.isalnum() and not c.isspace())
    digits = sum(1 for c in line if c.isdigit())
    return punct / max(1, chars), digits / max(1, chars)


# ----- Repeated-line detection -----
def detect_repeated_lines(docs: Iterable[Any], threshold_fraction: float = 0.20) -> Set[str]:
    page_line_sets = []
    for d in docs:
        raw = d.page_content or ""
        lines = {ln.strip() for ln in raw.splitlines() if ln.strip()}
        page_line_sets.append(lines)

    counter = Counter()
    for s in page_line_sets:
        counter.update(s)

    num_pages = max(1, len(page_line_sets))
    repeated = {line for line, cnt in counter.items() if (cnt / num_pages) >= threshold_fraction}
    logger.info("detect_repeated_lines: found %d repeated lines (threshold=%.2f)",
                len(repeated), threshold_fraction)
    return repeated


# ----- Per-page cleaning function -----
def clean_page_text_full(text: str, repeated_lines: Optional[Set[str]] = None) -> str:
    if text is None:
        return ""

    if repeated_lines is None:
        repeated_lines = set()

    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        s = raw_line.strip()
        if not s:
            continue

        if s in repeated_lines:
            continue

        if TOC_RE.search(s):
            continue
        if PAGE_NUM_ONLY_RE.match(s):
            continue
        if COPYRIGHT_RE.search(s):
            continue
        if CLICK_HERE_RE.search(s):
            continue

        if URL_RE.search(s) or EMAIL_RE.search(s) or PHONE_RE.search(s):
            continue

        words = s.split()
        if len(words) <= SHORT_LINE_WORDS:
            if s.isupper() and len(words) <= 3:
                kept_lines.append(s)
            else:
                continue

        punct_ratio, num_ratio = line_noise_scores(s)
        if punct_ratio > PUNCT_RATIO_THRESHOLD or num_ratio > NUMERIC_RATIO_THRESHOLD:
            continue

        kept_lines.append(s)

    cleaned = " ".join(kept_lines)
    cleaned = normalize_whitespace_and_hyphenation(cleaned)
    return cleaned

def clean_docs(docs: List[Any],
               repeated_lines: Optional[Set[str]] = None,
               repeated_threshold: float = 0.20,
               keep_raw: bool = True) -> List[Any]:
    if not docs:
        return docs

    for d in docs:
        if keep_raw:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("raw_page_content", d.page_content)

    if repeated_lines is None:
        repeated_lines = detect_repeated_lines(docs, threshold_fraction=repeated_threshold)

    removed_example_counter = 0
    for d in docs:
        original = d.page_content or ""
        cleaned = clean_page_text_full(original, repeated_lines=repeated_lines)
        d.page_content = cleaned
        if removed_example_counter < 20 and original.strip() and original.strip() != cleaned.strip():
            logger.debug("clean_docs: sample removed (page metadata=%s)\nORIG: %.120s\nCLEAN: %.120s\n",
                         d.metadata.get("source_file") if isinstance(d.metadata, dict) else "unknown",
                         original.replace("\n", " ")[:120],
                         cleaned[:120])
            removed_example_counter += 1

    logger.info("clean_docs: cleaned %d pages (kept raw in metadata=%s)",
                len(docs), keep_raw)
    return docs


def save_cleaned_texts(docs: List[Any], out_dir: Path, prefix: str = "page"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(docs):
        source_file = (d.metadata.get("source_file") if isinstance(d.metadata, dict) else None) or "unknown_pdf"
        page_index = d.metadata.get("page") if isinstance(d.metadata, dict) and "page" in d.metadata else i
        fname = f"{prefix}__{source_file}__p{page_index}.txt"
        safe_fname = re.sub(r"[^\w\-_\. ]+", "_", fname)
        p = out_dir / safe_fname
        with p.open("w", encoding="utf-8") as fh:
            fh.write(d.page_content or "")
    logger.info("save_cleaned_texts: exported %d cleaned text files to %s", len(docs), out_dir)
