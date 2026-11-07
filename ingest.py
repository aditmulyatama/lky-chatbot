"""Build the search index from the .txt files in data/docs.

Run:  python ingest.py
Output: data/index/chunks.json and data/index/embeddings.npy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
import rag  # noqa: E402  (import after loading the API key)

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHUNK_WORDS = 500     # roughly one or two paragraphs of a speech
OVERLAP_WORDS = 50    # overlap so an idea split across two chunks isn't lost
TOKENS_PER_MINUTE = 40_000   # your OpenAI rate limit for the embedding model
MAX_BATCH_TOKENS = 20_000    # tokens per request, well under the limit


def parse_doc(path: Path) -> tuple[dict, str]:
    """Read the Title/Date/Source header (above a '---' line) and the body text."""
    raw = path.read_text(encoding="utf-8")
    meta = {"title": path.stem, "date": "unknown", "source": ""}
    if "\n---" in raw:
        header, body = raw.split("\n---", 1)
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in meta:
                    meta[key] = value.strip()
    else:
        body = raw
    return meta, body.strip()


def chunk_words(text: str) -> list[str]:
    """Split text into overlapping chunks of about CHUNK_WORDS words."""
    words = text.split()
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def estimate_tokens(text: str) -> int:
    """Rough estimate: English text is about 1.4 tokens per word. Overestimating is the safe side."""
    return int(len(text.split()) * 1.4) + 10


def make_batches(texts: list[str]) -> list[tuple[list[str], int]]:
    """Group texts into batches that each stay under MAX_BATCH_TOKENS."""
    batches, current, current_tokens = [], [], 0
    for text in texts:
        tokens = estimate_tokens(text)
        if current and current_tokens + tokens > MAX_BATCH_TOKENS:
            batches.append((current, current_tokens))
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append((current, current_tokens))
    return batches


def embed_with_retry(batch: list[str], attempts: int = 5):
    """If we still hit the rate limit, wait and try again instead of crashing."""
    for attempt in range(1, attempts + 1):
        try:
            return rag.embed(batch)
        except openai.RateLimitError:
            if attempt == attempts:
                raise
            wait = 20 * attempt
            print(f"  rate limited, retrying in {wait}s (attempt {attempt}/{attempts})...")
            time.sleep(wait)


def main():
    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {DOCS_DIR}. Add documents first (see data/docs/README.md).")

    chunks = []
    for path in files:
        meta, body = parse_doc(path)
        pieces = chunk_words(body)
        for i, piece in enumerate(pieces):
            chunks.append({**meta, "chunk_id": f"{path.stem}#{i}", "text": piece})
        print(f"  {path.name}: {len(pieces)} chunks")

    print(f"Embedding {len(chunks)} chunks from {len(files)} documents...")
    vectors = []
    batches = make_batches([c["text"] for c in chunks])
    window_start, window_tokens = time.time(), 0
    for i, (batch, tokens) in enumerate(batches, 1):
        # Stay under the tokens-per-minute limit: wait for the next minute if needed
        if window_tokens + tokens > TOKENS_PER_MINUTE * 0.9:
            wait = 60 - (time.time() - window_start)
            if wait > 0:
                print(f"  waiting {wait:.0f}s to stay under the rate limit...")
                time.sleep(wait)
            window_start, window_tokens = time.time(), 0
        vectors.append(embed_with_retry(batch))
        window_tokens += tokens
        print(f"  batch {i}/{len(batches)} done ({len(batch)} chunks, ~{tokens} tokens)")
    embeddings = np.vstack(vectors)

    rag.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(rag.INDEX_DIR / "embeddings.npy", embeddings)
    with open(rag.INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Done. Saved index to {rag.INDEX_DIR}")


if __name__ == "__main__":
    main()