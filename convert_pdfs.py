"""Convert National Archives of Singapore speech PDFs into .txt files for ingest.py.

Run:  python convert_pdfs.py                 (reads data/pdfs/, writes data/docs/)
      python convert_pdfs.py --overwrite     (replace .txt files that already exist)

For each PDF it:
  - removes page numbers and the "lky/1967/lky0619.doc" style footer
  - rejoins lines into paragraphs, including paragraphs split across pages
  - turns the all-caps heading into a readable title
  - reads the date from the filename (lky19670619.pdf, LKY_24-11-1966.pdf),
    or from the heading if the filename has no date
  - takes the source URL from data/pdfs/sources.csv if you provide one

Optional data/pdfs/sources.csv (filename,url):
  lky19670619.pdf,https://www.nas.gov.sg/archivesonline/...
"""
import argparse
import csv
import re
import statistics
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_DIR = ROOT / "data" / "docs"

SMALL_WORDS = {"a", "an", "the", "of", "at", "on", "in", "by", "to", "for", "and", "or", "with", "from"}


def is_noise(text: str) -> bool:
    """Page numbers and the archive's file-path footer."""
    t = text.strip()
    return not t or t.isdigit() or bool(re.match(r"^lky/", t, re.IGNORECASE))


def extract_lines(path: Path) -> list[dict]:
    """Return text lines with their position, page by page, without noise."""
    lines = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages):
            page_lines = [l for l in page.extract_text_lines() if not is_noise(l["text"])]
            if not page_lines:
                continue
            margin = min(l["x0"] for l in page_lines)
            for i, l in enumerate(page_lines):
                lines.append({
                    "text": l["text"].strip(),
                    "indented": l["x0"] > margin + 15,
                    "gap": l["top"] - page_lines[i - 1]["top"] if i > 0 else None,
                    "page_start": i == 0 and page_no > 0,
                })
    return lines


def join_line(paragraph: str, line: str) -> str:
    # Keep word hyphenation like "post-" + "graduate" together, but not "--" dashes.
    if paragraph.endswith("-") and not paragraph.endswith("--"):
        return paragraph + line
    return f"{paragraph} {line}" if paragraph else line


def build_paragraphs(lines: list[dict]) -> list[str]:
    gaps = [l["gap"] for l in lines if l["gap"]]
    line_spacing = statistics.median(gaps) if gaps else 0
    paragraphs, current = [], ""
    for l in lines:
        if l["page_start"]:
            new_para = l["indented"]  # a new page continues the paragraph unless it starts indented
        else:
            new_para = l["indented"] or (l["gap"] is not None and l["gap"] > line_spacing * 1.5)
        if new_para and current:
            paragraphs.append(current)
            current = ""
        current = join_line(current, l["text"])
    if current:
        paragraphs.append(current)
    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs]


def smart_title(text: str) -> str:
    out = []
    for i, w in enumerate(text.lower().split()):
        bare = w.strip('"“”\'(')
        if i > 0 and bare in SMALL_WORDS and w == bare:
            out.append(w)
        else:  # capitalize the first letter, even right after an opening quote
            out.append(re.sub(r"[a-z]", lambda m: m.group().upper(), w, count=1))
    return " ".join(out)


MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
HEADING_DATE = r"(\d{1,2})(?:ST|ND|RD|TH)?\s+(" + "|".join(MONTHS) + r"),?\s+(\d{4})"


def make_title(heading: str, fallback: str) -> str:
    """'TRANSCRIPT OF SPEECH BY MR. LEE KUAN YEW ON "X" AT Y HELD ON 24TH NOVEMBER, 1966.'
    -> 'Speech on "X" at Y'"""
    t = heading.strip().rstrip(".")
    t = re.sub(r"^TRANSCRIPT OF\s+", "", t, flags=re.IGNORECASE)
    # remove the speaker: "BY THE PRIME MINISTER, MR. LEE KUAN YEW," or "BY MR. LEE KUAN YEW"
    t = re.sub(r"\s+BY\s+[^\"“]*?LEE KUAN YEW,?", "", t, flags=re.IGNORECASE)
    # remove the trailing date: "HELD ON 24TH NOVEMBER, 1966" or "ON 19TH JUNE, 1967"
    t = re.sub(r",?\s+(?:HELD\s+)?(?:ON\s+)?" + HEADING_DATE + r"$", "", t, flags=re.IGNORECASE)
    t = re.sub(r",?\s+(?:HELD\s+)?ON$", "", t, flags=re.IGNORECASE)
    if not t or len(t) > 200:
        return fallback
    return smart_title(t)


def _valid(y, m, d) -> bool:
    return 1900 <= int(y) <= 2015 and 1 <= int(m) <= 12 and 1 <= int(d) <= 31


def date_from_filename(name: str) -> str | None:
    """Supports lky19670619, LKY_1967-06-19, LKY_24-11-1966 (day-month-year) and similar."""
    patterns = [
        (r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})", "ymd"),   # 19670619, 1967-06-19
        (r"(\d{1,2})[-_.](\d{1,2})[-_.](\d{4})", "dmy"),  # 24-11-1966, 24_11_1966
    ]
    for pattern, order in patterns:
        for m in re.finditer(pattern, name):
            a, b, c = m.groups()
            y, mo, d = (a, b, c) if order == "ymd" else (c, b, a)
            if _valid(y, mo, d):
                return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None


def date_from_heading(heading: str) -> str | None:
    """Fallback: read '24TH NOVEMBER, 1966' from the transcript heading."""
    m = re.search(HEADING_DATE, heading, flags=re.IGNORECASE)
    if not m:
        return None
    d, month, y = m.groups()
    return f"{y}-{MONTHS.index(month.lower()) + 1:02d}-{int(d):02d}"


def load_sources() -> dict:
    path = PDF_DIR / "sources.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row[0].strip(): row[1].strip() for row in csv.reader(f) if len(row) >= 2}


def convert(path: Path, sources: dict) -> tuple[str, str]:
    paragraphs = build_paragraphs(extract_lines(path))
    if not paragraphs:
        raise ValueError("no text found (scanned PDF?)")

    # The archive heading is in capitals and may be split over several centered lines.
    heading_parts = []
    while paragraphs and paragraphs[0].isupper():
        heading_parts.append(paragraphs.pop(0))
    heading = " ".join(heading_parts)
    title = make_title(heading, path.stem) if heading else path.stem

    date = date_from_filename(path.stem) or date_from_heading(heading) or "unknown"
    source = sources.get(path.name, f"National Archives of Singapore ({path.name})")
    header = f"Title: {title}\nDate: {date}\nSource: {source}\n---\n"

    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if len(slug) > 60:
        slug = slug[:60].rsplit("_", 1)[0]  # cut at a word boundary
    filename = f"{date}_{slug}.txt" if date != "unknown" else f"{path.stem}.txt"
    return filename, header + "\n\n".join(paragraphs) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="replace .txt files that already exist")
    args = parser.parse_args()

    pdfs = sorted(p for p in PDF_DIR.glob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()

    done, skipped, failed = 0, 0, 0
    for pdf in pdfs:
        try:
            filename, text = convert(pdf, sources)
        except Exception as e:
            print(f"  FAILED  {pdf.name}: {e}")
            failed += 1
            continue
        out = OUT_DIR / filename
        if out.exists() and not args.overwrite:
            print(f"  exists  {pdf.name} -> {filename} (use --overwrite to replace)")
            skipped += 1
            continue
        out.write_text(text, encoding="utf-8")
        print(f"  ok      {pdf.name} -> {filename} ({len(text.split())} words)")
        done += 1

    print(f"\nConverted {done}, skipped {skipped}, failed {failed}.")
    if done and not sources:
        print("Tip: add data/pdfs/sources.csv (filename,url) so each source links to its archive page.")
    print("Check a few titles in data/docs, then run: python ingest.py")


if __name__ == "__main__":
    main()
