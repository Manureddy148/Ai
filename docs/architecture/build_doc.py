#!/usr/bin/env python3
"""Assemble docs/architecture/sections/*.html into one styled HTML document and render it
to docs/Architecture.pdf with the bundled headless Chromium.

Two-pass build: pass 1 renders once to discover the page on which every chapter and
section heading lands (via PyMuPDF text search); pass 2 injects those page numbers into
the table of contents and renders the final PDF.

Usage:  python3 docs/architecture/build_doc.py [--chromium /path/to/chrome] [--no-pdf]
"""
import argparse
import glob
import html
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECTIONS = os.path.join(HERE, "sections")
HTML_OUT = os.path.join(HERE, "Architecture.html")
PDF_OUT = os.path.join(os.path.dirname(HERE), "Architecture.pdf")
STYLE = os.path.join(HERE, "style.css")

TITLE = "Business Entity Resolution System"
SUBTITLE = "Architecture and Design Document"
META = [("Project", "Amazon ML Challenge — Business Entity Resolution"),
        ("Repository", "Manureddy148/Ai"),
        ("Component", "code/business_entity_resolution/src/er_pipeline.py"),
        ("Document version", "1.0"),
        ("Status", "Draft for review")]

CHROMIUM_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    shutil.which("chromium"), shutil.which("chromium-browser"), shutil.which("google-chrome"),
]


def find_chromium(explicit=None):
    for c in [explicit] + CHROMIUM_CANDIDATES:
        if c and os.path.exists(c):
            return c
    for c in glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"):
        return c
    sys.exit("headless Chromium not found; pass --chromium")


def load_sections():
    files = sorted(glob.glob(os.path.join(SECTIONS, "*.html")))
    if not files:
        sys.exit(f"no sections in {SECTIONS}")
    return [(f, open(f, encoding="utf-8").read()) for f in files]


_h = re.compile(r"<h([12])[^>]*>(.*?)</h\1>", re.S)
_tag = re.compile(r"<[^>]+>")


def headings(sections):
    """[(level, chapter_number, text, anchor)] in document order."""
    out = []
    for f, body in sections:
        m = re.search(r'<section[^>]*id="([^"]+)"', body)
        anchor = m.group(1) if m else os.path.basename(f)
        for level, inner in _h.findall(body):
            text = html.unescape(_tag.sub("", inner)).strip()
            text = re.sub(r"\s+", " ", text)
            out.append((int(level), text, anchor))
    return out


def toc_html(entries, pages):
    rows = []
    for level, text, anchor in entries:
        pg = pages.get(text, "")
        cls = "toc-h1" if level == 1 else "toc-h2"
        rows.append(f'<li class="{cls}"><span class="toc-text">{html.escape(text)}</span>'
                    f'<span class="toc-dots"></span><span class="toc-page">{pg}</span></li>')
    return '<nav class="toc"><h1 class="toc-title">Contents</h1><ul>' + "".join(rows) + "</ul></nav>"


def assemble(sections, pages):
    css = open(STYLE, encoding="utf-8").read()
    meta_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in META)
    cover = f"""
<section class="cover">
  <p class="cover-kicker">Technical documentation</p>
  <h1 class="cover-title">{html.escape(TITLE)}</h1>
  <p class="cover-sub">{html.escape(SUBTITLE)}</p>
  <table class="cover-meta">{meta_rows}</table>
  <p class="cover-abstract">This document describes the architecture of an end-to-end entity-resolution
  pipeline that links noisy business records from three independent sources without any external data:
  text normalisation, memory-bounded candidate generation, pairwise feature engineering, a gradient-boosted
  matching model, and a precision-oriented decision layer tuned for the challenge's macro-F<sub>0.5</sub>
  metric. It is written for engineers who need to run, audit, extend or re-implement the system.</p>
</section>"""
    body = "\n".join(b for _, b in sections)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(TITLE)} — {html.escape(SUBTITLE)}</title>
<style>{css}</style></head>
<body>
{cover}
{toc_html(headings(sections), pages)}
<main class="doc">
{body}
</main>
</body></html>"""


def render(chromium, html_path, pdf_path):
    cmd = [chromium, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", "--virtual-time-budget=10000", f"file://{html_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.exists(pdf_path):
        sys.exit(f"chromium failed:\n{r.stderr[-2000:]}")


def page_index(pdf_path, entries):
    """Map heading text -> 1-based page number where it first appears after the TOC."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages, start = {}, 0
    # skip the cover and the TOC pages (the TOC repeats every heading): TOC pages carry
    # "Contents" in the running header, so start after the last page that has it
    for pno in range(len(doc)):
        head = doc[pno].get_text("text", clip=pymupdf.Rect(0, 0, doc[pno].rect.width, 60))
        if "Contents" in head:
            start = pno + 1
    for level, text, _ in entries:
        found = None
        for pno in range(start, len(doc)):
            if doc[pno].search_for(text):
                found = pno
                break
        if found is not None:
            pages[text] = found + 1
            start = found  # headings are in document order; never look backwards
    return pages, len(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium")
    ap.add_argument("--no-pdf", action="store_true", help="only write Architecture.html")
    args = ap.parse_args()

    sections = load_sections()
    entries = headings(sections)
    open(HTML_OUT, "w", encoding="utf-8").write(assemble(sections, {}))
    print(f"wrote {HTML_OUT} ({len(sections)} sections, {len(entries)} headings)")
    if args.no_pdf:
        return
    chromium = find_chromium(args.chromium)
    render(chromium, HTML_OUT, PDF_OUT)                       # pass 1: discover page numbers
    pages, n = page_index(PDF_OUT, entries)
    # the TOC is rendered before the chapters; pass-1 numbers already include cover + TOC pages
    open(HTML_OUT, "w", encoding="utf-8").write(assemble(sections, pages))
    render(chromium, HTML_OUT, PDF_OUT)                       # pass 2: final
    pages2, n2 = page_index(PDF_OUT, entries)
    drift = sum(1 for k in pages if pages2.get(k) != pages[k])
    print(f"wrote {PDF_OUT}: {n2} pages (pass 1: {n}); TOC entries with drift: {drift}")
    for level, text, _ in entries:
        if level == 1:
            print(f"  p.{pages2.get(text, '?'):>3}  {text}")


if __name__ == "__main__":
    main()
