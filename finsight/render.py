"""Turn the filing html into a pdf so all three methods search the same numbered pages."""
import json
from pathlib import Path

import config


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=300_000)
        page.pdf(path=str(pdf_path), format="Letter",
                 margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"})
        browser.close()


def extract_page_texts(pdf_path: Path) -> list:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [p.get_text("text") for p in doc]


def render(ticker: str) -> None:
    html = config.RAW / ticker / "filing.htm"
    pdf = config.PDF / f"{ticker}.pdf"
    if not pdf.exists():
        html_to_pdf(html, pdf)
    texts = extract_page_texts(pdf)
    (config.PAGES / f"{ticker}.json").write_text(json.dumps(texts))


def load_pages(ticker: str) -> list:
    return json.loads((config.PAGES / f"{ticker}.json").read_text())


def page_image(ticker: str, page_no: int, dpi: int = config.VISUAL_DPI):
    """page_no is 1 based. Returns a PIL image."""
    import pymupdf
    from PIL import Image

    with pymupdf.open(config.PDF / f"{ticker}.pdf") as doc:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def page_count(ticker: str) -> int:
    return len(load_pages(ticker))
