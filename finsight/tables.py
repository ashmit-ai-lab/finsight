"""Table aware chunks.

Financial tables lose their structure when flattened to text: a number ends up far from its
row label and column year. Here every table row is written out as
"<table title> | <row label>: <year> = <value>; ..." so each number carries its own context.
Tables come from the filing html (exact structure), then get mapped to pdf pages by matching numbers.
"""
import re

from bs4 import BeautifulSoup

import config

_NUM_TOKEN = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\d.,])\d+\.\d+(?![\d,])")
_PREFIXES = {"$", "(", "$(", "($", "$ ("}
_SUFFIXES = {")", "%", ")%", "%)", ") %"}
_DASHES = {"—", "–", "-", "— ", "―"}


def _clean(t: str) -> str:
    return " ".join(t.replace("\xa0", " ").replace("​", "").split())


def _is_num(cell: str) -> bool:
    c = cell.replace("$", "").replace(",", "").replace("(", "").replace(")", "").replace("%", "").strip()
    if c in _DASHES:
        return True
    try:
        float(c)
        return True
    except ValueError:
        return False


def _is_header_row(r: list) -> bool:
    """A row of years like ['2024', '2023'] looks numeric but is really the column header."""
    nums = [c for c in r if _is_num(c)]
    return bool(nums) and all(re.fullmatch(r"(19|20)\d\d", c) for c in nums)


def _logical_cells(tr) -> list:
    """SEC tables split '$', the number and ')' into separate cells. Glue them back."""
    cells, prefix = [], ""
    for td in tr.find_all(["td", "th"]):
        t = _clean(td.get_text(" "))
        if not t:
            continue
        if t in _PREFIXES:
            prefix += t
            continue
        if t in _SUFFIXES and cells:
            cells[-1] += t
            continue
        cells.append((prefix + t).replace("$ ", "$"))
        prefix = ""
    return cells


def _caption(table, max_chars: int = 160) -> str:
    """Nearest text before the table, usually its title such as CONSOLIDATED BALANCE SHEETS."""
    parts, total = [], 0
    for s in table.find_all_previous(string=True):
        if s.find_parent("table") is not None:
            if parts:
                break
            continue
        if s.find_parent(["script", "style"]) is not None or s.find_parent("ix:header") is not None:
            continue
        t = _clean(str(s))
        if not t:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars or len(parts) >= 2:
            break
    return " ".join(reversed(parts))[-max_chars:]


def parse_tables(html: str) -> list:
    """Return [{caption, header, rows:[(label, values)]}] for tables that hold numbers."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for table in soup.find_all("table"):
        rows = [r for r in (_logical_cells(tr) for tr in table.find_all("tr")) if r]
        is_numeric_row = [len(r) >= 2 and any(_is_num(c) for c in r[1:]) and not _is_header_row(r)
                          for r in rows]
        if sum(is_numeric_row) < 2:
            continue  # layout table or text box, not a data table
        first = is_numeric_row.index(True)
        head_rows = rows[:first]
        header = max(head_rows, key=len) if head_rows else []
        context = " ".join(" ".join(r) for r in head_rows if r is not header)
        body, section = [], ""
        for r, numeric in zip(rows[first:], is_numeric_row[first:]):
            if not numeric:
                if len(r) == 1 and not _is_header_row(r):
                    section = r[0].rstrip(":")
                else:
                    header = r  # a new header row in the middle of the table
                continue
            label, values = (r[0], r[1:]) if not _is_num(r[0]) else ("", r)
            is_total = label.lower().startswith("total")
            if section and label and section != label:
                label = f"{section} > {label}"
            body.append((label, values, list(header)))
            if is_total:
                section = ""  # a Total row closes its section
        if body:
            out.append({"caption": _caption(table), "context": context, "rows": body})
    return out


def _serialize_row(label: str, values: list, header: list) -> str:
    if header and len(header) >= len(values):
        heads = header[-len(values):]
    else:
        heads = [""] * len(values)
    pairs = "; ".join(f"{h} = {v}" if h else v for h, v in zip(heads, values))
    return f"{label}: {pairs}" if label else pairs


def numbers_in(text: str) -> set:
    return set(_NUM_TOKEN.findall(text))


def _best_page(nums: set, page_nums: list, last_page: int) -> int:
    if not nums:
        return last_page
    scores = [len(nums & pn) for pn in page_nums]
    best = max(range(len(scores)), key=lambda i: (scores[i], -abs(i - last_page)))
    return best if scores[best] > 0 else last_page


def table_chunks(ticker: str, html: str, page_texts: list) -> list:
    page_nums = [numbers_in(t) for t in page_texts]
    chunks, last = [], 0
    for ti, tbl in enumerate(parse_tables(html)):
        title = " | ".join(x for x in (tbl["caption"], tbl["context"]) if x)
        rows = tbl["rows"]
        for start in range(0, len(rows), config.TABLE_ROWS_PER_CHUNK):
            part = rows[start:start + config.TABLE_ROWS_PER_CHUNK]
            lines = [_serialize_row(*r) for r in part]
            text = f"Table: {title}\n" + "\n".join(lines)
            page_idx = _best_page(numbers_in("\n".join(lines)), page_nums, last)
            last = page_idx
            chunks.append({"ticker": ticker, "page": page_idx + 1, "kind": "table",
                           "text": text, "table": ti})
    return chunks
