"""Plain chunks: fixed size word windows over each page's text (the usual RAG baseline)."""
import config


def plain_chunks(ticker: str, page_texts: list,
                 size: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list:
    chunks = []
    step = max(1, size - overlap)
    for p, text in enumerate(page_texts):
        words = text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            chunks.append({"ticker": ticker, "page": p + 1, "kind": "text", "text": " ".join(piece)})
            if start + size >= len(words):
                break
    return chunks


def build_chunks(ticker: str, scheme: str, page_texts: list, html: str = "") -> list:
    """scheme 'plain' = page text windows. 'table' = the same windows plus one chunk per group of table rows."""
    chunks = plain_chunks(ticker, page_texts)
    if scheme == "table":
        from .tables import table_chunks
        chunks += table_chunks(ticker, html, page_texts)
    elif scheme != "plain":
        raise ValueError(scheme)
    for i, c in enumerate(chunks):
        c["id"] = f"{ticker}:{scheme}:{i}"
    return chunks
