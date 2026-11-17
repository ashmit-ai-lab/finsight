"""Shared search + answer logic used by both the evaluation script and the app."""
import config

from . import llm, retrieval
from .render import load_pages

# name -> (chunking scheme, search mode). Each row changes one thing from the row above it.
METHODS = {
    "plain_dense": ("plain", "dense"),    # usual RAG baseline
    "plain_hybrid": ("plain", "hybrid"),  # + keyword search
    "table_dense": ("table", "dense"),    # + table aware chunks
    "table_hybrid": ("table", "hybrid"),  # both
    "visual": ("visual", None),           # page screenshots, no text extraction
}


class Searcher:
    def __init__(self, embedder=None, visual_retriever=None):
        self._embedder = embedder
        self._visual = visual_retriever
        self._text, self._vis, self._pages = {}, {}, {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = retrieval.Embedder()
        return self._embedder

    @property
    def visual(self):
        if self._visual is None:
            from .visual import VisualRetriever
            self._visual = VisualRetriever()
        return self._visual

    def pages(self, ticker):
        if ticker not in self._pages:
            self._pages[ticker] = load_pages(ticker)
        return self._pages[ticker]

    def search(self, query: str, ticker: str, method: str, k: int = 20) -> list:
        scheme, mode = METHODS[method]
        if scheme == "visual":
            from . import visual
            if ticker not in self._vis:
                self._vis[ticker] = visual.load(ticker)
            return visual.search_as_hits(query, ticker, self.visual, self._vis[ticker], self.pages(ticker), k)
        key = (scheme, ticker)
        if key not in self._text:
            self._text[key] = retrieval.TextIndex.load(scheme, ticker)
        return self._text[key].search(query, self.embedder, k=k, mode=mode)

    def ask(self, question: str, ticker: str, method: str, model: str = None) -> dict:
        hits = self.search(question, ticker, method)
        ctx = retrieval.fill_context(hits)
        out = llm.answer(question, ctx, model=model)
        out["retrieved_pages"] = retrieval.ranked_pages(hits)
        out["context"] = ctx
        return out


    def second_look(self, question: str, ticker: str, method: str, model: str = None) -> dict:
        """Ask again with twice the passages. Only used when the first answer said it found nothing."""
        hits = self.search(question, ticker, method, k=40)
        ctx = retrieval.fill_context(hits, budget=2 * config.CONTEXT_CHARS)
        out = llm.answer(question, ctx, model=model)
        out["retrieved_pages"] = retrieval.ranked_pages(hits)
        out["context"] = ctx
        return out


def extract_key_numbers(searcher: Searcher, ticker: str, company: str, method: str,
                        facts: dict, accession: str, model: str = None) -> list:
    """Ask for each headline number of the latest year, then check it against the official XBRL value."""
    from .questions import facts_in_filing, _nice_date
    from .scoring import number_correct
    rows, latest = [], {}
    for f in facts_in_filing(facts, accession):
        if f["metric"] not in latest or f["end"] > latest[f["metric"]]["end"]:
            latest[f["metric"]] = f
    for f in latest.values():
        when = "as of" if f["kind"] == "instant" else "for the fiscal year ended"
        q = f"What was {company}'s {f['metric']} {when} {_nice_date(f['end'])}?"
        from .graph import ask as graph_ask
        out = graph_ask(searcher, q, ticker, method, model=model, require_value=True)
        rows.append({
            "metric": f["metric"], "period_end": f["end"],
            "extracted": out.get("value"), "scale": out.get("scale"),
            "official_xbrl": f["value"], "match": number_correct(out.get("value"), out.get("scale"), f["value"], f["unit"]),
            "pages": out.get("pages"),
        })
    return rows
