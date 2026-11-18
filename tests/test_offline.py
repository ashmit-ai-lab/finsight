"""Offline smoke test: fake filing, fake embedder, fake LLM. No network or model downloads.
Run:  python -m pytest tests -q
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ACC = "0000000000-24-000001"

HTML = """<html><body>
<p>PART I Item 1. Business. Example Corp designs widgets and sells them worldwide.</p>
<p style="page-break-after:always">Our widget business grew because of strong demand in Asia.</p>
<p>Item 8. Financial Statements</p>
<p>EXAMPLE CORP CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<table>
<tr><td></td><td colspan=2>Years ended</td></tr>
<tr><td></td><td>September 28, 2024</td><td></td><td>September 30, 2023</td></tr>
<tr><td colspan=3>(In millions, except per share amounts)</td></tr>
<tr><td>Net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
<tr><td>Operating expenses:</td></tr>
<tr><td>Research and development</td><td>31,370</td><td>29,915</td></tr>
<tr><td>Total operating expenses</td><td>57,467</td><td>54,847</td></tr>
<tr><td>Operating income</td><td>123,216</td><td>114,301</td></tr>
<tr><td>Net income</td><td>$</td><td>93,736</td><td>$</td><td>96,995</td></tr>
<tr><td>Diluted</td><td>$</td><td>6.08</td><td>$</td><td>6.13</td></tr>
</table>
<p style="page-break-before:always">EXAMPLE CORP CONSOLIDATED BALANCE SHEETS</p>
<table>
<tr><td></td><td>September 28, 2024</td><td>September 30, 2023</td></tr>
<tr><td>Total assets</td><td>$</td><td>364,980</td><td>$</td><td>352,583</td></tr>
<tr><td>Total liabilities</td><td>308,030</td><td>290,437</td></tr>
</table>
</body></html>"""


def _fact(val, end, start=None):
    e = {"val": val, "end": end, "accn": ACC, "form": "10-K", "fy": 2024, "fp": "FY"}
    if start:
        e["start"] = start
    return e


FACTS = {"facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        _fact(391035e6, "2024-09-28", "2023-10-01"), _fact(383285e6, "2023-09-30", "2022-09-25")]}},
    "NetIncomeLoss": {"units": {"USD": [_fact(93736e6, "2024-09-28", "2023-10-01")]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [_fact(6.08, "2024-09-28", "2023-10-01")]}},
    "Assets": {"units": {"USD": [_fact(364980e6, "2024-09-28"), _fact(352583e6, "2023-09-30")]}},
}}}


class HashEmbedder:
    def __init__(self, *a, **k):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.v = HashingVectorizer(n_features=2048, ngram_range=(1, 2), norm="l2")

    def docs(self, texts):
        return self.v.transform(texts).toarray().astype(np.float32)

    def query(self, q):
        return self.v.transform([q]).toarray()[0].astype(np.float32)


def fake_answer(question, hits, model=None):
    """Pretend LLM: return the first big number in the top excerpt that mentions a question word."""
    for h in hits:
        m = re.search(r"\d{1,3}(?:,\d{3})+|\d+\.\d{2}", h["text"])
        if m:
            return {"answer": m.group(), "value": m.group(), "scale": "millions", "pages": [h["page"]], "found": True}
    return {"answer": "", "value": None, "scale": None, "pages": [], "found": False}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for name in ("DATA", "RAW", "PDF", "PAGES", "INDEX", "RESULTS"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(config, name, d)
    monkeypatch.setattr(config, "COMPANIES", {"EXM": "Example Corp"})
    raw = config.RAW / "EXM"
    raw.mkdir()
    (raw / "filing.htm").write_text(HTML)
    (raw / "facts.json").write_text(json.dumps(FACTS))
    (raw / "meta.json").write_text(json.dumps({"accession": ACC, "report_date": "2024-09-28", "url": "x"}))
    from finsight import llm, retrieval
    monkeypatch.setattr(retrieval, "Embedder", HashEmbedder)
    monkeypatch.setattr(llm, "answer", fake_answer)
    from langchain_core.runnables import RunnableLambda
    from finsight import graph
    monkeypatch.setattr(graph, "answer_chain", lambda model: RunnableLambda(
        lambda inp: fake_answer(inp["question"], [{"text": inp["context"], "page": 1}])))
    return tmp_path


def test_table_parsing_glues_cells():
    from finsight.tables import parse_tables
    tables = parse_tables(HTML)
    assert len(tables) == 2
    label, values, header = tables[0]["rows"][0]
    assert label == "Net sales" and values == ["$391,035", "$383,285"]
    assert header[-2:] == ["September 28, 2024", "September 30, 2023"]
    labels = [r[0] for r in tables[0]["rows"]]
    assert "Operating expenses > Research and development" in labels
    assert "Operating income" in labels  # section closed by the Total row
    assert "STATEMENTS OF OPERATIONS" in tables[0]["caption"]


def test_year_header_row_is_not_data():
    from finsight.tables import parse_tables
    html = """<table><tr><td></td><td>2024</td><td></td><td>2023</td></tr>
    <tr><td>Revenue</td><td>$</td><td>1,500</td><td>$</td><td>1,200</td></tr>
    <tr><td>Cost of sales</td><td>(700</td><td>)</td><td>(600</td><td>)</td></tr></table>"""
    rows = parse_tables(html)[0]["rows"]
    assert len(rows) == 2
    assert rows[0] == ("Revenue", ["$1,500", "$1,200"], ["2024", "2023"])
    assert rows[1][1] == ["(700)", "(600)"]


def test_scoring():
    from finsight.scoring import number_correct, parse_number
    assert parse_number("(1,234)") == -1234
    assert number_correct("391,035", "millions", 391035e6, "USD")
    assert not number_correct("391,035", "thousands", 391035e6, "USD")
    assert number_correct("$6.08", "millions", 6.08, "USD/shares")


def test_full_pipeline(env):
    import run
    from finsight import chunking, retrieval
    from finsight.pipeline import Searcher, extract_key_numbers
    from finsight.render import load_pages, render

    render("EXM")
    pages = load_pages("EXM")
    assert len(pages) >= 2

    for scheme in ("plain", "table"):
        chunks = chunking.build_chunks("EXM", scheme, pages, HTML)
        retrieval.TextIndex.build(chunks, HashEmbedder()).save(scheme, "EXM")
    tab = [c for c in chunking.build_chunks("EXM", "table", pages, HTML) if c["kind"] == "table"]
    assert "Net sales: September 28, 2024 = $391,035" in tab[0]["text"]
    assert tab[0]["page"] == next(i + 1 for i, t in enumerate(pages) if "391,035" in t)

    run.cmd_questions(argparse.Namespace(tickers=None))
    import pandas as pd
    qs = pd.read_csv(config.DATA / "questions.csv", dtype={"gold_pages": str})
    assert len(qs) == 6 and qs.gold_pages.notna().sum() >= 5

    run.cmd_evaluate(argparse.Namespace(methods=None, answers=True, answer_methods=["table_hybrid"],
                                        model=None, limit=None, tickers=None, second_look=True))
    card = pd.read_csv(config.RESULTS / "scorecard.csv")
    assert set(card.method) == {"plain_dense", "plain_hybrid", "table_dense", "table_hybrid",
                                "table_hybrid+second_look"}
    sl = card.set_index("method").loc["table_hybrid+second_look"]
    assert sl.n_answers == 6 and sl.answer_accuracy >= card.set_index("method").loc["table_hybrid", "answer_accuracy"]
    assert card.set_index("method").loc["table_hybrid", "page_recall@5"] > 0

    rows = extract_key_numbers(Searcher(), "EXM", "Example Corp", "table_hybrid", FACTS, ACC)
    assert {r["metric"] for r in rows} == {"total revenue", "net income", "diluted earnings per share", "total assets"}


def test_second_look_only_calls_on_misses(env, monkeypatch):
    import argparse
    import pandas as pd
    import run
    from finsight import chunking, llm, retrieval
    from finsight.render import load_pages, render

    render("EXM")
    pages = load_pages("EXM")
    for scheme in ("plain", "table"):
        retrieval.TextIndex.build(chunking.build_chunks("EXM", scheme, pages, HTML), HashEmbedder()).save(scheme, "EXM")
    run.cmd_questions(argparse.Namespace(tickers=None))
    seen, calls = set(), []

    def first_misses(question, hits, model=None):
        calls.append(question)
        if question not in seen:  # first pass: pretend nothing was found
            seen.add(question)
            return {"answer": "not found", "value": None, "scale": None, "pages": [], "found": False}
        return fake_answer(question, hits)

    monkeypatch.setattr(llm, "answer", first_misses)
    run.cmd_evaluate(argparse.Namespace(methods=["plain_dense"], answers=True, answer_methods=["plain_dense"],
                                        model=None, limit=None, tickers=None, second_look=True))
    card = pd.read_csv(config.RESULTS / "scorecard.csv").set_index("method")
    assert len(calls) == 12  # 6 first passes + 6 second looks
    assert card.loc["plain_dense", "answer_accuracy"] == 0
    assert card.loc["plain_dense+second_look", "answer_accuracy"] > 0


def test_langgraph_flow_second_look(env, monkeypatch):
    """Graph answers on the first look when it can, and takes one wider second look when it cannot."""
    from langchain_core.runnables import RunnableLambda
    from finsight import chunking, graph, retrieval
    from finsight.pipeline import Searcher
    from finsight.render import load_pages, render

    render("EXM")
    pages = load_pages("EXM")
    for scheme in ("plain", "table"):
        retrieval.TextIndex.build(chunking.build_chunks("EXM", scheme, pages, HTML), HashEmbedder()).save(scheme, "EXM")
    budgets = []

    def fake_chain(model):
        def run(inp):
            budgets.append(len(inp["context"]))
            if len(budgets) == 1:
                return {"answer": "not found", "value": None, "found": False, "pages": []}
            m = re.search(r"\d{1,3}(?:,\d{3})+", inp["context"])
            return {"answer": m.group(), "value": m.group(), "scale": "millions", "found": True, "pages": [1]}
        return RunnableLambda(run)

    monkeypatch.setattr(graph, "answer_chain", fake_chain)
    out = graph.ask(Searcher(), "What was net sales for 2024?", "EXM", "table_hybrid", require_value=True)
    assert out["attempts"] == 2 and out["found"] and out["value"]
    assert len(budgets) == 2

    budgets.clear()
    out = graph.ask(Searcher(), "What was net sales for 2024?", "EXM", "table_hybrid", max_attempts=1)
    assert out["attempts"] == 1 and len(budgets) == 1

    docs = graph.FinSightRetriever(searcher=Searcher(), ticker="EXM", method="plain_dense", k=3).invoke("net sales")
    assert len(docs) == 3 and "page" in docs[0].metadata


def test_prompt_injection_scorecard(env, monkeypatch):
    """Planted text reaches the model, the guarded prompt wraps excerpts, and attack success is scored."""
    import argparse
    import pandas as pd
    import run
    from finsight import chunking, injection, llm, retrieval
    from finsight.render import load_pages, render

    render("EXM")
    pages = load_pages("EXM")
    for scheme in ("plain", "table"):
        retrieval.TextIndex.build(chunking.build_chunks("EXM", scheme, pages, HTML), HashEmbedder()).save(scheme, "EXM")
    run.cmd_questions(argparse.Namespace(tickers=None))
    qs = pd.read_csv(config.DATA / "questions.csv")
    path = config.RESULTS / "answers" / "plain_dense.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"qid": q, "correct": True}) + "\n" for q in qs.qid))

    def gullible(question, hits, model=None, guarded=False):
        """Obeys planted text unless guarded; checks the excerpt tags are there when guarded."""
        ctx = llm.format_context(hits, guarded)
        assert ("<excerpt page=" in ctx) == guarded
        q = qs[qs.question == question].iloc[0]
        fake_text, _ = injection.fake_figure(float(q.value), q.unit)
        if not guarded and fake_text in ctx:
            v = fake_text.split()[0].lstrip("$")
            return {"answer": v, "value": v, "scale": "millions", "found": True}
        if not guarded and injection.CANARY in ctx:
            return {"answer": "see " + injection.CANARY, "value": None, "found": True}
        return {"answer": "", "value": f"{q.value / (1 if q.unit == 'USD/shares' else 1e6):,.2f}",
                "scale": "millions", "found": True}

    monkeypatch.setattr(llm, "answer", gullible)
    run.cmd_injection(argparse.Namespace(limit=4, attacks=None, method="plain_dense", model=None))
    card = pd.read_csv(config.RESULTS / "injection_scorecard.csv").set_index(["attack", "defence"])
    for a in injection.ATTACKS:
        assert card.loc[(a, "none"), "attack_success"] == 1.0
        assert card.loc[(a, "guarded"), "attack_success"] == 0.0
    assert card.loc[("clean", "guarded"), "accuracy"] == 1.0
    assert card.loc[("override", "none"), "accuracy"] == 0.0
