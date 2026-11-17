"""FinSight demo app.  Run:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

import config
from finsight import sec
from finsight.graph import ask as graph_ask
from finsight.pipeline import METHODS, Searcher, extract_key_numbers
from finsight.render import page_image

st.set_page_config(page_title="FinSight", layout="wide")

LABELS = {
    "table_hybrid": "Table aware + hybrid search",
    "table_dense": "Table aware + dense search",
    "plain_hybrid": "Plain chunks + hybrid search",
    "plain_dense": "Plain chunks + dense search (baseline)",
    "visual": "Page images (visual search)",
}


@st.cache_resource
def get_searcher():
    return Searcher()


def available_methods():
    ms = [m for m in LABELS if METHODS[m][0] != "visual" and (config.INDEX / METHODS[m][0]).exists()]
    if any((config.INDEX / "visual").glob("*.pt")):
        ms.append("visual")
    return ms


ready = [t for t in config.COMPANIES if (config.INDEX / "plain" / f"{t}.json").exists()]
st.title("FinSight")
st.caption("Ask questions about company annual reports, pull out key numbers checked against official SEC data, "
           "and read the mood of recent news.")
if not ready:
    st.error("No indexes found. Run the setup steps in the README first.")
    st.stop()

with st.sidebar:
    ticker = st.selectbox("Company", ready, format_func=lambda t: f"{config.COMPANIES[t]} ({t})")
    method = st.selectbox("Search method", available_methods(), format_func=LABELS.get)
    meta = sec.load_meta(ticker)
    st.markdown(f"Annual report for period ending **{meta['report_date']}**  \n[Source filing]({meta['url']})")

s = get_searcher()
MODELS = [config.LLM_MODEL] + [m for m in config.FALLBACK_MODELS if m != config.LLM_MODEL]


def with_fallback(fn):
    """If one model's free daily allowance is used up, quietly try the next one."""
    last = None
    for model in MODELS:
        try:
            return fn(model), model
        except Exception as e:  # daily limit, retired model, network
            last = e
    st.error(f"No model could answer right now: {str(last)[:200]}")
    st.stop()

ask_tab, extract_tab, news_tab, score_tab = st.tabs(["Ask", "Extract numbers", "News mood", "Scorecard"])

with ask_tab:
    q = st.text_input("Question", placeholder=f"What was {config.COMPANIES[ticker]}'s total revenue last year?")
    if q:
        with st.spinner("Searching and answering"):
            out, used = with_fallback(lambda m: graph_ask(s, q, ticker, method, model=m))
        st.subheader(out.get("answer", ""))
        looks = "with a second, wider look" if out.get("attempts", 1) > 1 else "on the first look"
        st.caption(f"Answered by {used} {looks} (LangGraph flow)")
        if not out.get("found"):
            st.warning("The retrieved pages did not contain a clear answer.")
        cited = [p for p in (out.get("pages") or []) if isinstance(p, int)]
        st.write("Cited pages:", ", ".join(map(str, cited)) or "none",
                 " | Retrieved pages:", ", ".join(map(str, out["retrieved_pages"][:5])))
        if cited:
            st.image(page_image(ticker, cited[0], dpi=110), caption=f"Page {cited[0]}", width="stretch")
        with st.expander("Retrieved passages"):
            for h in out["context"]:
                st.markdown(f"**Page {h['page']}** ({h['kind']})")
                st.text(h["text"][:1500])

with extract_tab:
    st.write("Reads the report and pulls out each headline number for the latest year, then checks it "
             "against the official number the company tagged in its SEC filing.")
    if st.button("Extract key numbers"):
        with st.spinner("Extracting"):
            rows, used = with_fallback(lambda m: extract_key_numbers(
                s, ticker, config.COMPANIES[ticker], method, sec.load_facts(ticker), meta["accession"], model=m))
        df = pd.DataFrame(rows)
        st.metric("Matches official data", f"{int(df.match.sum())} / {len(df)}")
        st.dataframe(df, width="stretch")

with news_tab:
    if st.button("Read latest headlines"):
        from finsight.news import SentimentModel, fetch_headlines, mood_summary

        @st.cache_resource
        def sentiment():
            return SentimentModel()

        heads = fetch_headlines(config.COMPANIES[ticker])
        if not heads:
            st.warning("No headlines returned.")
        else:
            preds = sentiment().predict([h["title"] for h in heads])
            summ = mood_summary(preds)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positive", summ["positive"])
            c2.metric("Negative", summ["negative"])
            c3.metric("Neutral", summ["neutral"])
            c4.metric("Net mood", summ["net_mood"])
            st.dataframe(pd.DataFrame([{**h, **p} for h, p in zip(heads, preds)]), width="stretch")

with score_tab:
    card = config.RESULTS / "scorecard.csv"
    if card.exists():
        st.write("How each search method did on the automatic test set (correct answers from SEC XBRL data).")
        st.dataframe(pd.read_csv(card), width="stretch")
        bym = config.RESULTS / "recall5_by_metric.csv"
        if bym.exists():
            st.write("Page recall at 5, by metric")
            st.dataframe(pd.read_csv(bym), width="stretch")
    else:
        st.info("Run  python run.py evaluate  to create the scorecard.")
    ne = config.RESULTS / "news_eval.json"
    if ne.exists():
        st.write("News mood model check on finance tweets (a source it was not trained on)")
        st.json(json.loads(ne.read_text()))
