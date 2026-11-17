"""News mood: classify recent headlines about a company as positive, negative or neutral."""
import re
from urllib.parse import quote_plus

import config


def fetch_headlines(company: str, n: int = 30) -> list:
    import feedparser
    url = (f"https://news.google.com/rss/search?q={quote_plus(company + ' stock')}"
           "&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:n]:
        title = re.sub(r"\s+-\s+[^-]+$", "", e.get("title", "")).strip()  # drop " - Source"
        out.append({"title": title, "link": e.get("link"), "published": e.get("published", ""),
                    "source": e.get("source", {}).get("title", "")})
    return out


class SentimentModel:
    def __init__(self, name: str = config.SENTIMENT_MODEL):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=name, truncation=True)

    def predict(self, texts: list) -> list:
        return [{"label": r["label"].lower(), "score": float(r["score"])}
                for r in self.pipe(texts, batch_size=32)]


def mood_summary(preds: list) -> dict:
    n = max(1, len(preds))
    pos = sum(p["label"] == "positive" for p in preds)
    neg = sum(p["label"] == "negative" for p in preds)
    return {"positive": pos, "negative": neg, "neutral": len(preds) - pos - neg,
            "net_mood": round((pos - neg) / n, 3)}


def evaluate_on_tweets(model: SentimentModel, limit: int = None) -> dict:
    """Honest check: FinBERT was trained on news sentences (Financial PhraseBank), so we test it on a
    different source (finance tweets) and compare with a tiny model trained on that source."""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")  # 0 bearish, 1 bullish, 2 neutral
    train, test = ds["train"], ds["validation"]
    texts, y = test["text"], test["label"]
    if limit:
        texts, y = texts[:limit], y[:limit]
    to_id = {"negative": 0, "positive": 1, "neutral": 2}
    fin = [to_id.get(p["label"], 2) for p in model.predict(list(texts))]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    base = list(clf.predict(vec.transform(texts)))
    majority = [max(set(train["label"]), key=list(train["label"]).count)] * len(y)

    def m(pred):
        return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4)}

    return {"n_test": len(y), "finbert_off_the_shelf": m(fin), "tfidf_logreg_in_domain": m(base),
            "majority_class": m(majority)}
