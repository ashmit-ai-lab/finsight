"""Command line entry point.

  python run.py download        fetch filings + official numbers from SEC
  python run.py render          filing html -> pdf -> page text
  python run.py index [--visual]   build search indexes (visual needs colpali-engine)
  python run.py questions       build the auto test set from XBRL
  python run.py evaluate [--answers] [--limit N]   scorecard
  python run.py news-eval       check the news mood model honestly
  python run.py check-graph     confirm the LangGraph flow reproduces the evaluated results
"""
import argparse
import json
import random

import pandas as pd

import config
from finsight import chunking, llm, retrieval, scoring, sec
from finsight.pipeline import METHODS, Searcher
from finsight.render import load_pages, render


def tickers(args):
    return [t.upper() for t in args.tickers] if args.tickers else list(config.COMPANIES)


def cmd_download(args):
    ciks = sec.ticker_to_cik()
    for t in tickers(args):
        out = sec.download(t, ciks[t])
        print(f"{t}: {sec.load_meta(t)['url']} -> {out}")


def cmd_render(args):
    for t in tickers(args):
        render(t)
        print(f"{t}: {len(load_pages(t))} pages")


def cmd_index(args):
    emb = None if args.visual_only else retrieval.Embedder()
    for t in tickers(args) if emb else []:
        pages = load_pages(t)
        html = (config.RAW / t / "filing.htm").read_text(encoding="utf-8", errors="ignore")
        for scheme in ("plain", "table"):
            chunks = chunking.build_chunks(t, scheme, pages, html)
            retrieval.TextIndex.build(chunks, emb).save(scheme, t)
            n_tab = sum(c["kind"] == "table" for c in chunks)
            print(f"{t} {scheme}: {len(chunks)} chunks ({n_tab} table chunks)")
    if args.visual or args.visual_only:
        from finsight import visual
        vr = visual.VisualRetriever()
        for t in tickers(args):
            print(f"{t} visual: embedding page images on {vr.device}")
            visual.build(t, vr)


def cmd_questions(args):
    from finsight.questions import make_questions
    rows = []
    for t in tickers(args):
        meta = sec.load_meta(t)
        qs = make_questions(t, config.COMPANIES[t], sec.load_facts(t), meta["accession"], load_pages(t))
        rows += qs
        print(f"{t}: {len(qs)} questions, {sum(bool(q['gold_pages']) for q in qs)} with gold pages")
    pd.DataFrame(rows).to_csv(config.DATA / "questions.csv", index=False)
    print(f"saved {len(rows)} questions to data/questions.csv")


def _available_methods():
    ms = [m for m, (scheme, _) in METHODS.items() if scheme != "visual"]
    if any((config.INDEX / "visual").glob("*.pt")):
        ms.append("visual")
    return ms


def _load_questions():
    qs = pd.read_csv(config.DATA / "questions.csv", dtype={"gold_pages": str}).fillna({"gold_pages": ""})
    manual = config.ROOT / "questions_manual.csv"
    if manual.exists():
        m = pd.read_csv(manual, dtype={"gold_pages": str}).fillna({"gold_pages": ""})
        m["type"] = "free_text"
        qs = pd.concat([qs, m], ignore_index=True)
    return qs


def _cache(method):
    path = config.RESULTS / "answers" / f"{method}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["qid"]] = r
    return path, done


def cmd_evaluate(args):
    qs = _load_questions()
    if args.tickers:
        qs = qs[qs.ticker.isin(tickers(args))]
    methods = args.methods or _available_methods()
    s = Searcher()
    detail, card = [], []
    # 1. retrieval: no LLM needed, runs on every question that has gold pages
    for _, q in qs[qs.gold_pages != ""].iterrows():
        gold = {int(p) for p in q.gold_pages.split()}
        for m in methods:
            pages = retrieval.ranked_pages(s.search(q.question, q.ticker, m, k=30))
            detail.append({"qid": q.qid, "ticker": q.ticker, "metric": q.get("metric", ""), "method": m,
                           "r@1": scoring.recall_at_k(pages, gold, 1), "r@3": scoring.recall_at_k(pages, gold, 3),
                           "r@5": scoring.recall_at_k(pages, gold, 5), "rr": scoring.reciprocal_rank(pages, gold),
                           "top_pages": " ".join(map(str, pages[:5])), "gold": q.gold_pages})
    det = pd.DataFrame(detail)
    det.to_csv(config.RESULTS / "retrieval_details.csv", index=False)

    # 2. answers: costs LLM calls, so it is cached and can be limited
    acc = {}
    if args.answers:
        ans_methods = args.answer_methods or [m for m in ("plain_dense", "table_hybrid", "visual") if m in methods]
        pool = list(qs.index)
        random.Random(0).shuffle(pool)
        sel = qs.loc[pool[: args.limit] if args.limit else pool]
        caches = {m: _cache(m) for m in ans_methods}
        try:
            # question by question, every method in turn, so a stopped run stays balanced
            for i, (_, q) in enumerate(sel.iterrows()):
                for m in ans_methods:
                    path, done = caches[m]
                    if q.qid in done:
                        continue
                    rec = _grade(q, m, s.ask(q.question, q.ticker, m, model=args.model))
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                    done[q.qid] = rec
                    print(f"  {i + 1}/{len(sel)} {m:13s} {'ok' if rec['correct'] else 'x '} {q.qid}")
            if args.second_look:
                # only questions the first pass could not answer cost a new call
                for m in list(ans_methods):
                    name = f"{m}+second_look"
                    caches[name] = path2, done2 = _cache(name)
                    base = caches[m][1]
                    for _, q in sel.iterrows():
                        if q.qid in done2 or q.qid not in base:
                            continue
                        first = base[q.qid]
                        if first["found"] and (q.type != "numeric" or first["value"] not in (None, "", "null")):
                            rec = {**first, "method": name, "second_look": False}
                        else:
                            rec = {**_grade(q, name, s.second_look(q.question, q.ticker, m, model=args.model)),
                                   "second_look": True}
                            print(f"  second look {m:13s} {'ok' if rec['correct'] else 'x '} {q.qid}")
                        with path2.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, default=str) + "\n")
                        done2[q.qid] = rec
                    ans_methods.append(name)
        except llm.DailyLimit as e:
            print(f"\nGroq daily limit reached. Progress is saved; run the same command later to continue.\n  {e}\n")
        # score only questions every method has answered, so the comparison is like for like
        common = [qid for qid in sel.qid if all(qid in caches[m][1] for m in ans_methods)]
        for m in ans_methods:
            acc[m] = [float(caches[m][1][qid]["correct"]) for qid in common]

    for m in methods + [m for m in acc if m not in methods]:
        d = det[det.method == m] if len(det) else det
        row = {"method": m, "n_retrieval": len(d)}
        if len(d):
            lo, hi = scoring.bootstrap_ci(list(d["r@5"]))
            row.update({"page_recall@1": d["r@1"].mean(), "page_recall@3": d["r@3"].mean(),
                        "page_recall@5": d["r@5"].mean(), "r@5_95ci": f"{lo:.2f} to {hi:.2f}", "mrr": d["rr"].mean()})
        if m in acc and acc[m]:
            lo, hi = scoring.bootstrap_ci(acc[m])
            row.update({"n_answers": len(acc[m]), "answer_accuracy": sum(acc[m]) / len(acc[m]),
                        "acc_95ci": f"{lo:.2f} to {hi:.2f}"})
        card.append(row)
    card = pd.DataFrame(card)
    card.to_csv(config.RESULTS / "scorecard.csv", index=False)
    md = card.to_markdown(index=False, floatfmt=".3f")
    (config.RESULTS / "scorecard.md").write_text(md, encoding="utf-8")
    print(md)
    if len(det):
        by_metric = det.pivot_table(index="metric", columns="method", values="r@5", aggfunc="mean")
        by_metric.to_csv(config.RESULTS / "recall5_by_metric.csv")
        print("\npage recall@5 by metric\n" + by_metric.to_markdown(floatfmt=".2f"))


def _grade(q, method, out):
    if q.type == "numeric":
        ok = scoring.number_correct(out.get("value"), out.get("scale"), float(q.value), q.unit)
    else:
        ok = bool(llm_judge(q, out))
    return {"qid": q.qid, "method": method, "correct": ok, "found": bool(out.get("found")),
            "value": out.get("value"), "scale": out.get("scale"), "answer": out.get("answer"),
            "pages": out.get("pages"), "truth": q.get("value", q.get("reference_answer"))}


def llm_judge(q, out):
    return llm.judge(q.question, str(q.reference_answer), str(out.get("answer"))).get("correct", False)


def cmd_check_graph(args):
    """Run the LangGraph flow on cached questions and check it gives the same results as the plain path."""
    from finsight.graph import ask as graph_ask
    m = args.method
    _, done = _cache(f"{m}+second_look")
    if not done:
        raise SystemExit(f"No cached results for {m}+second_look; run evaluate --answers --second-look first.")
    qs = _load_questions()
    s, same, n = Searcher(), 0, 0
    try:
        for qid, old in list(done.items())[: args.limit]:
            q = qs[qs.qid == qid].iloc[0]
            out = graph_ask(s, q.question, q.ticker, m, model=args.model, require_value=q.type == "numeric")
            new = _grade(q, m, out)
            n += 1
            same += new["correct"] == old["correct"]
            print(f"  {qid}: before {'ok' if old['correct'] else 'x'}, graph {'ok' if new['correct'] else 'x'}"
                  f" ({out['attempts']} look{'s' if out['attempts'] > 1 else ''})")
    except llm.DailyLimit as e:
        print(f"Groq daily limit reached after {n} questions. {e}")
    print(f"LangGraph flow matched the earlier result on {same} of {n} questions")


def cmd_news_eval(args):
    from finsight.news import SentimentModel, evaluate_on_tweets
    res = evaluate_on_tweets(SentimentModel(), limit=args.limit)
    (config.RESULTS / "news_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "render", "index", "questions", "evaluate", "news-eval", "check-graph"):
        sp = sub.add_parser(name)
        sp.add_argument("--tickers", nargs="*")
        if name == "index":
            sp.add_argument("--visual", action="store_true", help="also build the page image index")
            sp.add_argument("--visual-only", action="store_true", help="only build the page image index")
        if name == "evaluate":
            sp.add_argument("--methods", nargs="*", choices=list(METHODS))
            sp.add_argument("--answers", action="store_true", help="also score LLM answers (uses Groq)")
            sp.add_argument("--answer-methods", nargs="*", choices=list(METHODS))
            sp.add_argument("--model", default=None)
            sp.add_argument("--second-look", action="store_true",
                            help="when the model says not found, ask again with twice the passages")
        if name in ("evaluate", "news-eval"):
            sp.add_argument("--limit", type=int, default=None)
        if name == "check-graph":
            sp.add_argument("--method", default="plain_dense", choices=list(METHODS))
            sp.add_argument("--limit", type=int, default=10)
            sp.add_argument("--model", default=None)
    args = p.parse_args()
    {"download": cmd_download, "render": cmd_render, "index": cmd_index, "questions": cmd_questions,
     "evaluate": cmd_evaluate, "news-eval": cmd_news_eval,
     "check-graph": cmd_check_graph}[args.cmd](args)


if __name__ == "__main__":
    main()
