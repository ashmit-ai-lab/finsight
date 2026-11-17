"""Download the latest annual report (form 10-K) and the official XBRL numbers from SEC EDGAR."""
import json
import time
from pathlib import Path

import requests

import config

_last_call = 0.0


def _get(url: str) -> requests.Response:
    global _last_call
    if not config.SEC_USER_AGENT:
        raise SystemExit('Set SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT="Ashmit Singh you@example.com"')
    wait = 0.15 - (time.time() - _last_call)  # SEC allows about 10 requests per second
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": config.SEC_USER_AGENT}, timeout=120)
    _last_call = time.time()
    r.raise_for_status()
    return r


def ticker_to_cik() -> dict:
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def latest_annual_filing(cik: int) -> dict:
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = sub["filings"]["recent"]
    for i, form in enumerate(rec["form"]):  # newest first
        if form == "10-K":
            return {
                "cik": cik,
                "accession": rec["accessionNumber"][i],
                "primary_doc": rec["primaryDocument"][i],
                "filing_date": rec["filingDate"][i],
                "report_date": rec["reportDate"][i],
            }
    raise RuntimeError(f"No 10-K found for CIK {cik}")


def download(ticker: str, cik: int) -> Path:
    out = config.RAW / ticker
    out.mkdir(parents=True, exist_ok=True)
    meta = latest_annual_filing(cik)
    acc = meta["accession"].replace("-", "")
    meta["url"] = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{meta['primary_doc']}"
    meta["ticker"] = ticker
    (out / "filing.htm").write_bytes(_get(meta["url"]).content)
    facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    (out / "facts.json").write_text(json.dumps(facts))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_meta(ticker: str) -> dict:
    return json.loads((config.RAW / ticker / "meta.json").read_text())


def load_facts(ticker: str) -> dict:
    return json.loads((config.RAW / ticker / "facts.json").read_text())
