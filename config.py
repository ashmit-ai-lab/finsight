"""Central settings. Secrets live in .env (copy .env.example to .env)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(ROOT / ".env")

# Folders
DATA = ROOT / "data"
RAW = DATA / "raw"          # downloaded filing html + official numbers (json)
PDF = DATA / "pdf"          # filing rendered to pdf, so every method shares the same pages
PAGES = DATA / "pages"      # page text per filing
INDEX = DATA / "index"      # search indexes
RESULTS = ROOT / "results"  # scorecards and per question logs
for _d in (RAW, PDF, PAGES, INDEX, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# Companies (ticker -> display name). Non banks, so the same accounting tags exist for all.
COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "KO": "Coca-Cola",
    "NKE": "Nike",
}

# SEC asks every script to identify itself: "Your Name your.email@example.com"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

# Groq (OpenAI compatible endpoint). Check https://console.groq.com/docs/models if a model is retired.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")      # answers + extraction
FALLBACK_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]  # app only, never in evaluation
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")   # grades free text answers
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")             # "", low, medium, high
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")  # optional: answer from page images

# Retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VISUAL_MODEL = os.getenv("VISUAL_MODEL", "vidore/colSmol-256M")  # use vidore/colqwen2.5-v0.2 on a GPU
VISUAL_DPI = 96
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TABLE_ROWS_PER_CHUNK = 10
CONTEXT_CHARS = 6000   # same context budget for every method, so answers are compared fairly

# News mood
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
