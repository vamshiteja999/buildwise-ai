"""Central configuration for BuildWise AI. All paths are relative to the project root unless absolute."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {os.getenv(name)!r}") from exc

def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Invalid number for {name}: {os.getenv(name)!r}") from exc

DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw")).expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = (ROOT / DATA_DIR).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "data/qdrant")).expanduser()
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = (ROOT / STORAGE_DIR).resolve()
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
COLLECTION = os.getenv("QDRANT_COLLECTION", "bw_c")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen3:1.7b")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", ANSWER_MODEL)  # model that decides chat / search / follow-up
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b")
EMBED_BATCH_SIZE = env_int("EMBED_BATCH_SIZE", 8)
EMBED_TIMEOUT = env_int("EMBED_TIMEOUT", 180)
GENERATE_TIMEOUT = env_int("GENERATE_TIMEOUT", 240)
MAX_RETRIES = env_int("MAX_RETRIES", 3)
CHUNK_TARGET_WORDS = env_int("CHUNK_TARGET_WORDS", 350)
CHUNK_MAX_WORDS = env_int("CHUNK_MAX_WORDS", 450)
CHUNK_OVERLAP_SENTENCES = env_int("CHUNK_OVERLAP_SENTENCES", 1)
MIN_PAGE_TEXT_CHARS = env_int("MIN_PAGE_TEXT_CHARS", 35)
OCR_ENABLED = env_bool("OCR_ENABLED", True)
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_DPI = env_int("OCR_DPI", 150)
TOP_K_VECTOR = env_int("TOP_K_VECTOR", 12)
TOP_K_KEYWORD = env_int("TOP_K_KEYWORD", 12)
FINAL_TOP_K = env_int("FINAL_TOP_K", 5)
RERANK_ENABLED = env_bool("RERANK_ENABLED", False)
RERANK_MODEL = os.getenv("RERANK_MODEL", ANSWER_MODEL)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 12000)
TEMPERATURE = env_float("TEMPERATURE", 0.1)
# Switchable experiments (both default OFF so a baseline run is unchanged).
PROMPT_V2 = env_bool("PROMPT_V2", True)            # stricter answer/refusal prompt
CITATION_BOOST = env_bool("CITATION_BOOST", False)
QUERY_UNDERSTANDING = env_bool("QUERY_UNDERSTANDING", True)  # greetings, follow-ups, several questions, exclusions
AGENT_RETRY = env_bool("AGENT_RETRY", False)          # retry once with alternative searches when the first attempt refuses
AGENT_RETRY_QUERIES = int(os.getenv("AGENT_RETRY_QUERIES", "2"))
VERIFY = env_bool("VERIFY", True)  # second check: refuse answers the cited text does not support
KEYWORD_V2 = env_bool("KEYWORD_V2", True)  # BM25 keyword search (stopwords ignored, rare words weigh more)  # boost chunks whose file/section matches a number in the query
EXTRACT_V2 = env_bool("EXTRACT_V2", True)            # row-preserving tables/drawings, keep edition lines, drop alt-text junk
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
GOLDEN_CSV = ROOT / os.getenv("GOLDEN_CSV", "golden_questions.csv")
EVAL_REPORT = REPORT_DIR / os.getenv("EVAL_REPORT", "evaluation_results.md")
INGEST_REPORT = REPORT_DIR / "ingestion_report.json"
