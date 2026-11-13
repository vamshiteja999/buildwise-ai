import difflib
import hashlib, json, logging, math, re, time, uuid
from pathlib import Path
from urllib.parse import urlparse
import fitz
import requests
from qdrant_client import QdrantClient, models
from config import (
    ROOT, DATA_DIR, STORAGE_DIR, COLLECTION, OLLAMA_URL, EMBED_MODEL,
    ANSWER_MODEL, ROUTER_MODEL, AGENT_RETRY, AGENT_RETRY_QUERIES, EMBED_BATCH_SIZE, EMBED_TIMEOUT, GENERATE_TIMEOUT,
    MAX_RETRIES, MIN_PAGE_TEXT_CHARS, OCR_ENABLED, OCR_LANGUAGE, OCR_DPI,
    CHUNK_TARGET_WORDS, CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTENCES,
    TOP_K_VECTOR, TOP_K_KEYWORD, FINAL_TOP_K, RERANK_ENABLED,
    MAX_CONTEXT_CHARS, TEMPERATURE, LOG_DIR, PROMPT_V2, CITATION_BOOST, EXTRACT_V2, KEYWORD_V2, VERIFY, QUERY_UNDERSTANDING, SUPPORTED_EXTENSIONS
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "buildwise.log"), logging.StreamHandler()]
)
log = logging.getLogger("buildwise")

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def qdrant_point_id(stable_hash):
    """Convert a stable SHA-256 hash into a deterministic UUID accepted by Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_hash))

def request_json(url, payload, timeout):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Local model request failed after {MAX_RETRIES} attempts: {last}")

def health_check():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Created document folder: %s", DATA_DIR)
    if not DATA_DIR.is_dir():
        raise RuntimeError(f"DATA_DIR is not a directory: {DATA_DIR}")
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=8).json()
    except Exception as exc:
        raise RuntimeError(f"Ollama is unavailable at {OLLAMA_URL}. Start Ollama and retry. Details: {exc}")
    names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
    if EMBED_MODEL.split(":")[0] not in names:
        raise RuntimeError(f"Embedding model '{EMBED_MODEL}' is missing. Run: ollama pull {EMBED_MODEL}")
    if ANSWER_MODEL.split(":")[0] not in names:
        raise RuntimeError(f"Answer model '{ANSWER_MODEL}' is missing. Run: ollama pull {ANSWER_MODEL}")
    try:
        probe = request_json(f"{OLLAMA_URL}/api/embed",
            {"model": EMBED_MODEL, "input": ["search_document: BuildWise environment check"]}, EMBED_TIMEOUT)
        if not probe.get("embeddings") or not probe["embeddings"][0]:
            raise RuntimeError("Embedding model returned an empty vector.")
    except Exception as exc:
        raise RuntimeError(f"Embedding model check failed: {exc}")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return True

_QDRANT_CLIENT = None


def _allow_sqlite_threads():
    """Streamlit runs each page action on a different thread. Qdrant's local store is SQLite, and on macOS it opens the
    connection with check_same_thread=True, so a click handled on another thread fails. Qdrant reads this class flag
    when it opens the store; setting it to False before the client is created allows use from any thread. Safe here:
    one shared client, short sequential writes."""
    try:
        from qdrant_client.local.persistence import CollectionPersistence
        CollectionPersistence.CHECK_SAME_THREAD = False
    except Exception:  # newer/older qdrant-client without this class: nothing to do
        pass


def get_client():
    # Qdrant embedded/local mode permits one active client per storage path.
    # Reuse a process-wide client to avoid lock conflicts during hybrid retrieval.
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _allow_sqlite_threads()
        _QDRANT_CLIENT = QdrantClient(path=str(STORAGE_DIR))
        if not _QDRANT_CLIENT.collection_exists(COLLECTION):
            _QDRANT_CLIENT.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
    return _QDRANT_CLIENT

def embed(texts, prefix="search_document: "):
    vectors = []
    for start in range(0, len(texts), max(1, EMBED_BATCH_SIZE)):
        batch = [prefix + t for t in texts[start:start + max(1, EMBED_BATCH_SIZE)]]
        data = request_json(f"{OLLAMA_URL}/api/embed", {"model": EMBED_MODEL, "input": batch}, EMBED_TIMEOUT)
        result = data.get("embeddings", [])
        if len(result) != len(batch):
            raise RuntimeError(f"Ollama returned {len(result)} embeddings for {len(batch)} inputs.")
        vectors.extend(result)
    return vectors

URL_RE = re.compile(r"https?://\S+", re.I)
TIMESTAMP_RE = re.compile(r"^\s*(?:\d{1,2}/\d{1,2}/\d{2,4}\s+)?\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\s*$", re.I)
PAGE_COUNTER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:/|of)\s*\d+\s*$", re.I)
DATE_LINE_RE = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$")
CFR_RE = re.compile(r"^\s*((?:29\s+CFR\s+)?\d{1,4}\.\d{1,4}(?:\([a-zA-Z0-9]+\))*)\s*(.*)$")
CLAUSE_RE = re.compile(r"^\s*((?:29\s+CFR\s+)?\d{1,4}\.\d{1,4}(?:\([a-zA-Z0-9]+\))*)\s*(.*)$")


def normalize_text(text):
    text = (text or "").replace("\x00", " ").replace("\\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_key(line):
    """Normalize a candidate header/footer line without changing document text."""
    return re.sub(r"\s+", " ", (line or "").strip()).casefold()


def _is_standalone_page_artifact(line):
    """Only classify complete standalone lines; dates/URLs inside prose are valid content."""
    value = (line or "").strip()
    if not value:
        return False
    return bool(
        PAGE_COUNTER_RE.fullmatch(value)
        or TIMESTAMP_RE.fullmatch(value)
        or DATE_LINE_RE.fullmatch(value)
        or re.fullmatch(r"(?:https?://|www\.)\S+", value, re.I)
        or re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", value, re.I)
    )


CELL_TIMESTAMP_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm]$")


def clean_page_noise(text, edge_noise_lines=None):
    """Remove only standalone page artifacts and text proven to be in page margins.

    URLs and dates embedded in paragraphs are retained. The optional edge line set
    is populated from PyMuPDF block coordinates in extract_pages().
    """
    edge_noise_lines = edge_noise_lines or set()
    kept = []
    for line in (text or "").splitlines():
        value = line.strip()
        if not value:
            kept.append(line)
            continue
        if " | " in value:
            # Rebuilt table rows can glue a print header/footer to other cells; drop artifact cells.
            rest = " | ".join(
                c for c in value.split(" | ")
                if not _is_standalone_page_artifact(c) and not CELL_TIMESTAMP_RE.match(c.strip())
                and _line_key(c) not in edge_noise_lines
            )
            if not rest:
                continue
            if rest != value:
                value, line = rest, rest
        if _is_standalone_page_artifact(value) or _line_key(value) in edge_noise_lines:
            continue
        kept.append(line)
    return normalize_text("\n".join(kept))


def extract_doc_header(first_page):
    lines = [x.strip() for x in first_page.splitlines() if x.strip()]
    header = {"part": "", "subpart": "", "standard_number": "", "document_title": ""}
    for line in lines[:35]:
        low = line.lower()
        if "part " in low and not header["part"]:
            header["part"] = line[:180]
        if "subpart " in low and not header["subpart"]:
            header["subpart"] = line[:180]
        m = re.search(r"\b(?:29\s+CFR\s+)?\d{1,4}\.\d{1,4}\b", line)
        if m and not header["standard_number"]:
            header["standard_number"] = m.group(0)
        if any(k in low for k in ("fall protection", "scaffolds", "stairways", "ladders")) and not header["document_title"]:
            header["document_title"] = line[:240]
    return header


def _pdf_edge_lines(page):
    """Return normalized text lines located in the top/bottom page margins."""
    edge_lines = set()
    try:
        page_rect = page.rect
        top_limit = page_rect.height * 0.085
        bottom_limit = page_rect.height * 0.90
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, block_text = block[:5]
            if y1 <= top_limit or y0 >= bottom_limit:
                for line in str(block_text).splitlines():
                    key = _line_key(line)
                    if key:
                        edge_lines.add(key)
    except Exception as exc:
        log.debug("Could not inspect PDF edge blocks: %s", exc)
    return edge_lines


JUNK_LINE_RE = re.compile(r"^\s*(?:square|circle|diamond|disc|bullet)\s+used\s+as\s+a\s+bullet\s*$", re.I)
VERSION_LINE_RE = re.compile(r"\b(?:edition|version|revision|rev\.?|issued)\b", re.I)


def _rebuild_rows(words, page_width):
    """Rebuild reading lines from word boxes (x0, y0, x1, y1, text).

    Words on the same visual line are joined in x order; a wide gap becomes ' | ' so a table row
    like 'Metal Roofs | A,B,C | 1,2,3,4,5' stays together. On large drawing sheets, very distant
    words become separate lines, grouped into vertical bands so legends stay contiguous.
    Returns (lines, gap_line_count).
    """
    words = [w for w in words if str(w[4]).strip()]
    if not words:
        return [], 0
    large = page_width > 1200
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur = [], [words[0]]
    for w in words[1:]:
        ref = cur[0]
        h = max(ref[3] - ref[1], 1.0)
        if abs((w[1] + w[3]) / 2 - (ref[1] + ref[3]) / 2) <= 0.5 * h:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)

    segments, gap_lines = [], 0
    for row in rows:
        row.sort(key=lambda w: w[0])
        cells, seg_cells, cell = [], [], [row[0]]
        had_gap = False
        for prev, w in zip(row, row[1:]):
            h = max(prev[3] - prev[1], w[3] - w[1], 1.0)
            gap = w[0] - prev[2]
            if large and gap > 6 * h:
                seg_cells.append(cell)
                cells.append(seg_cells)
                seg_cells, cell = [], [w]
            elif gap > 1.2 * h:
                seg_cells.append(cell)
                cell = [w]
                had_gap = True
            else:
                cell.append(w)
        seg_cells.append(cell)
        cells.append(seg_cells)
        if had_gap:
            gap_lines += 1
        for seg in cells:
            text = " | ".join(" ".join(str(x[4]) for x in c) for c in seg)
            x0, y0 = seg[0][0][0], seg[0][0][1]
            segments.append((x0, y0, text))
    if large:
        band = page_width / 6.0
        segments.sort(key=lambda s: (int(s[0] // band), s[1], s[0]))
    else:
        segments.sort(key=lambda s: (s[1], s[0]))
    return [s[2] for s in segments], gap_lines


SECTION_TOKEN = "<<<SECTION>>>"
JUNK_PREFIX_RE = re.compile(
    r"^(?:Dash bullet|Square|Circle|Diamond|Disc)(?=\s*[A-Z])(?!\s*(?:Feet|Foot|Footage|Inch|Inches|Yard|Meter))\s*")
KEYNOTE_ROW_RE = re.compile(r"^\d{2}\s\d{2}\s\d{2}\b")
TITLEBLOCK_LABEL_RE = re.compile(
    r"^(?:project\s+title|drawing\s+title|job\s+number|project\s+number|drawing\s+number|"
    r"sheet\s+number|issued\s+for\s*:?|revisions|drawn\s+by)\b", re.I)


def _split_drawing_sections(lines):
    """Split a rebuilt drawing sheet into a keynote legend, a title block and the remaining
    annotations, so each becomes its own chunk instead of one diluted page-sized chunk."""
    n = len(lines)
    used = [False] * n
    sections = []
    start = next((i for i, l in enumerate(lines) if re.search(r"\blegend\b", l, re.I) and len(l) < 40), None)
    if start is not None:
        idx, i = [start], start + 1
        while i < n:
            l = lines[i]
            if KEYNOTE_ROW_RE.match(l) or l.strip() in {"#", "DESCRIPTION"}:
                idx.append(i)
            elif len(idx) > 1 and idx[-1] == i - 1 and i + 1 < n and KEYNOTE_ROW_RE.match(lines[i + 1]):
                idx.append(i)  # wrapped description line between two legend rows
            else:
                break
            i += 1
        if len(idx) > 2:
            sections.append("\n".join(lines[j] for j in idx))
            for j in idx:
                used[j] = True
    labels = [i for i, l in enumerate(lines) if not used[i] and TITLEBLOCK_LABEL_RE.match(l)]
    if len(labels) >= 2:
        lo, hi = max(min(labels) - 12, 0), min(max(labels) + 3, n)
        idx = [j for j in range(lo, hi) if not used[j]]
        sections.append("SHEET TITLE BLOCK (project, address, job number, drawing number, issue stage)\n"
                        + "\n".join(lines[j] for j in idx))
        for j in idx:
            used[j] = True
    rest = [lines[j] for j in range(n) if not used[j]]
    if rest:
        sections.append("DRAWING ANNOTATIONS (dimensions, callouts and labels)\n" + "\n".join(rest))
    return sections


def _layout_text(page):
    """Row-preserving text for table-like or large-format pages; None to keep normal extraction."""
    try:
        words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
        width = page.rect.width
        lines, gap_lines = _rebuild_rows(words, width)
    except Exception as exc:
        log.debug("Layout rebuild failed: %s", exc)
        return None
    if not lines:
        return None
    if width > 1200:
        sections = _split_drawing_sections(lines)
        return ("\n" + SECTION_TOKEN + "\n").join(sections)
    if gap_lines >= 3 and gap_lines / len(lines) < 0.5:
        return "\n".join(lines)
    return None


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HEADING_MARK = "@@HEADING@@ "


def _docx_text(el):
    out = []
    for node in el.iter():
        if node.tag == W_NS + "t" and node.text:
            out.append(node.text)
        elif node.tag == W_NS + "tab":
            out.append(" ")
        elif node.tag in (W_NS + "br", W_NS + "cr"):
            out.append("\n")
    return "".join(out).strip()


def extract_docx_pages(path):
    """Read a .docx with the standard library only. Each heading starts a section; tables keep their rows."""
    import zipfile
    from xml.etree import ElementTree as ET
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W_NS + "body")
    sections, heading, blocks = [], "", []
    for el in (list(body) if body is not None else []):
        if el.tag == W_NS + "p":
            text = _docx_text(el)
            if not text:
                continue
            style = el.find(f"{W_NS}pPr/{W_NS}pStyle")
            sid = style.get(W_NS + "val", "") if style is not None else ""
            if re.match(r"(?i)^(heading\s*\d|title)", sid):
                if blocks:
                    sections.append((heading, "\n\n".join(blocks)))
                heading, blocks = text[:160], [text]
            else:
                blocks.append(text)
        elif el.tag == W_NS + "tbl":
            rows = []
            for tr in el.findall(W_NS + "tr"):
                cells = [" ".join(t for t in (_docx_text(p) for p in tc.findall(W_NS + "p")) if t)
                         for tc in tr.findall(W_NS + "tc")]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    if blocks:
        sections.append((heading, "\n\n".join(blocks)))
    return [(i, (HEADING_MARK + text if head else text), False, "")
            for i, (head, text) in enumerate(sections, 1)]


CSV_ROWS_PER_PAGE = 50


def extract_csv_pages(path):
    """Read a .csv with the standard library. Every row becomes 'Column: value | Column: value' on its own
    paragraph, so a row is never split and each one names its own columns. 50 rows make one section."""
    import csv
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return [(1, normalize_text(text), False, "")] if text.strip() else []
    names, seen = [], {}
    for i, h in enumerate(rows[0], 1):
        h = h.strip() or f"Column {i}"
        seen[h] = seen.get(h, 0) + 1
        names.append(h if seen[h] == 1 else f"{h} ({seen[h]})")
    lines = []
    for r in rows[1:]:
        cells = [f"{names[i] if i < len(names) else 'Column ' + str(i + 1)}: {c.strip()}"
                 for i, c in enumerate(r) if c.strip()]
        if cells:
            lines.append(" | ".join(cells))
    pages = []
    for start in range(0, len(lines), CSV_ROWS_PER_PAGE):
        group = lines[start:start + CSV_ROWS_PER_PAGE]
        head = f"Rows {start + 1} to {start + len(group)}"
        pages.append((len(pages) + 1, HEADING_MARK + head + "\n\n" + "\n\n".join(group), False, ""))
    return pages


def extract_pages(path):
    if path.suffix.lower() == ".csv":
        return extract_csv_pages(path)
    if path.suffix.lower() == ".docx":
        return extract_docx_pages(path)
    if path.suffix.lower() != ".pdf":
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        return [(1, clean_page_noise(text), False, "")]

    raw_pages = []
    with fitz.open(path) as pdf:
        page_count = len(pdf)
        edge_occurrences = {}
        page_edge_lines = []
        for index, page in enumerate(pdf):
            layout = _layout_text(page) if EXTRACT_V2 else None
            text = normalize_text(layout if layout else page.get_text("text"))
            if EXTRACT_V2:
                text = "\n".join(JUNK_PREFIX_RE.sub("", l) for l in text.splitlines() if not JUNK_LINE_RE.match(l))
            edge_lines = _pdf_edge_lines(page)
            source_url = next(iter(URL_RE.findall(text)), "")
            ocr_used = False
            if OCR_ENABLED and len(text) < MIN_PAGE_TEXT_CHARS:
                try:
                    import pytesseract
                    from PIL import Image
                    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    ocr_text = normalize_text(pytesseract.image_to_string(image, lang=OCR_LANGUAGE))
                    if len(ocr_text) > len(text):
                        text, ocr_used = ocr_text, True
                except Exception as exc:
                    log.warning("OCR unavailable for %s page %d: %s", path.name, index + 1, exc)
            raw_pages.append((index + 1, text, ocr_used, source_url))
            page_edge_lines.append(edge_lines)
            for key in edge_lines:
                edge_occurrences[key] = edge_occurrences.get(key, 0) + 1

    # A repeated line in page margins is treated as a running header/footer.
    # Requiring at least two pages avoids deleting one-off content from short PDFs.
    repeat_threshold = max(2, int(page_count * 0.20 + 0.999)) if page_count > 1 else 2
    repeated_edge_lines = {
        key for key, count in edge_occurrences.items() if count >= repeat_threshold
    }

    if EXTRACT_V2:
        # Keep repeated edition/version lines (for example "Florida Building Code, Eighth Edition (2023)")
        # unless the file name already states them (then the embedding prefix has them).
        stem_key = _line_key(re.sub(r"[_]+", " ", path.stem))
        repeated_edge_lines = {
            k for k in repeated_edge_lines
            if not (VERSION_LINE_RE.search(k) and len(k) <= 120 and k not in stem_key)
        }

    cleaned_pages = []
    for (page_num, text, ocr_used, source_url), edge_lines in zip(raw_pages, page_edge_lines):
        # URL/date/page markers are removed only when standalone. URLs captured above
        # remain available in source_url metadata even when their footer line is stripped.
        standalone_edge_artifacts = {
            key for key in edge_lines if _is_standalone_page_artifact(key)
        }
        cleaned = clean_page_noise(
            text, edge_noise_lines=standalone_edge_artifacts | repeated_edge_lines
        )
        cleaned_pages.append((page_num, cleaned, ocr_used, source_url))
    if EXTRACT_V2:
        cleaned_pages = _drop_repeated_edge_lines(cleaned_pages, path)
    return cleaned_pages


def _edge_key(line):
    """Line fingerprint that ignores page numbers, so 'X  Page 5 of 16' equals 'X  Page 6 of 16'."""
    low = re.sub(r"\bpage\s*\d+(?:\s*of\s*\d+)?", " ", line.lower())
    low = re.sub(r"\d+", "#", low)
    return re.sub(r"[^a-z#]+", " ", low).strip()


def _drop_repeated_edge_lines(pages, path):
    """Remove running headers/footers that repeat in the first or last 3 lines of many pages.

    Catches footers PyMuPDF returns as one line with the page number attached
    ('BD-002 Project Submission  Page 5 of 16'), which the margin-block check misses.
    """
    if len(pages) < 4:
        return pages
    stem_key = _edge_key(re.sub(r"[_]+", " ", path.stem))
    def edge_positions(text):
        lines = text.splitlines()
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return lines, set(idx[:3]) | set(idx[-3:])
    counts = {}
    for _, text, _, _ in pages:
        lines, edge = edge_positions(text)
        for key in {_edge_key(lines[i]) for i in edge}:
            if 8 <= len(key) <= 80:  # running headers/footers are short; long lines are content
                counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(len(pages) * 0.30 + 0.999))
    repeated = {k for k, n in counts.items() if n >= threshold
                and not (VERSION_LINE_RE.search(k) and k not in stem_key)}
    if not repeated:
        return pages
    out = []
    for page_num, text, ocr_used, source_url in pages:
        lines, edge = edge_positions(text)
        kept = [l for i, l in enumerate(lines) if not (i in edge and _edge_key(l) in repeated)]
        out.append((page_num, normalize_text("\n".join(kept)), ocr_used, source_url))
    return out

def sentence_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip()) if s.strip()]

def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))

def split_long_text(text):
    sentences = sentence_split(text)
    if not sentences:
        return [text] if text else []
    result, current = [], []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and word_count(candidate) > CHUNK_MAX_WORDS:
            result.append(" ".join(current))
            overlap = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            current = overlap + [sentence]
        else:
            current.append(sentence)
    if current:
        result.append(" ".join(current))
    return result

def parent_section(section):
    matches = re.findall(r"\([^)]+\)", section)
    if len(matches) <= 1:
        return ""
    return re.sub(r"\([^)]+\)$", "", section)

def parse_osha_clauses(page_text):
    lines = page_text.splitlines()
    clauses, current_id, current_lines = [], None, []
    preamble = []
    for line in lines:
        match = CLAUSE_RE.match(line)
        if match:
            if current_id:
                clauses.append((current_id, normalize_text(" ".join(current_lines))))
            elif line.strip():
                preamble.append(line.strip())
            current_id = match.group(1).replace("29 CFR ", "").strip()
            remainder = match.group(2).strip()
            current_lines = [remainder] if remainder else []
        elif current_id:
            if line.strip():
                current_lines.append(line.strip())
        elif line.strip():
            preamble.append(line.strip())
    if current_id:
        clauses.append((current_id, normalize_text(" ".join(current_lines))))
    return clauses, normalize_text(" ".join(preamble))

def document_type(path, header):
    name = path.name.lower()
    if re.match(r"\s*\d{3,4}\.\d{1,4}\b", path.stem):   # e.g. "1926.502 - Fall protection..."
        return "osha_regulation"
    if "permit" in name or "application" in name:
        return "permit_guide"
    if "bid" in name:
        return "bid_package"
    if "spec" in name:
        return "specification"
    if "building code" in name or "florida building" in name:
        return "building_code"
    return "general_document"

def make_chunks(path, pages, document_id):
    first_text = pages[0][1] if pages else ""
    header = extract_doc_header(first_text)
    # Regulatory PDFs often repeat the standard number/title in the filename but
    # omit it from the first extracted text lines. Preserve it in chunk metadata
    # and the embedding prefix so direct citation queries can find the document.
    if not header.get("standard_number"):
        filename_standard = re.search(r"\b(\d{1,4}\.\d{1,4})\b", path.stem)
        if filename_standard:
            header["standard_number"] = filename_standard.group(1)
    if not header.get("document_title"):
        header["document_title"] = re.sub(r"[_-]+", " ", path.stem).strip()[:240]
    dtype = document_type(path, header)
    expanded = []
    for page_num, page_text, ocr_used, source_url in pages:
        parts = [x.strip() for x in (page_text or "").split(SECTION_TOKEN) if x.strip()]
        for part in (parts or [page_text]):
            sect = ""
            if part and part.startswith(HEADING_MARK):
                part = part[len(HEADING_MARK):]
                sect = part.split("\n", 1)[0].strip()[:160]
            expanded.append((page_num, part, ocr_used, source_url, sect))
    pages = expanded
    all_records = []
    # Parent clauses are retained across page boundaries for the same source document.
    clause_text_by_id = {}
    for page_num, page_text, ocr_used, source_url, sect in pages:
        if not page_text:
            continue
        if dtype == "osha_regulation":
            clauses, preamble = parse_osha_clauses(page_text)
            if not clauses:
                clauses = [(header.get("standard_number") or "Unclassified", page_text)]
            for section, clause_text in clauses:
                if not clause_text:
                    continue
                clause_text_by_id[section] = clause_text
                parent = parent_section(section)
                context_parts = []
                while parent:
                    if parent in clause_text_by_id:
                        context_parts.insert(0, clause_text_by_id[parent])
                    parent = parent_section(parent)
                enriched = " ".join(context_parts + [clause_text]).strip()
                for part_index, raw_part in enumerate(split_long_text(enriched)):
                    prefix = " > ".join(x for x in [
                        header.get("document_title"), header.get("part"),
                        header.get("subpart"), header.get("standard_number"), section
                    ] if x)
                    all_records.append({
                        "raw_text": raw_part, "embedding_text": f"{prefix}\n{raw_part}",
                        "section": section, "page_number": page_num, "source_url": source_url,
                        "ocr_used": ocr_used, "part_index": part_index
                    })
        else:
            # General PDFs and text documents use paragraph-aware chunks; table rows remain intact.
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
            current = []
            for para in paragraphs:
                if word_count(para) > CHUNK_MAX_WORDS:
                    if current:
                        raw = "\n\n".join(current); current = []
                        all_records.append({"raw_text": raw, "embedding_text": f"{path.stem}\n{raw}",
                            "section": sect, "page_number": page_num, "source_url": source_url, "ocr_used": ocr_used})
                    for part in split_long_text(para):
                        all_records.append({"raw_text": part, "embedding_text": f"{path.stem}\n{part}",
                            "section": sect, "page_number": page_num, "source_url": source_url, "ocr_used": ocr_used})
                else:
                    candidate = "\n\n".join(current + [para])
                    if current and word_count(candidate) > CHUNK_TARGET_WORDS:
                        raw = "\n\n".join(current)
                        all_records.append({"raw_text": raw, "embedding_text": f"{path.stem}\n{raw}",
                            "section": sect, "page_number": page_num, "source_url": source_url, "ocr_used": ocr_used})
                        current = [para]
                    else:
                        current.append(para)
            if current:
                raw = "\n\n".join(current)
                all_records.append({"raw_text": raw, "embedding_text": f"{path.stem}\n{raw}",
                    "section": sect, "page_number": page_num, "source_url": source_url, "ocr_used": ocr_used})
    # Sibling clauses are grouped without crossing the configured word target.
    if dtype == "osha_regulation":
        grouped, current = [], []
        for rec in all_records:
            candidate = " ".join([x["raw_text"] for x in current] + [rec["raw_text"]])
            if current and word_count(candidate) > CHUNK_TARGET_WORDS:
                grouped.append(current); current = [rec]
            else:
                current.append(rec)
        if current: grouped.append(current)
        merged = []
        for group in grouped:
            if len(group) == 1:
                merged.append(group[0]); continue
            raw = "\n\n".join(x["raw_text"] for x in group)
            unique = list(dict.fromkeys(x["section"] for x in group))
            sections = unique[0] if len(unique) == 1 else f"{unique[0]} through {unique[-1]}"
            first = group[0]
            merged.append({**first, "raw_text": raw, "embedding_text": first["embedding_text"] + "\n" + raw,
                           "section": sections, "page_number": first["page_number"]})
        all_records = merged
    result = []
    for idx, rec in enumerate(all_records):
        raw = normalize_text(rec["raw_text"])
        if not raw:
            continue
        result.append({
            "id": sha256_text(f"{document_id}:{idx}:{raw}"),
            "vector_text": rec["embedding_text"],
            "text": raw,
            "payload": {
                "document_id": document_id, "source": path.name, "source_url": rec["source_url"],
                "document_type": dtype, "page_number": rec["page_number"],
                "section": rec["section"], "content_hash": sha256_text(raw),
                "ocr_used": rec["ocr_used"], "document_title": header.get("document_title", ""),
                "part": header.get("part", ""), "subpart": header.get("subpart", ""),
                "standard_number": header.get("standard_number", "")
            }
        })
    return result

def ingest_all():
    started = time.time()
    health_check()
    paths = sorted(p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    client = get_client()
    existing = {}
    try:
        offset = None
        while True:
            points, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                existing[payload.get("document_id")] = payload.get("content_hash")
            if offset is None: break
    except Exception as exc:
        log.warning("Could not read existing point manifest; continuing safely: %s", exc)
    docs_indexed = docs_skipped = page_count = chunk_count = 0
    seen_document_ids = set(existing)
    errors = []
    for path in paths:
        try:
            raw_bytes = path.read_bytes()
            doc_id = sha256_bytes(raw_bytes)
            if doc_id in seen_document_ids:
                docs_skipped += 1
                log.info("Skipped duplicate/already indexed document: %s", path.name)
                continue
            pages = extract_pages(path)
            page_count += len(pages)
            chunks = make_chunks(path, pages, doc_id)
            if not chunks:
                log.warning("No extractable text: %s", path.name)
                continue
            vectors = embed([x["vector_text"] for x in chunks])
            points = [
                models.PointStruct(id=qdrant_point_id(x["id"]), vector=v, payload={**x["payload"], "text": x["text"], "chunk_hash": x["id"]})
                for x, v in zip(chunks, vectors)
            ]
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            docs_indexed += 1
            seen_document_ids.add(doc_id)
            chunk_count += len(chunks)
            log.info("Indexed %s: %d pages, %d chunks", path.name, len(pages), len(chunks))
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            log.exception("Failed to ingest %s", path.name)
    elapsed = round(time.time() - started, 2)
    summary = {"documents_found": len(paths), "documents_indexed": docs_indexed,
               "documents_skipped": docs_skipped, "pages_processed": page_count,
               "chunks_indexed": chunk_count, "errors": errors, "elapsed_seconds": elapsed}
    from config import INGEST_REPORT, REPORT_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    INGEST_REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nINGESTION SUMMARY")
    print(json.dumps(summary, indent=2))
    return summary

STOPWORDS = frozenset("""a an and are as at be been by can could did do does for from had has have how i if in into is it
its may must of on or our shall should so than that the their there these they this to was we were what when where which
who whom whose why will with would you your""".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_token_cache = {}


def _tokens(text):
    """Lowercase tokens; 'bd-002' and '1926.501' also give their parts; trailing plural 's' removed."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        parts = [tok] + (re.split(r"[.\-]", tok) if re.search(r"[.\-]", tok) else [])
        for part in parts:
            if len(part) < 2 or part in STOPWORDS:
                continue
            out.append(part[:-1] if len(part) > 3 and part.endswith("s") and not part.endswith("ss") else part)
    return out


def bm25_rank(query, docs, k1=1.5, b=0.75):
    """docs: list of (id, text). Returns [(id, score)] best first. Stopwords ignored, rare words weigh more."""
    q_terms = set(_tokens(query))
    if not q_terms or not docs:
        return []
    doc_tokens = []
    for doc_id, text in docs:
        key = (doc_id, hash(text))
        toks = _token_cache.get(key)
        if toks is None:
            toks = _token_cache[key] = _tokens(text)
        doc_tokens.append(toks)
    n = len(docs)
    avg_len = sum(len(t) for t in doc_tokens) / n or 1.0
    sets = [set(t) for t in doc_tokens]
    df = {t: sum(1 for st in sets if t in st) for t in q_terms}
    scored = []
    for (doc_id, _), toks in zip(docs, doc_tokens):
        if not toks:
            continue
        tf = {}
        for t in toks:
            if t in q_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def keyword_search(query, limit):
    if KEYWORD_V2:
        client = get_client()
        points, offset = [], None
        while True:
            batch, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
            points.extend(batch)
            if offset is None:
                break
        by_id = {pt.id: pt for pt in points}
        docs = [(pt.id, (pt.payload or {}).get("text", "") + " " + ((pt.payload or {}).get("section") or ""))
                for pt in points]
        return [(by_id[i], sc) for i, sc in bm25_rank(query, docs)[:limit]]
    terms = [x.lower() for x in re.findall(r"[a-zA-Z0-9.()_-]+", query) if len(x) > 1]
    if not terms:
        return []
    client = get_client()
    # Qdrant scroll is local; lexical scoring is deterministic and dependency-free.
    results, offset = [], None
    while len(results) < 1000:
        points, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
        results.extend(points)
        if offset is None: break
    scored = []
    for point in results:
        payload = point.payload or {}
        text = (payload.get("text", "") + " " + payload.get("section", "")).lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, point))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(p, float(s)) for s, p in scored[:limit]]

CITE_IN_QUERY = re.compile(r"\b(\d{3,4}\.\d{1,4})\b")

def citation_boost(query, payload):
    """Small bonus for chunks whose file name / standard number / section contains a number
    that appears in the question (for example 1926.501)."""
    nums = CITE_IN_QUERY.findall(query)
    if not nums:
        return 0.0
    hay = " ".join([payload.get("source", ""), payload.get("standard_number", ""), payload.get("section", "")])
    return 0.02 if any(n in hay for n in nums) else 0.0

def hybrid_search(query, limit=FINAL_TOP_K):
    client = get_client()
    query_vector = embed([query], prefix="search_query: ")[0]
    vector_hits = client.search(collection_name=COLLECTION, query_vector=query_vector,
                                limit=TOP_K_VECTOR, with_payload=True)
    keyword_hits = keyword_search(query, TOP_K_KEYWORD)
    fused = {}
    for rank, point in enumerate(vector_hits, 1):
        fused[point.id] = [point, 1.0 / (60 + rank)]
    for rank, (point, _) in enumerate(keyword_hits, 1):
        if point.id in fused:
            fused[point.id][1] += 1.0 / (60 + rank)
        else:
            fused[point.id] = [point, 1.0 / (60 + rank)]
    if CITATION_BOOST:
        for item in fused.values():
            item[1] += citation_boost(query, item[0].payload or {})
    ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
    if RERANK_ENABLED:
        # Lightweight local reranking proxy: exact query-term coverage + reciprocal-rank fusion.
        terms = set(re.findall(r"\w+", query.lower()))
        ranked.sort(key=lambda x: (sum(t in (x[0].payload or {}).get("text", "").lower() for t in terms), x[1]), reverse=True)
    return [x[0] for x in ranked[:limit]]

REFUSAL_TEXT = "I could not find this in the documents."
LAST_VERIFY = {}
# A drafted answer whose own first sentence says the fact is missing is a refusal in substance.
DECLINE_RE = re.compile(
    r"\b(not|n't)\s+(explicitly\s+|clearly\s+|directly\s+)?(stated|specified|provided|mentioned|included|given|found|addressed|listed)\b"
    r"|\b(does|do|did)\s+not\s+(specify|state|provide|mention|contain|include|address|say|give|list)\b"
    r"|\bno\s+(specific\s+)?(information|mention|details?)\b", re.I)


def verify_answer(question, answer, context_parts):
    """Second check on a drafted answer. Returns the answer, or the refusal sentence if it is not supported.
    1) Free check: the first sentence says the fact is not stated -> refuse.
    2) One short model call: does the cited text itself state the fact? Fails open (keeps the answer)
       if the model gives anything other than SUPPORTED / UNSUPPORTED."""
    first = re.split(r"(?<=[.!?])\s", answer.strip(), maxsplit=1)[0]
    if DECLINE_RE.search(first):
        LAST_VERIFY.update(decision="refused", reason="answer says fact is not stated")
        return REFUSAL_TEXT
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    blocks = [b for b in context_parts if any(b.startswith(f"[{n}]") for n in cited)] or context_parts
    context = "\n".join(blocks)[:6000]
    prompt = (
        "You check an answer against source text.\n"
        "Reply SUPPORTED only if the CONTEXT itself states the specific fact that the ANSWER gives to the "
        "QUESTION (the same value, rule or name).\n"
        "Reply UNSUPPORTED if the answer relies on a related topic, a different subpart or document, a general "
        "statement that points elsewhere, or anything not written in the CONTEXT.\n"
        "Reply with one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\nVERDICT:")
    try:
        data = request_json(f"{OLLAMA_URL}/api/generate",
            {"model": ANSWER_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
             "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 6}}, GENERATE_TIMEOUT)
        verdict = re.sub(r"<think>.*?</think>", "", data.get("response") or "", flags=re.S).strip().upper()
    except Exception as exc:  # never block an answer because the checker failed
        LAST_VERIFY.update(decision="kept", reason=f"checker error: {exc}")
        return answer
    if verdict.startswith("UNSUPPORTED"):
        LAST_VERIFY.update(decision="refused", reason="model check: unsupported")
        return REFUSAL_TEXT
    LAST_VERIFY.update(decision="kept", reason=f"model check: {verdict[:20] or 'empty'}")
    return answer


def _merge_hits(hit_lists, limit):
    """Reciprocal-rank fusion of several ranked hit lists (same scheme as hybrid_search)."""
    fused = {}
    for hits in hit_lists:
        for rank, point in enumerate(hits, 1):
            if point.id in fused:
                fused[point.id][1] += 1.0 / (60 + rank)
            else:
                fused[point.id] = [point, 1.0 / (60 + rank)]
    return [x[0] for x in sorted(fused.values(), key=lambda x: x[1], reverse=True)[:limit]]


def answer_question(question, extra_queries=None):
    hits = hybrid_search(question, FINAL_TOP_K)
    if extra_queries:   # retry: widen the evidence with searches for alternative wordings
        lists = [hybrid_search(question, FINAL_TOP_K * 2)] + [hybrid_search(q, FINAL_TOP_K * 2) for q in extra_queries]
        hits = _merge_hits(lists, FINAL_TOP_K)
    if not hits:
        return "I could not find this in the documents.", []
    context_parts, citations = [], []
    for i, point in enumerate(hits, 1):
        p = point.payload or {}
        text = p.get("text", "")
        citation = f"[{i}] Source: {p.get('source','unknown')}, page {p.get('page_number','?')}, section {p.get('section') or 'not specified'}"
        block = f"{citation}\n{text}"
        if sum(len(x) for x in context_parts) + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        citations.append(citation)
    if PROMPT_V2:
        instructions = (
            "You are BuildWise, a construction-document assistant. Use only the supplied context.\n"
            "Your reply MUST start with one line: 'ANSWERABLE: YES' or 'ANSWERABLE: NO'.\n"
            "Write YES only if the context states the specific fact the question asks for. "
            "Write NO if the context only covers a related topic (a different subpart, document or "
            "project), mentions the subject without giving the requested value, or does not address the "
            "question. After 'ANSWERABLE: NO' write nothing else.\n"
            "After 'ANSWERABLE: YES', give the answer first, in at most 6 short sentences or bullet points. "
            "End every sentence or bullet with a citation such as [1], using only the [number] labels shown "
            "in the context; never invent section numbers. Quote numbers and units exactly as written.\n\n"
        )
    else:
        instructions = (
            "You are BuildWise, a construction-document assistant. Answer only using the supplied context. "
            "Do not use outside knowledge. If the context does not contain the answer, reply exactly: "
            "'I could not find this in the documents.' Cite supporting sources using [number] and include "
            "the source, page, and section. Do not invent citations.\n\n"
        )
    prompt = instructions + f"CONTEXT:\n{chr(10).join(context_parts)}\n\nQUESTION: {question}\nANSWER:"
    data = request_json(f"{OLLAMA_URL}/api/generate",
        {"model": ANSWER_MODEL, "prompt": prompt, "stream": False,
         "think": False,
         "keep_alive": "30m",
         "options": {"temperature": TEMPERATURE, "num_ctx": 8192, "num_predict": 450}},
        GENERATE_TIMEOUT)
    raw = data.get("response") or ""
    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if PROMPT_V2:
        m = re.search(r"ANSWERABLE:\s*(YES|NO)\b[ \t]*[.:]?\s*", answer, re.I)
        if m:
            if m.group(1).upper() == "NO":
                return "I could not find this in the documents.", citations
            answer = answer[m.end():]
            nxt = re.search(r"ANSWERABLE:\s*(YES|NO)\b", answer, re.I)
            if nxt:  # the small model sometimes repeats the block; keep the first one
                answer = answer[:nxt.start()]
            answer = answer.strip()
    if not answer:
        answer = "I could not find this in the documents."
    if VERIFY and answer != REFUSAL_TEXT:
        answer = verify_answer(question, answer, context_parts)
    return answer, citations


# ---------------------------------------------------------------------------------------------
# Agentic retry: if the first attempt refuses, search again with alternative wordings (once).
# The code decides when to retry and when to stop; the model only writes the alternative queries.
# ---------------------------------------------------------------------------------------------
LAST_AGENT = {}


def alternative_queries(question, n=2):
    prompt = (
        "Write alternative search queries for finding the answer to this question in construction documents "
        "(building code, OSHA rules, permit forms, drawing notes). Use different words, synonyms and the key technical "
        "terms. Do not answer the question. One query per line, no numbering, no explanations.\n\n"
        f"QUESTION: {question}\n\nQUERIES:")
    data = request_json(f"{OLLAMA_URL}/api/generate",
        {"model": ROUTER_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
         "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 90}}, GENERATE_TIMEOUT)
    text = re.sub(r"<think>.*?</think>", "", data.get("response") or "", flags=re.S)
    out, seen = [], {question.strip().lower()}
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip(" \t\"'")
        line = re.sub(r"^(?:query|search)\s*\d*\s*:\s*", "", line, flags=re.I)
        if len(line.split()) >= 2 and line.lower() not in seen and not line.startswith(("/", "<")):
            seen.add(line.lower())
            out.append(line)
    return out[:n]


def answer_with_retry(question):
    """answer_question plus one retry on refusal when AGENT_RETRY is on. Same return shape."""
    LAST_AGENT.clear()
    answer, citations = answer_question(question)
    if not AGENT_RETRY or answer.strip() != REFUSAL_TEXT:
        return answer, citations
    try:
        alts = alternative_queries(question, AGENT_RETRY_QUERIES)
    except Exception as exc:
        LAST_AGENT.update(retried=False, reason=f"could not write alternative queries: {exc}")
        return answer, citations
    if not alts:
        LAST_AGENT.update(retried=False, reason="no usable alternative queries")
        return answer, citations
    answer2, citations2 = answer_question(question, extra_queries=alts)
    LAST_AGENT.update(retried=True, queries=alts, recovered=answer2.strip() != REFUSAL_TEXT)
    log.info("Retry after refusal: queries=%s recovered=%s", alts, LAST_AGENT["recovered"])
    return (answer2, citations2) if answer2.strip() != REFUSAL_TEXT else (answer, citations)


# ---------------------------------------------------------------------------------------------
# Query understanding: chat vs question, follow-ups, several questions in one message, exclusions.
# ---------------------------------------------------------------------------------------------
LAST_PLAN = {}
WELCOME = ("Hi! I answer questions about your construction documents (OSHA fall protection, building code, "
           "permits, drawings, uploaded specifications) and cite where each answer comes from. What would you like to know?")
CHAT_REPLIES = [
    (re.compile(r"^(hi|hello|hey|hiya|yo|greetings|good\s+(morning|afternoon|evening))(\s+(there|bot|buildwise))?[\s!.,]*$", re.I),
     "Hello! " + WELCOME[4:]),
    (re.compile(r"^(?:oh\s+)?(?:wow|woah|whoa)[,\s]+(?:great|cool|nice|awesome|amazing|good)[\s!.]*$|^(amazing|wonderful|excellent|good job|well done)[\s!.]*$", re.I),
     "Glad that helps! Ask another question about your documents any time."),
    (re.compile(r"^(ok(ay)?|alright|sounds good|cool|nice|great|awesome|got it|makes sense)(\s+(cool|thanks|great|nice|thank you|got it))?[\s!.]*$", re.I),
     "Glad that helps! Ask another question about your documents any time."),
    (re.compile(r"^(thanks|thank you|thx|ty|cheers)(\s+(so much|a lot|again|very much))?[\s!.]*$"
                r"|^(great|cool|nice|ok|okay|got it|perfect|sure)[\s!.]*$", re.I),
     "You're welcome! Ask another question about your documents any time."),
    (re.compile(r"^(bye|goodbye|see you|see ya|good night|cya)(\s+(later|soon))?[\s!.]*$", re.I),
     "Goodbye! Come back any time you need something from your documents."),
    (re.compile(r"^(what can you do|help|how does this work|how are you)[\s?!.]*$", re.I),
     "I'm doing well, thanks for asking! What construction document can I help you with?"),
    # A short acknowledgement is conversational, not a document-search query.
    (re.compile(r"^(nope|nah|not really|nothing else|that's all|thats all)[\s!.]*$", re.I),
     "No problem. What would you like to look up in your construction documents?"),
]
EXCLUSION_RE = re.compile(r"\b(don'?t|do not|dont)\s+(give|show|include|want|tell|list|need)\b|\bwithout\s+(giving|showing|including)\b|\bexclud(e|ing)\b", re.I)
# Recognize numbered/bulleted multi-part messages even when list text starts in lowercase.
NUMBERED_RE = re.compile(r"(?m)^\s*(?:\(?\d{1,2}[.)]|[-*•])\s+\S")
ANAPHORA_RE = re.compile(r"\b(it|its|that|those|them|these|this one|the same|above|previous|former|latter)\b|^\s*(and|also|what about|how about|why|so|then)\b", re.I)


IDENTITY_RE = re.compile(
    r"\bwho\s+(?:are|r)\s+(?:you|u)\b|\bwhat\s+are\s+you\b|\bwhat(?:'s|\s+is)?\s+your\s+name\b|\b(?:what|which)\s+(?:llm\s+|ai\s+)?model\s+(?:are\s+you|do\s+you|is\s+this|you)\b"
    r"|\bare\s+you\s+(?:a\s+|an\s+)?(?:qwen|llama|gpt|chatgpt|claude|gemini|ai|bot|llm|robot|human)\b|\bwho\s+(?:made|built|created)\s+you\b", re.I)


IMAGE_CAPABILITY_RE = re.compile(
    r"(?:can|could|do|will)\s+(?:you|u)\s+(?:show|display|see|view|read|open|give|send|find|get)\s+(?:me\s+)?(?:any\s+|the\s+|some\s+|all\s+)?"
    r"(?:images?|pictures?|photos?|pics?|drawings?|diagrams?|figures?|charts?|graphics|scans?|scanned\s+(?:pages?|documents?|files?))(?:\s+in\s+(?:the\s+)?(?:documents?|files?|pdfs?))?[\s?!.]*", re.I)
UPLOAD_HELP_RE = re.compile(
    r"(?:how\s+(?:do|can|to)\s+(?:i|we|u)\s+|can\s+(?:i|we)\s+)?(?:add|upload|import|index|load)\s+(?:a\s+|my\s+|new\s+|more\s+|another\s+)?(?:documents?|files?|pdfs?|docs?|docx|csv)(?:\s+(?:here|to\s+(?:this\s+app|the\s+app|this|it|the\s+system)))?[\s?!.]*", re.I)
UPLOAD_INTENT_RE = re.compile(
    r"\b(?:i\s+)?(?:want|wanna|need|would\s+like|like|plan|trying|going|ready|let\s+me|lemme|help\s+me|how\s+(?:do|can|to)\s+(?:i|we|u)|can\s+(?:i|we))"
    r"(?:\s+\w+){0,3}?\s+(?:add|upload|import|index|load)(?:ing)?\s+(?:a\s+|an\s+|my\s+|new\s+|more\s+|another\s+|some\s+|the\s+)?"
    r"(?:documents?|files?|pdfs?|docs?|docx|csv|word\s+files?)\b", re.I)
IMAGE_REQUEST_RE = re.compile(
    r"\b(?:show|display|give|send|find|get|share)\b(?:\s+\w+){0,3}?\s+(?:images?|pictures?|photos?|pics?|photographs?)\b", re.I)
IMAGE_REPLY = ("I can't show images or page pictures. I read documents as text (scanned pages and drawing sheets are read "
               "with OCR) and cite the source file and page, so you can open that page yourself. Ask what a drawing, "
               "keynote or figure says and I'll search its text.")
UPLOAD_REPLY = ("Open the Admin page in the sidebar, enter the admin password, and upload PDF, Word (.docx), CSV, text or "
                "Markdown files. They are indexed right away and show up when you ask 'what documents do you have'.")


def identity_reply():
    return (f"I'm BuildWise AI, a local assistant for construction documents. I run on your own computer with the "
            f"language model {ANSWER_MODEL} through Ollama, so your documents never leave this machine. "
            "Ask me about the indexed documents, or say 'what documents do you have'.")


def fast_chat_reply(question):
    """Answer obvious small talk without calling the model or searching documents."""
    q = re.sub(r"\s+", " ", (question or "").strip())
    if not q or len(q.split()) > 12:
        return None
    if IDENTITY_RE.search(q):
        return identity_reply()
    if IMAGE_CAPABILITY_RE.fullmatch(q):
        return IMAGE_REPLY
    if IMAGE_REQUEST_RE.search(q):
        return IMAGE_REPLY + " If you tell me the topic (for example pool lifts or guardrails), I can look up what the documents say about it."
    if UPLOAD_HELP_RE.fullmatch(q) or UPLOAD_INTENT_RE.search(q):
        return UPLOAD_REPLY
    if len(q.split()) > 8:
        return None
    if re.fullmatch(r"(?:(?:hey|hi|hello|yo)[,\s]+)(?:qwen|buildwise|bot|assistant|ai|there)[,\s]+(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up)[?!.]*", q, re.I):
        return "I'm doing well, thanks for asking! What construction document can I help you with?"
    for pattern, reply in CHAT_REPLIES:
        if pattern.fullmatch(q):
            return reply

    # Also recognize a greeting followed by a small-talk question, e.g.
    # "hey how are you?". Keep this an exact full-message match so a real
    # construction question appended to a greeting is not swallowed.
    if re.fullmatch(
        r"(?:hi|hello|hey|hiya|yo|greetings)[,\s]+(?:and\s+)?how\s+are\s+you[?!.]*",
        q,
        re.I,
    ):
        return "I'm doing well, thanks for asking! What construction document can I help you with?"

    if re.fullmatch(r"(?:(?:hey|hi|hello|yo)[,\s]+)?(?:what'?s\s*up|wh?a+ts?\s*up|wassup|wazzup|sup|howdy|how'?s\s+it\s+going|hows\s+it\s+going|how\s+r\s+u|how\s+are\s+u|how\s+you\s+doing|how\s+are\s+things)[?!.\s]*", q, re.I):
        return "Hey! I'm ready. What would you like to look up in your construction documents?"

    # Tolerate a one-character typo in the common phrase "how are you?".
    # Fuzzy matching is deliberately restricted to this fixed phrase so technical
    # construction questions are not accidentally classified as small talk.
    normalized = re.sub(r"[^a-z0-9 ]", "", q.casefold()).strip()
    if len(normalized.split()) == 3 and difflib.SequenceMatcher(None, normalized, "how are you").ratio() >= 0.88:
        return "I'm doing well, thanks for asking! What construction document can I help you with?"
    if re.fullmatch(r"(?:h+i+|h+e+y+|y+o+|h+e+l+o+)[!. ]*", q, re.I):    # 'hii', 'heyy', 'yooo'
        return "Hello! " + WELCOME[4:]
    return _fuzzy_small_talk(q)


SMALL_TALK_WORDS = ["hello", "hiya", "howdy", "hey there", "hello there", "greetings", "good morning", "good afternoon",
                    "good evening", "good night", "thanks", "thank you", "thanks a lot", "goodbye", "see you later",
                    "how are you", "whats up", "wassup", "okay", "alright", "awesome"]


def _fuzzy_small_talk(q):
    """Typo-tolerant small talk: 'helo', 'thnaks', 'gud morning' map to the nearest known phrase.
    Only short messages are considered, and the match must be close, so real questions are not caught."""
    norm = re.sub(r"(.)\1{2,}", r"\1", re.sub(r"[^a-z ]", "", q.lower())).strip()   # 'hellooo' -> 'hello'
    if not norm or len(norm.split()) > 3 or len(norm) < 4:
        return None
    best = difflib.get_close_matches(norm, SMALL_TALK_WORDS, n=1, cutoff=0.8)
    if not best:
        return None
    for pattern, reply in CHAT_REPLIES:
        if pattern.fullmatch(best[0]):
            return reply
    if best[0] in ("whats up", "wassup", "how are you"):
        return "Hey! I'm ready. What would you like to look up in your construction documents?"
    return None


FOLLOWUP_REF_RE = re.compile(r"\b(?:section|item|topic|part|option|number|step|point)\s+(?:[a-z]|\d+)\b", re.I)


def looks_like_followup(question):
    """A message that depends on earlier turns: pronouns, 'what about...', 'and...', section/item references, or very short."""
    q = question.strip()
    if ANAPHORA_RE.search(q) or FOLLOWUP_REF_RE.search(q):
        return True
    # A short message is usually a follow-up, unless it names something itself (a capitalised name or code such as
    # "BlueWave HVAC" or "BD-002" after the first word): then it is a complete question on its own.
    words = q.split()
    names_something = any(re.match(r"[A-Z][A-Za-z0-9-]*", w) and w not in ("I", "I'm") for w in words[1:])
    return len(words) <= 4 and not names_something


def needs_rewrite(question, history):
    """Cheap test that decides whether a message needs contextual rewriting/splitting."""
    if EXCLUSION_RE.search(question):
        return "exclusion"
    if question.count("?") >= 2 or len(NUMBERED_RE.findall(question)) >= 2:
        return "several questions"
    if len(question.split()) >= 8 and re.search(r"\band\s+(?:what|how|which|where|when|who|why|is|are|does|do)\b", question, re.I):
        return "several questions"
    # A phrase such as "what about roofs?" needs earlier conversation context.
    if ANAPHORA_RE.search(question):
        return "follow-up"
    # Short replies after a prior exchange may refer to that exchange.
    if history and len(question.split()) <= 10 and looks_like_followup(question):
        return "follow-up"
    return ""


def format_recent_history(history, max_messages=10, max_chars=9000):
    """Keep enough prior conversation to resolve references beyond the last two turns.

    Long answers retain both their opening (often where numbered topics are listed)
    and their ending, rather than being cut off after the first 300 characters.
    """
    lines = []
    for role, content in (history or [])[-max_messages:]:
        label = "User" if role == "user" else "Assistant"
        value = str(content or "").strip()
        if len(value) > 1100:
            value = value[:850] + "\n...[middle of message omitted]...\n" + value[-200:]
        lines.append(f"{label}: {value}")
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def parse_plan(text):
    """Parse the rewrite reply. Returns ('chat', []) or ('question', [queries]) or None if unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    kind = re.search(r"TYPE:\s*(CHAT|QUESTION)", text, re.I)
    queries = [q.strip(" \t\"'") for q in re.findall(r"(?im)^\s*QUERY:\s*(.+?)\s*$", text)]
    queries = [q for q in queries if 3 <= len(q) <= 400][:3]
    if kind and kind.group(1).upper() == "CHAT" and not queries:
        return "chat", []
    if queries:
        return "question", list(dict.fromkeys(queries))
    return None


def parse_route(text):
    """CHAT:/ANSWER:/SEARCH: reply -> ('chat'|'answer', text) or ('question', [queries]); None if unreadable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    first = re.match(r"\W*(CHAT|ANSWER|SEARCH)\s*:", text, re.I)
    if first and first.group(1).upper() in ("CHAT", "ANSWER"):
        reply = text[first.end():].strip()
        reply = re.split(r"\n\s*(?:CHAT|ANSWER|SEARCH)\s*:", reply, flags=re.I)[0].strip()
        return (first.group(1).lower(), reply) if reply else None
    queries = []
    for q in re.findall(r"^\W*SEARCH\s*:\s*(.+)$", text, re.I | re.M):
        q = re.sub(r"^\s*<[^>]*>\s*[-:]*\s*", "", q)       # echoed placeholder such as "<question> - ..."
        q = q.strip(" \t\"'<>")
        if len(q.split()) >= 2 and not q.startswith("/") and "no_think" not in q:
            queries.append(q)
    queries = queries[:3]
    return ("question", queries) if queries else None


def last_document_question(history):
    """Most recent user message that was a real document question (skips greetings, identity and thanks)."""
    for role, content in reversed(history or []):
        if role != "user":
            continue
        text = str(content or "")
        if len(text.split()) <= 2 or fast_chat_reply(text) or META_RE.search(text):
            continue
        return text
    return ""


TECHNICAL_HINT_RE = re.compile(
    r"\d|\b(?:inch\w*|feet|foot|ft|lbs?|pounds?|height|width|depth|minimum|maximum|requir\w*|permit\w*|code|osha|rail\w*|"
    r"roof\w*|slope|load|strength|fee|pool|lift|door|wall|floor|guard\w*|scaffold\w*|fall|section|chapter|page)\b", re.I)


def _with_context(question, history, why):
    """Fallback for a follow-up when the model could not rewrite it: keep the previous question's topic."""
    prev_user = last_document_question(history)
    if prev_user and why and why != "short message" and "exclusion" not in why and looks_like_followup(question):
        return f"{prev_user.strip()} {question.strip()}"
    return question


SUMMARY_RE = re.compile(r"\b(summari[sz]e|summary|recap|repeat|rephrase|shorter|simpler|simplify|in one (?:line|sentence)|tl;?dr|explain (?:that|it) again)\b", re.I)
_Q_WORDS = r"(?:what|how|which|where|when|who|why|is|are|does|do|can)"


def split_questions(question):
    """Split '1) a? 2) b?', 'a? b?' or 'what is x and what is y' into separate questions (rules, no model)."""
    q = question.strip()
    parts = [x.strip(" ,;") for x in re.split(r"(?:^|\s)\(?\d{1,2}[.)]\s+", q) if x.strip(" ,;")]
    if len(parts) < 2:
        parts = [x.strip() for x in re.split(r"(?<=\?)\s+", q) if x.strip()]
    if len(parts) < 2:
        parts = [x.strip(" ,;") for x in re.split(r"\s*[,;]?\s+and\s+(?=" + _Q_WORDS + r"\b)", q, flags=re.I) if x.strip(" ,;")]
    parts = [x if x.endswith("?") else x + "?" for x in parts if len(x.split()) >= 3]
    return parts[:3] if len(parts) >= 2 else None


def strip_exclusion(question):
    """'dont give me X, give me Y' -> 'Y';  'everything about A except B' -> 'everything about A'."""
    q = question.strip()
    m = re.search(r"(?:don'?t|do not|dont)\s+(?:give|show|include|want|tell|list|need)\b([^,;.]*)[,;.]\s*(?:just\s+|only\s+)?(?:(?:give|show|tell|list)\s+me\s+|i\s+want\s+)?(.+)$", q, re.I)
    if m and len(m.group(2).split()) >= 2:
        kept = m.group(2).strip(" .")
        tail = [w for w in re.findall(r"[A-Za-z0-9-]+", m.group(1))[-2:]]
        if len(kept.split()) <= 4 and tail:        # a short clause like 'the roofing ones' needs the shared topic
            kept = f"{kept} {' '.join(tail)}"
        return kept
    m = re.match(r"(.+?)\s+(?:except(?:\s+for)?|other than|excluding|without)\s+.+$", q, re.I)
    if m and len(m.group(1).split()) >= 2:
        return m.group(1).strip(" .,")
    return None


def _content_overlap(a, b):
    return len(set(_tokens(a)) & set(_tokens(b)))


def understand_query(question, history=None):
    """Return {'kind': 'chat'|'question', 'queries': [...], 'reply': str|None, 'how': str, 'why': str}."""
    history = history or []
    reply = fast_chat_reply(question)
    if reply:
        return {"kind": "chat", "queries": [], "reply": reply, "how": "fast", "why": "greeting or small talk"}
    why = needs_rewrite(question, history)
    if not why and not history and len(question.split()) <= 6:
        why = "short message"
    if not why:
        return {"kind": "question", "queries": [question], "reply": None, "how": "passthrough", "why": ""}
    if why == "several questions":
        parts = split_questions(question)
        if parts:
            return {"kind": "question", "queries": parts, "reply": None, "how": "rules", "why": why}
    if why == "exclusion":
        kept = strip_exclusion(question)
        if kept:
            return {"kind": "question", "queries": [_with_context(kept, history, "follow-up") if history else kept],
                    "reply": None, "how": "rules", "why": why}
    turns = format_recent_history(history)
    prompt = (
        "You are the router for BuildWise, an assistant that answers questions about construction documents "
        "(OSHA fall protection, building code, permits, drawings, uploaded specifications).\n"
        "Read the NEW MESSAGE and the CONVERSATION, then reply with exactly ONE of these forms:\n"
        "CHAT: <short friendly reply>   - greetings, thanks, small talk, or questions about you. Never invent document facts.\n"
        "ANSWER: <answer>   - ONLY when the message asks about the conversation itself (a summary or a repeat) and the "
        "full answer is already written in the CONVERSATION. Use only facts written there.\n"
        "SEARCH: <question>   - anything that needs facts from the documents. Replace vague words (it, that, this, a section "
        "letter, an item number) with the real subject from the CONVERSATION so the question stands alone. "
        "Write one SEARCH line per independent question, at most 3. "
        "If the user excludes something, search only for what they do want.\n"
        "If unsure, choose SEARCH and repeat the user's message unchanged.\n"
        "Write the real text after the colon, never the angle-bracket placeholder.\n\n"
        f"CONVERSATION:\n{turns or '(none)'}\n\nNEW MESSAGE: {question}\n\nReply:")
    try:
        data = request_json(f"{OLLAMA_URL}/api/generate",
            {"model": ROUTER_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
             "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 220}}, GENERATE_TIMEOUT)
        parsed = parse_route(data.get("response") or "")
    except Exception as exc:  # never lose a question because the rewriter failed
        return {"kind": "question", "queries": [_with_context(question, history, why)], "reply": None, "how": "fallback", "why": f"{why}; rewrite error: {exc}"}
    if parsed is None:
        return {"kind": "question", "queries": [_with_context(question, history, why)], "reply": None, "how": "fallback", "why": f"{why}; unreadable rewrite"}
    kind, payload = parsed
    if kind == "chat" and TECHNICAL_HINT_RE.search(question):
        kind, payload = "question", [question]   # numbers, units or construction words: never treat as small talk
        why += "; chat route overruled"
    if kind == "answer" and not SUMMARY_RE.search(question):
        kind, payload = "question", [question]   # only summary/repeat requests may skip the search
        why += "; answer not allowed"
    if kind in ("chat", "answer"):
        norm = lambda t: re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()
        if len(norm(payload).split()) < 4 or difflib.SequenceMatcher(None, norm(payload), norm(question)).ratio() > 0.7:
            payload = "Doing well, thanks! " + WELCOME[4:]  # the small model just echoed the message
        return {"kind": "chat", "queries": [], "reply": payload, "how": "model", "why": f"{why}; {kind}"}
    queries = [q.strip("<> ") for q in payload]
    prev_user = last_document_question(history)
    weak = (len(queries) == 1 and len(queries[0].split()) <= 9 and _content_overlap(queries[0], prev_user) < 3
            and why.startswith("follow-up") and looks_like_followup(question))
    if prev_user and len(queries) == 1 and (weak or difflib.SequenceMatcher(None, queries[0].lower(), question.lower()).ratio() > 0.85):
        # the rewrite did not add context: prepend the previous question so the search keeps its topic
        queries = [f"{prev_user.strip()} {question.strip()}"]
        why += "; added previous question"
    return {"kind": "question", "queries": queries, "reply": None, "how": "model", "why": why}


def _renumber(text, part):
    """[3] -> [2.3] so citations from different parts of a combined answer stay distinct."""
    return re.sub(r"\[(\d+)\]", lambda m: "[%d.%s]" % (part, m.group(1)), text)


FILE_TYPES = {"csv": (".csv",), "pdf": (".pdf",), "docx": (".docx",), "word": (".docx",), "spreadsheet": (".csv", ".xlsx"),
              "excel": (".xlsx", ".csv"), "text": (".txt", ".md")}
FILE_TYPE_RE = re.compile(
    r"\b(csv|pdf|docx|word|spreadsheet|excel)\b.*\b(?:fil+es?|documents?|docs|list)\b|\b(?:fil+es?|documents?|docs)\b.*\b(csv|pdf|docx|spreadsheet|excel)\b", re.I)
FILE_INTENT_RE = re.compile(
    r"\b(?:do\s+you\s+have|have\s+you\s+got|is\s+there|are\s+there|(?:don'?t|do\s+not)\s+see|not\s+in\s+(?:the\s+)?list|"
    r"need\s+to\s+know\s+about|tell\s+me\s+about|about\s+your|your\s+\w+\s+file|show\s+me|list)\b", re.I)
META_RE = re.compile(
    r"(?:what|which)\s+(?:\w+\s+){0,3}(?:documents?|files?|pdfs?|docs)\b.*\b(?:have|got|indexed|loaded|available|retrieve|search|cover)"
    r"|(?:what|which)\s+(?:types?|kinds?)\s+of\s+(?:documents?|files?)"
    r"|how\s+many\s+(?:documents?|files?|pdfs?|docs)"
    r"|(?:list|show)\s+(?:me\s+)?(?:all\s+|the\s+)?(?:your\s+)?(?:documents?|files?|sources)"
    r"|what\s+(?:can|could)\s+you\s+(?:retr\w*|search|find|look\s+up|answer|do)[\s?!.]*$"
    r"|what\s+do\s+you\s+know(?:\s+about)?\s*$", re.I)


def list_indexed_documents(with_pages=False):
    """[(source, chunk_count)] (or (source, chunk_count, last_page)) straight from the index, no model."""
    client = get_client()
    counts, pages, offset = {}, {}, None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for pt in batch:
            payload = pt.payload or {}
            src = payload.get("source") or "unknown"
            counts[src] = counts.get(src, 0) + 1
            pn = payload.get("page_number")
            if isinstance(pn, int):
                pages[src] = max(pages.get(src, 0), pn)
        if offset is None:
            break
    if with_pages:
        return [(n, counts[n], pages.get(n)) for n in sorted(counts)]
    return sorted(counts.items())


ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
            "ninth": 9, "9th": 9, "tenth": 10, "10th": 10, "eleventh": 11, "11th": 11, "twelfth": 12, "12th": 12}
ORDINAL_DOC_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r"|(?:document|file|number|no\.?|#)\s*(\d{1,2}))\b.*\b(?:document|file|list|pages?|passages?)|\b(?:document|file)\s+(?:number\s+)?(\d{1,2})\b", re.I)


def ordinal_reply(question, history):
    """'how many pages for the 3rd document in list?' after a document listing."""
    if len(question.split()) > 14:
        return None
    last = next((c for r, c in reversed(history or []) if r != "user"), "")
    if "documents indexed" not in str(last):
        return None
    m = ORDINAL_DOC_RE.search(question)
    if not m:
        return None
    n = None
    for tok in re.findall(r"[A-Za-z0-9]+", question.lower()):
        if tok in ORDINALS:
            n = ORDINALS[tok]
            break
    if n is None:
        d = re.search(r"\b(\d{1,2})\b", question)
        n = int(d.group(1)) if d else None
    try:
        docs = list_indexed_documents(with_pages=True)
    except Exception:
        return None
    if not n or n > len(docs):
        return None
    name, chunks, last_page = docs[n - 1]
    pages = f"{last_page} pages" if last_page else "a page count is not recorded for this file type"
    return f"Document {n} is {name}: {pages}, split into {chunks} searchable passages."


def meta_reply(question):
    if len(question.split()) > 28:
        return None
    ftype = FILE_TYPE_RE.search(question) and FILE_INTENT_RE.search(question) and not re.search(r"\d", question)
    if not ftype and (len(question.split()) > 18 or not META_RE.search(question)):
        return None
    try:
        docs = list_indexed_documents()
    except Exception:
        return None
    if ftype and not META_RE.search(question):
        kind = (ftype_kind := (FILE_TYPE_RE.search(question).group(1) or FILE_TYPE_RE.search(question).group(2) or "").lower())
        exts = FILE_TYPES.get(kind, ())
        found = [(n, c) for n, c in docs if n.lower().endswith(exts)]
        if not found:
            return f"I don't have any {kind.upper()} files indexed. Upload one on the Admin page and I can search it."
        lines = [f"Yes, I have {len(found)} {kind.upper()} file(s) indexed:"]
        lines += ["- %s (%d passages)" % (n, c) for n, c in found]
        lines.append("Ask a question about its contents and I'll search it.")
        return "\n".join(lines)
    if not docs:
        return "No documents are indexed yet. Upload some on the Admin page, then ask again."
    lines = ["I have %d documents indexed:" % len(docs)]
    lines += ["- %s (%d passages)" % (name, n) for name, n in docs]
    lines.append("Ask me anything these cover, such as OSHA fall protection, accessibility rules, permit requirements or drawing notes.")
    return "\n".join(lines)


SUMMARY_INTENT_RE = re.compile(
    r"\bsummar\w*|\boverview\b|\bdescribe\b|\bwhat\b.{0,25}\b(?:file|document|csv|pdf|sheet|spreadsheet)\b.{0,15}\babout\b"
    r"|\bwhat\s+(?:is|'s)\s+(?:in|inside)\s+(?:that|this|the)\b|\bwhat\s+(?:is|are)\b.{0,50}\babout\b|\b(?:know|learn|tell\s+me)\s+(?:more\s+)?about\b", re.I)
_GENERIC_NAME_WORDS = {"pdf", "csv", "docx", "the", "and", "for", "with", "building", "code", "administrative", "guideline",
                       "occupational", "safety", "health", "administration", "scope", "application", "definitions", "applicable",
                       "subpart", "florida", "2026", "ninth", "edition", "chapter", "services", "duty", "systems", "criteria",
                       "practices", "requirements", "training", "have", "fall", "protection"}


def _name_tokens(name):
    base = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    return {t for t in re.findall(r"[a-z0-9]+", base.lower()) if len(t) >= 4 and t not in _GENERIC_NAME_WORDS}


def resolve_document(question, history, docs):
    """Which indexed document does the message mean? Returns (name, None) or (None, clarification) or (None, None)."""
    names = [d[0] for d in docs]
    q = question.lower()
    m = re.search(r"\bchapter\s+(\d+)\b", q)
    if m:
        chap = [n for n in names if re.search(r"\bchapter\s+" + m.group(1) + r"\b", n.lower())]
        if len(chap) == 1:
            return chap[0], None
    m = re.search(r"\b1926\.?(\d{3})\b", q)
    if m:
        std = [n for n in names if n.lower().startswith("1926." + m.group(1))]
        if len(std) == 1:
            return std[0], None
    qtokens = set(re.findall(r"[a-z0-9]+", q))
    hits = [(len(_name_tokens(n) & qtokens), n) for n in names]
    hits = [h for h in hits if h[0] > 0]
    if hits:
        best = max(h[0] for h in hits)
        top = [n for c, n in hits if c == best]
        if len(top) == 1:
            return top[0], None
    for kind, exts in FILE_TYPES.items():
        if re.search(r"\b" + kind + r"\b", q):
            same = [n for n in names if n.lower().endswith(exts)]
            if len(same) == 1:
                return same[0], None
            if len(same) > 1:
                listing = "\n".join(f"{i}. {n}" for i, n in enumerate(same, 1))
                return None, f"I have {len(same)} {kind} files. Which one should I summarize? Tell me the name:\n" + listing
    if re.search(r"\b(?:that|this|it|the)\b", q) and re.search(r"\b(?:file|document|doc|sheet)\b", q):
        for role, content in reversed(history or []):
            if role == "user":
                continue
            found = [n for n in names if n in str(content)]
            if len(found) == 1:
                return found[0], None
            if len(found) > 1:
                listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
                return None, "Which document should I summarize? Tell me the name or number:\n" + listing
            break
    return None, None


def summary_reply(question, history):
    """'summarize that file' / 'what is X about' for one indexed document. Returns (answer, citations) or None."""
    if (len(question.split()) > 22 or not SUMMARY_INTENT_RE.search(question)
            or re.search(r"\b(?:say|says|state|states|specif\w*|require\w*|mean|means)\b", question, re.I)):
        return None
    try:
        docs = list_indexed_documents()
    except Exception:
        return None
    name, ask_back = resolve_document(question, history, docs)
    if ask_back:
        return ask_back, []
    if not name:
        return None
    client = get_client()
    chunks, offset = [], None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False)
        chunks += [pt.payload for pt in batch if (pt.payload or {}).get("source") == name]
        if offset is None:
            break
    chunks.sort(key=lambda p: (p.get("page_number") or 0))
    used, text = 0, ""
    for p in chunks:
        piece = (p.get("text") or "").strip()
        if text and len(text) + len(piece) > 6000:
            break
        text += piece[:6000] + "\n\n"
        used += 1
    if not text.strip():
        return None
    prompt = ("Summarize what this document is about in 3 to 4 plain sentences. Say what kind of document it is and its "
              "main topics. Use only the text below and do not invent details.\n\n"
              f"DOCUMENT: {name}\nTEXT:\n{text}\nSUMMARY:")
    try:
        data = request_json(f"{OLLAMA_URL}/api/generate",
            {"model": ANSWER_MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
             "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 260}}, GENERATE_TIMEOUT)
    except Exception:
        return None
    summary = re.sub(r"<think>.*?</think>", "", data.get("response") or "", flags=re.S).strip()
    if not summary:
        return None
    note = f"(Summary of {name}, based on {used} of {len(chunks)} passages.)"
    return f"{summary}\n\n{note}", [f"[1] Source: {name}, {used} of {len(chunks)} passages"]


def ask(question, history=None):
    """App entry point. history = [(role, text), ...] of earlier turns. Returns (answer, citations)."""
    LAST_PLAN.clear()
    if QUERY_UNDERSTANDING:
        summ = summary_reply(question, history)
        if summ:
            LAST_PLAN.update(kind="question", queries=[question], how="summary", why="document summary")
            return summ
        reply = meta_reply(question) or ordinal_reply(question, history)
        if reply:
            LAST_PLAN.update(kind="chat", queries=[], how="fast", why="question about the document set")
            return reply, []
    if not QUERY_UNDERSTANDING:
        LAST_PLAN.update(kind="question", queries=[question], how="off")
        return answer_with_retry(question)
    plan = understand_query(question, history)
    LAST_PLAN.update(plan)
    log.info(
        "Query route: kind=%s method=%s reason=%s query_count=%d",
        plan.get("kind"), plan.get("how"), plan.get("why"), len(plan.get("queries") or [])
    )
    if plan["kind"] == "chat":
        return plan["reply"], []
    queries = plan["queries"]
    if len(queries) == 1:
        return answer_with_retry(queries[0])
    parts, all_citations = [], []
    for i, query in enumerate(queries, 1):
        answer, citations = answer_with_retry(query)
        parts.append("**%d. %s**\n%s" % (i, query, _renumber(answer, i)))
        all_citations += [_renumber(c, i) for c in citations]
    return "\n\n".join(parts), all_citations
