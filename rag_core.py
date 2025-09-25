import os, glob, json, shutil
from typing import List, Tuple
import numpy as np
import httpx
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import pandas as pd
except Exception:
    pd = None

# ---- ENV / CONFIG ----
LM_BASE_URL = os.environ.get("LM_BASE_URL", "http://127.0.0.1:1234/v1")
LM_API_KEY  = os.environ.get("LM_API_KEY", "lm-studio")
MODEL_ID    = os.environ.get("LM_MODEL_ID", "Qwen2.5-7B-Instruct")

DATA_DIR    = os.environ.get("RAG_DATA_DIR", "./my_data")
INDEX_DIR   = os.environ.get("RAG_INDEX_DIR", "./index")

TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_CHUNK_CHARS   = int(os.environ.get("RAG_MAX_CHUNK_CHARS", "1200"))

CHUNK_SIZE    = int(os.environ.get("RAG_CHUNK_SIZE", "180"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "30"))

MAX_FILE_SIZE_BYTES = int(os.environ.get("RAG_MAX_FILE_SIZE_BYTES", str(20*1024*1024)))
MAX_CHARS_PER_FILE  = int(os.environ.get("RAG_MAX_CHARS_PER_FILE", "500000"))
MAX_PDF_PAGES       = int(os.environ.get("RAG_MAX_PDF_PAGES", "200"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("RAG_MAX_CHUNKS_PER_FILE", "1500"))
MAX_TOTAL_CHUNKS    = int(os.environ.get("RAG_MAX_TOTAL_CHUNKS", "4000"))

MAX_TOKENS           = int(os.environ.get("RAG_MAX_TOKENS", "256"))
TIMEOUT_SEC          = int(os.environ.get("RAG_TIMEOUT_SEC", "60"))

# Tắt proxy
for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"]:
    os.environ.pop(k, None)

_http = httpx.Client(transport=httpx.HTTPTransport(retries=0))
client = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY, http_client=_http)

# ---- utils ----
def safe_params(chunk_size: int, chunk_overlap: int):
    if chunk_size <= 0: chunk_size = 180
    if chunk_overlap < 0: chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    min_step = max(1, chunk_size - chunk_overlap)
    return chunk_size, chunk_overlap, min_step

def recursive_split(text: str, chunk_size=180, chunk_overlap=30, max_chunks=1500) -> List[str]:
    cs, co, step = safe_params(chunk_size, chunk_overlap)
    chunks, n, start = [], len(text), 0
    while start < n and len(chunks) < max_chunks:
        end = min(start + cs, n)
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        start += step
    return chunks

def split_csv_by_lines(csv_text: str, lines_per_chunk=100, max_chunks=1500) -> List[str]:
    lines = csv_text.splitlines()
    if not lines:
        return []
    header, rows = lines[0], lines[1:]
    chunks = []
    for i in range(0, len(rows), lines_per_chunk):
        if len(chunks) >= max_chunks: break
        block_lines = [header] + rows[i:i+lines_per_chunk]
        block = "\n".join(block_lines).strip()
        if block: chunks.append(block)
    return chunks

# ---- load data ----
def load_texts(folder: str) -> List[Tuple[str, str, str]]:
    supported = (".pdf", ".txt", ".md", ".csv", ".json", ".log", ".xlsx", ".xls")
    out = []
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        lower = path.lower()
        if not lower.endswith(supported): continue
        try:
            if os.path.getsize(path) > MAX_FILE_SIZE_BYTES:
                print(f"[skip] {os.path.basename(path)} > {MAX_FILE_SIZE_BYTES//(1024*1024)}MB")
                continue
        except Exception:
            pass
        try:
            if lower.endswith(".pdf"):
                r = PdfReader(path)
                t_parts, pages, total = [], 0, 0
                for p in r.pages:
                    pages += 1
                    if pages > MAX_PDF_PAGES: break
                    s = p.extract_text() or ""
                    if not s: continue
                    total += len(s)
                    if total > MAX_CHARS_PER_FILE:
                        t_parts.append(s[: max(0, MAX_CHARS_PER_FILE - (total-len(s)))])
                        break
                    t_parts.append(s)
                text, kind = "\n".join(t_parts), "text"
            elif lower.endswith((".xlsx", ".xls")) and pd is not None:
                xl = pd.ExcelFile(path, engine="openpyxl") if lower.endswith(".xlsx") else pd.ExcelFile(path)
                parts, total = [], 0
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    s = f"### Sheet: {sheet_name}\n" + df.to_csv(index=False)
                    total += len(s)
                    if total > MAX_CHARS_PER_FILE:
                        parts.append(s[: max(0, MAX_CHARS_PER_FILE - (total-len(s)))])
                        break
                    parts.append(s)
                text, kind = "\n\n".join(parts), "csv"
            elif lower.endswith(".csv"):
                if pd is not None:
                    try:
                        df = pd.read_csv(path, encoding="utf-8", engine="python")
                    except UnicodeDecodeError:
                        df = pd.read_csv(path, encoding="latin1", engine="python")
                    text = df.to_csv(index=False)
                else:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                text = text[:MAX_CHARS_PER_FILE]; kind = "csv"
            elif lower.endswith(".json"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text, kind = text[:MAX_CHARS_PER_FILE], "text"
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text, kind = text[:MAX_CHARS_PER_FILE], "text"
            if text.strip():
                out.append((path, text, kind))
        except Exception as e:
            print(f"[warn] Cannot read {path}: {e}")
    return out

# ---- index io ----
def _paths():
    os.makedirs(INDEX_DIR, exist_ok=True)
    return os.path.join(INDEX_DIR, "vectors.npy"), os.path.join(INDEX_DIR, "meta.json")

def ensure_index():
    vec_path, meta_path = _paths()
    if os.path.exists(vec_path) and os.path.exists(meta_path):
        vectors = np.load(vec_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return vectors, meta

    print(f"[info] Building index from data in {DATA_DIR} ...")
    docs = load_texts(DATA_DIR)
    if not docs:
        print(f"[!] No documents found in {DATA_DIR}.")
        return None, None

    chunks, meta_files = [], []
    total_chunks = 0
    for path, text, kind in docs:
        parts = split_csv_by_lines(text, lines_per_chunk=100, max_chunks=MAX_CHUNKS_PER_FILE) if kind == "csv" \
                else recursive_split(text, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_FILE)
        for c in parts:
            chunks.append(c); meta_files.append(path)
            total_chunks += 1
            if total_chunks >= MAX_TOTAL_CHUNKS:
                print(f"[info] Reached MAX_TOTAL_CHUNKS={MAX_TOTAL_CHUNKS}, stopping early.")
                break
        if total_chunks >= MAX_TOTAL_CHUNKS: break

    if not chunks:
        print("[!] No chunks to index."); return None, None

    print(f"[info] Encoding {len(chunks)} chunks ...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    vec_path, meta_path = _paths()
    np.save(vec_path, vectors)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta_files, "chunks": chunks}, f)
    return vectors, {"meta": meta_files, "chunks": chunks}

def load_index():
    vec_path, meta_path = _paths()
    if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
        return None, None
    vectors = np.load(vec_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta

def rebuild_index():
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    return ensure_index()

# ---- retrieval & QA ----
def top_k_cosine(query_vec: np.ndarray, matrix: np.ndarray, k: int):
    sims = matrix @ query_vec.T
    sims = sims.ravel()
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def retrieve(vectors, meta, query: str, k=TOP_K):
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    idxs, _ = top_k_cosine(qv, vectors, k)
    ctx_blocks, total = [], 0
    sep = "\n\n---\n\n"
    for i in idxs:
        i = int(i)
        chunk = meta["chunks"][i]
        if len(chunk) > MAX_CHUNK_CHARS: chunk = chunk[:MAX_CHUNK_CHARS]
        block = f"[{meta['meta'][i]}]\n{chunk}"
        add_len = len(block) + (len(sep) if ctx_blocks else 0)
        if total + add_len > MAX_CONTEXT_CHARS: break
        ctx_blocks.append(block); total += add_len
    return sep.join(ctx_blocks)

def ask(query: str):
    vectors, meta = load_index()
    if vectors is None:
        vectors, meta = ensure_index()
        if vectors is None:
            return "No documents to index.", []
    context = retrieve(vectors, meta, query, k=TOP_K)
    if len(context) > MAX_CONTEXT_CHARS: context = context[:MAX_CONTEXT_CHARS]
    messages = [
        {"role": "system", "content": "You are a helpful data assistant. ONLY use the information from CONTEXT. If the answer is not in the context, say you don't know. The CONTEXT may contain CSV-style tables."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer in Vietnamese and cite file paths as sources."}
    ]
    resp = client.chat.completions.create(
        model=MODEL_ID, messages=messages,
        temperature=0.2, max_tokens=MAX_TOKENS, timeout=TIMEOUT_SEC,
    )
    answer = resp.choices[0].message.content
    # extract sources
    srcs, seen = [], set()
    for line in context.splitlines():
        if line.startswith("[") and "]" in line:
            p = line[1:line.index("]")]
            if p not in seen: srcs.append(p); seen.add(p)
    return answer, srcs
