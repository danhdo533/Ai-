import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag_core import ask, ensure_index, rebuild_index, DATA_DIR

load_dotenv()

API_KEY = os.environ.get("API_KEY", "change-me-very-secret")

app = FastAPI(title="RAG Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def auth(x_api_key: str | None):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rebuild")
def api_rebuild(x_api_key: str | None = Header(None)):
    auth(x_api_key)
    vectors, meta = rebuild_index()
    n = 0 if vectors is None else vectors.shape[0]
    return {"status": "rebuilt", "chunks": n}

@app.post("/ask")
def api_ask(q: str = Form(...), x_api_key: str | None = Header(None)):
    auth(x_api_key)
    answer, sources = ask(q)
    return JSONResponse({"answer": answer, "sources": sources})

@app.post("/upload")
async def api_upload(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    auth(x_api_key)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"saved": file.filename, "path": dest}
