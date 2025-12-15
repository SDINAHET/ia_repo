from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import io
import re
import json
import requests
import numpy as np
import faiss

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text:latest"

TOP_K_CONTEXT = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 80
MAX_CONTEXT_CHARS = 3500

# ----------------------------
# APP
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: limite à ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks = []
    i = 0
    step = max(1, size - overlap)
    while i < len(text):
        chunk = text[i : i + size].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        i += step

    return chunks

# ----------------------------
# OLLAMA HELPERS
# ----------------------------
def ollama_embed(texts):
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=180,
        )
        if r.status_code != 200:
            raise RuntimeError(f"OLLAMA embeddings error {r.status_code}: {r.text}")
        payload = r.json()
        if "embedding" not in payload:
            raise RuntimeError(f"OLLAMA embeddings bad response: {payload}")
        vectors.append(payload["embedding"])

    return np.array(vectors, dtype="float32")


def ollama_chat(system: str, user: str):
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        },
        timeout=600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OLLAMA chat error {r.status_code}: {r.text}")

    payload = r.json()
    try:
        return payload["message"]["content"]
    except Exception:
        raise RuntimeError(f"OLLAMA chat bad response: {payload}")

# ----------------------------
# RAG INDEX (FAISS)
# ----------------------------
class RagIndex:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.dim = None

    def build(self, chunks):
        self.chunks = chunks
        embs = ollama_embed(chunks)  # (n, dim)
        if embs.size == 0:
            raise RuntimeError("Embeddings empty - cannot build index")
        faiss.normalize_L2(embs)
        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search(self, query, k=TOP_K_CONTEXT):
        if self.index is None:
            return []
        q = ollama_embed([query])
        faiss.normalize_L2(q)
        _, ids = self.index.search(q, k)
        results = []
        for i in ids[0]:
            if i == -1:
                continue
            results.append(self.chunks[i])
        return results

rag = RagIndex()

# ----------------------------
# PROMPT (strict JSON + evidence)
# ----------------------------
SYSTEM = """Tu es un assistant d'analyse documentaire.
RÈGLES STRICTES:
- Tu utilises UNIQUEMENT les extraits fournis (pas de connaissance externe).
- Si une info n'est pas explicitement dans les extraits: écris NOT_FOUND.
- Chaque réponse DOIT contenir une citation exacte du texte dans "evidence" (copie mot pour mot).
- Réponds UNIQUEMENT en JSON valide (aucun texte avant/après).
"""

def extract_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

# ----------------------------
# ROUTE
# ----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()

    # 1) Extraction
    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(data)
    else:
        text = data.decode("utf-8", errors="ignore").strip()

    if not text.strip():
        return {"error": "Texte vide ou illisible (PDF scanné ?). OCR nécessaire."}

    # 2) Chunking
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "Aucun chunk généré (PDF scanné / texte non extractible). OCR nécessaire."}

    # 3) Build RAG index
    rag.build(chunks)

    # 4) Retrieve context
    query = "concepts clés, définitions, dates, chiffres, causes, conséquences, points importants"
    top_chunks = rag.search(query, k=TOP_K_CONTEXT)

    context = "\n\n---\n\n".join(top_chunks)
    context = context[:MAX_CONTEXT_CHARS]

    # Helper local (3 appels plus petits => moins de timeouts)
    def ask(task: str):
        prompt = f"""EXTRAITS DU DOCUMENT:
\"\"\"
{context}
\"\"\"

TÂCHE:
{task}

RAPPEL:
- Réponds UNIQUEMENT en JSON valide (aucun texte avant/après).
- Si une info n'est pas explicitement dans les extraits: NOT_FOUND.
- Chaque item doit contenir "evidence" avec une citation exacte (copie mot pour mot).
"""
        raw = ollama_chat(SYSTEM, prompt)
        return extract_json(raw)

    # 5) Generation (split)
    try:
        summary_obj = ask('Retourne uniquement {"summary":"8-12 lignes max"}')

        open_obj = ask(
            'Retourne uniquement {"open_questions":[{"question":"...","answer":"...","evidence":"..."}, ...]} '
            'avec exactement 8 items'
        )

        mcq_obj = ask(
            'Retourne uniquement {"mcq":[{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},"correct":"A|B|C|D","explanation":"...","evidence":"..."}, ...]} '
            'avec exactement 5 items'
        )

        return {
            "summary": summary_obj.get("summary", "NOT_FOUND"),
            "open_questions": open_obj.get("open_questions", []),
            "mcq": mcq_obj.get("mcq", []),
        }

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
