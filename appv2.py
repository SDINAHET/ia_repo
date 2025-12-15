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

# JSON robustness
DEBUG_MODEL_OUTPUT = False
JSON_RETRIES = 2  # nombre de retries si JSON invalide

# Ollama timeouts / ctx
OLLAMA_EMBED_TIMEOUT = 180
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.2

# ----------------------------
# APP
# ----------------------------
app = FastAPI()

# Mets ton front local ici si tu veux CORS strict.
# Sinon remets ["*"] pour tester.
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # pour tests rapides: ["*"]
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
        chunk = text[i: i + size].strip()
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
            timeout=OLLAMA_EMBED_TIMEOUT,
        )
        if r.status_code != 200:
            raise RuntimeError(f"OLLAMA embeddings error {r.status_code}: {r.text}")
        payload = r.json()
        if "embedding" not in payload:
            raise RuntimeError(f"OLLAMA embeddings bad response: {payload}")
        vectors.append(payload["embedding"])
    return np.array(vectors, dtype="float32")


def ollama_chat(system: str, user: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": OLLAMA_TEMPERATURE, "num_ctx": OLLAMA_NUM_CTX},
        },
        timeout=OLLAMA_CHAT_TIMEOUT,
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
- Si une info n'est pas explicitement dans les extraits: écris "NOT_FOUND" (string JSON avec guillemets).
- Chaque item doit contenir une citation exacte du texte dans "evidence" (copie mot pour mot).
- Tu réponds UNIQUEMENT avec un JSON STRICTEMENT VALIDE:
  - aucun texte avant/après
  - aucune virgule finale
  - pas de commentaires
  - clés/valeurs entre guillemets
"""

def repair_common_json(candidate: str) -> str:
    """
    Répare les erreurs JSON fréquentes des LLM.
    Ex: NOT_FOUND (token) -> "NOT_FOUND" (string)
    """
    candidate = re.sub(
        r'(:\s*)NOT_FOUND(\s*[,\n\r}])',
        r'\1"NOT_FOUND"\2',
        candidate
    )
    return candidate


def extract_json_strict(raw: str):
    """
    Extraction JSON robuste:
    - prend le 1er '{' et le dernier '}'
    - répare NOT_FOUND
    - json.loads strict
    """
    if raw is None:
        raise ValueError("Empty model output")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    candidate = raw[start:end + 1].strip()
    candidate = repair_common_json(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e}\n---\n{candidate}\n---") from e


def ask_with_retry(context: str, task: str):
    """
    Appelle le modèle et force un JSON valide (avec retries).
    """
    prompt = f"""EXTRAITS DU DOCUMENT:
\"\"\"
{context}
\"\"\"

TÂCHE:
{task}

RAPPEL:
- Réponds UNIQUEMENT avec un JSON STRICTEMENT VALIDE (aucun texte avant/après).
- Si une info n'est pas explicitement dans les extraits: écris "NOT_FOUND".
- Chaque item doit contenir "evidence" avec une citation exacte (copie mot pour mot).
"""

    last_err = None
    for _ in range(JSON_RETRIES + 1):
        raw = ollama_chat(SYSTEM, prompt)

        if DEBUG_MODEL_OUTPUT:
            print("\n========== RAW MODEL OUTPUT ==========\n")
            print(raw)
            print("\n======================================\n")

        try:
            return extract_json_strict(raw)
        except Exception as e:
            last_err = e
            prompt += "\nIMPORTANT: Ton dernier JSON était invalide. Réponds à nouveau en JSON strict.\n"

    raise last_err

# ----------------------------
# ROUTE
# ----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}

    data = await file.read()
    if not data:
        return {"error": "Empty file"}

    filename = (file.filename or "").lower()

    # 1) Extraction
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    else:
        text = data.decode("utf-8", errors="ignore").strip()

    if not text.strip():
        return {"error": "Texte vide ou illisible (PDF scanné ?). OCR nécessaire."}

    # 2) Chunking
    chunks = chunk_text(text)
    if not chunks:
        return {"error": "Aucun chunk généré (PDF scanné / texte non extractible). OCR nécessaire."}

    # 3) Build index
    try:
        rag.build(chunks)
    except Exception as e:
        return {"error": f"RAG/FAISS build failed: {str(e)}"}

    # 4) Retrieve context
    query = "concepts clés, définitions, dates, chiffres, causes, conséquences, points importants"
    top_chunks = rag.search(query, k=TOP_K_CONTEXT)

    context = "\n\n---\n\n".join(top_chunks)
    context = context[:MAX_CONTEXT_CHARS]

    # 5) Generation (split en 3 prompts)
    try:
        summary_obj = ask_with_retry(
            context,
            'Retourne uniquement {"summary":[{"item":"...","evidence":"..."}, ...]} avec 8 à 12 items.'
        )

        open_obj = ask_with_retry(
            context,
            'Retourne uniquement {"open_questions":[{"question":"...","answer":"...","evidence":"..."}, ...]} avec exactement 8 items.'
        )

        mcq_obj = ask_with_retry(
            context,
            'Retourne uniquement {"mcq":[{"question":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},"correct":"A|B|C|D","explanation":"...","evidence":"..."}, ...]} avec exactement 5 items.'
        )

        return {
            "summary": summary_obj.get("summary", []),
            "open_questions": open_obj.get("open_questions", []),
            "mcq": mcq_obj.get("mcq", []),
        }

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
