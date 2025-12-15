from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import io
import re
import json
import random
import requests
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Any

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text:latest"

# Retrieval / chunking
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 120

# Stable settings (avoid timeouts)
OLLAMA_NUM_CTX = 4096
MAX_CONTEXT_CHARS = 3200
TOP_K_CONTEXT = 6
TOP_K_PER_ITEM = 4

# Robust JSON
DEBUG_MODEL_OUTPUT = False
JSON_RETRIES = 2

# Ollama timeouts
OLLAMA_EMBED_TIMEOUT = 180
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_TEMPERATURE = 0.2
OLLAMA_NUM_PREDICT = 500  # limit generation length

# Optional web search (disabled by default)
WEB_SEARCH_ENABLED = False
WEB_SEARCH_PROVIDER = "tavily"
WEB_SEARCH_API_KEY = ""
WEB_SEARCH_TIMEOUT = 20
WEB_MAX_SNIPPETS = 5

# ----------------------------
# APP
# ----------------------------
app = FastAPI()

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# PDF EXTRACTION (pages)
# ----------------------------
def extract_pages_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ----------------------------
# CHUNKING WITH METADATA
# ----------------------------
@dataclass
class Chunk:
    text: str
    page: int
    start: int
    end: int

def chunk_pages(pages: List[str], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    step = max(1, size - overlap)

    for pageno, raw in enumerate(pages, start=1):
        text = normalize_text(raw)
        if len(text) < MIN_CHUNK_CHARS:
            continue

        i = 0
        while i < len(text):
            piece = text[i:i + size].strip()
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(text=piece, page=pageno, start=i, end=min(i + size, len(text))))
            i += step

    return chunks

# ----------------------------
# OLLAMA HELPERS
# ----------------------------
def ollama_embed(texts: List[str]) -> np.ndarray:
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=OLLAMA_EMBED_TIMEOUT,
        )
        if r.status_code != 200:
            raise RuntimeError(f"OLLAMA_EMBED_ERROR {r.status_code}: {r.text}")
        payload = r.json()
        if "embedding" not in payload:
            raise RuntimeError(f"OLLAMA_EMBED_BAD_RESPONSE: {payload}")
        vectors.append(payload["embedding"])
    return np.array(vectors, dtype="float32")

def ollama_chat(system: str, user: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
            },
            timeout=OLLAMA_CHAT_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        return payload["message"]["content"]
    except requests.exceptions.ReadTimeout:
        raise RuntimeError("OLLAMA_TIMEOUT")
    except Exception as e:
        raise RuntimeError(f"OLLAMA_CHAT_ERROR: {e}")

# ----------------------------
# RAG INDEX (FAISS)
# ----------------------------
class RagIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.index = None
        self.dim = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        embs = ollama_embed([c.text for c in chunks])
        if embs.size == 0:
            raise RuntimeError("Embeddings empty - cannot build index")

        faiss.normalize_L2(embs)
        self.dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search(self, query: str, k: int) -> List[Chunk]:
        if self.index is None:
            return []
        q = ollama_embed([query])
        faiss.normalize_L2(q)
        _, ids = self.index.search(q, k)
        out = []
        for i in ids[0]:
            if i == -1:
                continue
            out.append(self.chunks[int(i)])
        return out

rag = RagIndex()

def format_context(chunks: List[Chunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for c in chunks:
        block = f"[PAGE {c.page}] {c.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)

# ----------------------------
# JSON STRICT HELPERS
# ----------------------------
SYSTEM = """Tu es un assistant d'analyse documentaire.

RÈGLES STRICTES:
- Tu utilises UNIQUEMENT les extraits fournis.
- Si une info n'est pas explicitement dans les extraits: "NOT_FOUND".
- Chaque item doit contenir "evidence" avec une citation EXACTE copiée mot pour mot depuis les extraits.
- Tu réponds UNIQUEMENT avec un JSON STRICTEMENT VALIDE (aucun texte avant/après).
- Quand possible, inclure "page" (numéro de page) en plus de "evidence".
"""

def repair_common_json(candidate: str) -> str:
    return re.sub(r'(:\s*)NOT_FOUND(\s*[,\n\r}])', r'\1"NOT_FOUND"\2', candidate)

def extract_json_strict(raw: str) -> Dict[str, Any]:
    if raw is None:
        raise ValueError("Empty model output")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    candidate = raw[start:end + 1].strip()
    candidate = repair_common_json(candidate)
    return json.loads(candidate)

def ask_with_retry(context: str, task: str) -> Dict[str, Any]:
    context = (context or "")[:MAX_CONTEXT_CHARS]

    prompt = f"""EXTRAITS DU DOCUMENT:
\"\"\"\n{context}\n\"\"\"

TÂCHE:
{task}

RAPPEL:
- JSON strict uniquement
- evidence = citation exacte mot pour mot
- si absent => "NOT_FOUND"
"""
    last_err = None
    for _ in range(JSON_RETRIES + 1):
        raw = ollama_chat(SYSTEM, prompt)
        if DEBUG_MODEL_OUTPUT:
            print(raw)
        try:
            return extract_json_strict(raw)
        except Exception as e:
            last_err = e
            prompt += "\nIMPORTANT: Ton dernier JSON était invalide. Réponds à nouveau en JSON strict.\n"
    raise last_err

# ----------------------------
# QCM PRO: shuffle A/B/C/D
# ----------------------------
def shuffle_mcq(mcq_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in mcq_items:
        correct = item.get("correct_answer", "NOT_FOUND")
        distractors = item.get("distractors", [])
        distractors = [d for d in distractors if isinstance(d, str)]

        choices = [correct] + distractors
        while len(choices) < 4:
            choices.append("NOT_FOUND")
        choices = choices[:4]

        random.shuffle(choices)
        letters = ["A", "B", "C", "D"]
        mapped = {letters[i]: choices[i] for i in range(4)}

        correct_letter = "A"
        for k, v in mapped.items():
            if v == correct:
                correct_letter = k
                break

        out.append({
            "question": item.get("question", "NOT_FOUND"),
            "choices": mapped,
            "correct": correct_letter,
            "explanation": item.get("explanation", "NOT_FOUND"),
            "evidence": item.get("evidence", "NOT_FOUND"),
            "page": item.get("page", "NOT_FOUND"),
        })
    return out

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
        pages = extract_pages_from_pdf(data)
        if not any(p.strip() for p in pages):
            return {"error": "Texte vide ou illisible (PDF scanné ?). OCR nécessaire."}
    else:
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return {"error": "Texte vide"}
        pages = [text]

    # 2) Chunking + index
    chunks = chunk_pages(pages)
    if not chunks:
        return {"error": "Aucun chunk généré (PDF scanné / texte non extractible). OCR nécessaire."}

    try:
        rag.build(chunks)
    except Exception as e:
        return {"error": f"RAG/FAISS build failed: {str(e)}"}

    try:
        # 3) Global context
        global_query = "points clés, définitions, règles, obligations, droits, dates, conditions, procédures"
        global_chunks = rag.search(global_query, k=TOP_K_CONTEXT)
        global_context = format_context(global_chunks)

        # 4) Summary (8-12)
        summary_obj = ask_with_retry(
            global_context,
            'Retourne uniquement {"summary":[{"item":"...","evidence":"...","page":1}, ...]} avec 8 à 12 items.'
        )

        # 5) Generate questions
        q_obj = ask_with_retry(
            global_context,
            'Retourne uniquement {"questions":[{"question":"..."}, ...]} avec exactement 8 questions ouvertes pertinentes (style examen).'
        )
        questions = [q.get("question", "NOT_FOUND") for q in q_obj.get("questions", [])][:8]
        while len(questions) < 8:
            questions.append("NOT_FOUND")

        # 6) Answer each question with targeted RAG
        open_questions = []
        for q in questions:
            q_chunks = rag.search(q, k=TOP_K_PER_ITEM)
            q_context = format_context(q_chunks)
            ans = ask_with_retry(
                q_context,
                f'Retourne uniquement {{"question":"{q}","answer":"...","evidence":"...","page":1}}.'
            )
            open_questions.append(ans)

        # 7) QCM: generate 1 by 1 (stable)
        mcq_context = format_context(
            rag.search("questions type examen, règles, obligations, droits, définitions, conditions", k=TOP_K_CONTEXT)
        )

        mcq_items = []
        for _ in range(5):
            one = ask_with_retry(
                mcq_context,
                'Retourne uniquement {"question":"...","correct_answer":"...","distractors":["...","...","..."],'
                '"explanation":"...","evidence":"...","page":1} '
                'Question courte, distracteurs plausibles, style examen.'
            )
            mcq_items.append(one)

        mcq = shuffle_mcq(mcq_items)

        return {
            "summary": summary_obj.get("summary", []),
            "open_questions": open_questions,
            "mcq": mcq,
            "debug": {
                "web_search_enabled": WEB_SEARCH_ENABLED,
                "chat_model": CHAT_MODEL,
                "embed_model": EMBED_MODEL,
                "top_k_global": TOP_K_CONTEXT,
                "top_k_per_item": TOP_K_PER_ITEM,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            }
        }

    except RuntimeError as e:
        if str(e) == "OLLAMA_TIMEOUT":
            return {"error": "Ollama a dépassé le timeout pendant la génération. Réduis le contexte ou relance."}
        return {"error": str(e)}

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
