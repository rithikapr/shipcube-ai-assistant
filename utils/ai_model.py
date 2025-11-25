# utils/ai_model.py
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import ollama 

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"

QNA_JSON_PATH = DATA_DIR / "qna.json"
QNA_EMB_PATH = DATA_DIR / "qna.embeddings.npy"
QNA_INDEX_PATH = DATA_DIR / "qna.faiss.index"
#DOC_KB_DIR = DATA_DIR / "doc_kb"          # (you can delete this if unused)
GLOBAL_KB_DIR = DATA_DIR / "global_kb"    # NEW unified KB from build_global_kb.py

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------
# Global lazy-loaded objects
# ---------------------------------------------------------------------
_doc_vectordb: Optional[LCFAISS] = None   # (legacy, can remove)
_global_vectordb: Optional[LCFAISS] = None  # NEW

_model: Optional[SentenceTransformer] = None
_qna_index = None
_qna_embeddings = None
QNA: List[Dict] = []  # exported for other modules


# ---------------------------------------------------------------------
# Helpers to load things lazily
# ---------------------------------------------------------------------
# at top of file, _ensure_qna_loaded already exists

def get_top_faq(limit: int = 6):
    """Return top N FAQ entries from qna.json."""
    _ensure_qna_loaded()
    if not isinstance(QNA, list):
        return []
    return QNA[:limit]

def _ensure_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _ensure_qna_loaded():
    """Load QNA JSON + FAISS index only once."""
    global QNA, _qna_index, _qna_embeddings

    if QNA and _qna_index is not None and _qna_embeddings is not None:
        return

    if not QNA_JSON_PATH.exists():
        print("[QNA] qna.json not found at", QNA_JSON_PATH)
        QNA = []
        return

    # Load Q&A rows
    with QNA_JSON_PATH.open("r", encoding="utf-8") as f:
        QNA = json.load(f)

    if QNA_EMB_PATH.exists() and QNA_INDEX_PATH.exists():
        _qna_embeddings = np.load(str(QNA_EMB_PATH)).astype("float32")
        _qna_index = faiss.read_index(str(QNA_INDEX_PATH))
    else:
        print("[QNA] embeddings/index files not found; retrieval will be disabled.")
        _qna_index = None
        _qna_embeddings = None


# def _ensure_doc_kb_loaded():
#     """
#     Load the LangChain FAISS index built from your PDFs (doc_kb).
#     Requires build_doc_kb_langchain.py to have been run.
#     """
#     global _doc_vectordb
#     if _doc_vectordb is not None:
#         return

#     if not DOC_KB_DIR.exists():
#         print("[doc_kb] directory does not exist:", DOC_KB_DIR)
#         _doc_vectordb = None
#         return

#     try:
#         embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#         _doc_vectordb = LCFAISS.load_local(
#             str(DOC_KB_DIR),
#             embeddings,
#             allow_dangerous_deserialization=True,
#         )
#         print("[doc_kb] loaded FAISS index from", DOC_KB_DIR)
#     except Exception as e:
#         print("[doc_kb] error loading index:", e)
#         _doc_vectordb = None


# ---------------------------------------------------------------------
# Public helper: sidebar FAQs by tag
# ---------------------------------------------------------------------

def get_items_for_tag(tag: str, limit: int = 10) -> List[Dict]:
    _ensure_qna_loaded()
    if not QNA:
        return []
    tag_lower = (tag or "").lower()
    items = [
        q for q in QNA
        if (q.get("tag") or q.get("context") or "").lower() == tag_lower
    ]
    return items[:limit]


# ---------------------------------------------------------------------
# Retrieval over QNA FAISS index
# ---------------------------------------------------------------------

def _search_qna(query: str, top_k: int = 5, score_threshold: float = 0.2) -> List[Dict]:
    """
    Search the original curated Q&A FAISS index.
    Returns list of dicts with question, answer, score, source.
    """
    _ensure_qna_loaded()
    if _qna_index is None or not QNA:
        return []

    model = _ensure_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # FAISS IndexFlatIP with cosine similarity (because we normalized)
    D, I = _qna_index.search(q_vec, top_k)
    scores = D[0]
    idxs = I[0]

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        if score < score_threshold:
            continue
        if idx >= len(QNA):
            continue
        item = QNA[idx]
        results.append(
            {
                "question": item.get("question"),
                "answer": item.get("answer"),
                "score": float(score),
                "source": "retrieval_global",  # keep your old label
            }
        )
    return results


# ---------------------------------------------------------------------
# Retrieval over PDF doc_kb (LangChain FAISS)
# ---------------------------------------------------------------------

# def _search_doc_kb(query: str, top_k: int = 5) -> List[Dict]:
#     """
#     Search the PDF knowledge base built in data/doc_kb.
#     Uses LangChain FAISS similarity_search_with_score.
#     """
#     _ensure_doc_kb_loaded()
#     if _doc_vectordb is None:
#         return []

#     # LangChain returns (Document, distance); lower distance = more similar.
#     docs_scores = _doc_vectordb.similarity_search_with_score(query, k=top_k)

#     results = []
#     for doc, dist in docs_scores:
#         # Convert distance to a rough similarity score in [0,1]
#         sim = 1.0 / (1.0 + float(dist))

#         # Heuristic threshold – adjust if you want more/less aggressive
#         if sim < 0.25:
#             continue

#         src_file = doc.metadata.get("source_file") or doc.metadata.get("source")
#         results.append(
#             {
#                 "question": None,
#                 "answer": doc.page_content.strip(),
#                 "score": float(sim),
#                 "source": f"doc_kb:{src_file}" if src_file else "doc_kb",
#             }
#         )
#     return results

def _ensure_global_kb_loaded():
    """
    Load the unified LangChain FAISS index built by build_global_kb.py
    from data/global_kb (PDF chunks + FAQ docs).
    """
    global _global_vectordb
    if _global_vectordb is not None:
        return

    if not GLOBAL_KB_DIR.exists():
        print("[global_kb] directory does not exist:", GLOBAL_KB_DIR)
        _global_vectordb = None
        return

    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        _global_vectordb = LCFAISS.load_local(
            str(GLOBAL_KB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[global_kb] loaded FAISS index from", GLOBAL_KB_DIR)
    except Exception as e:
        print("[global_kb] error loading index:", e)
        _global_vectordb = None

def _search_global_kb(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search the unified global_kb index (PDF + FAQ).
    Returns list of dicts with question, answer/text, score, and source.
    """
    _ensure_global_kb_loaded()
    if _global_vectordb is None:
        return []

    docs_scores = _global_vectordb.similarity_search_with_score(query, k=top_k)
    results = []

    for doc, dist in docs_scores:
        # Convert distance to a rough similarity score in [0,1]
        sim = 1.0 / (1.0 + float(dist))

        meta = doc.metadata or {}
        src = meta.get("source", "global_kb")
        faq_question = meta.get("question")

        results.append(
            {
                "question": faq_question,                  # for FAQ docs; None for PDFs
                "answer": doc.page_content.strip(),        # raw chunk text
                "score": float(sim),
                "source": src,                             # "faq" or PDF path
                "metadata": meta,
            }
        )
    return results

# ---------------------------------------------------------------------
# MAIN: combined retrieval
# ---------------------------------------------------------------------

# def get_retrieval_answer(
#     query: str,
#     top_k: int = 5,
#     score_threshold: float = 0.2,
#     tag: Optional[str] = None,
# ) -> List[Dict]:
#     """
#     Combined retrieval:
#     - For brand-specific questions (mentioning ShipCube), prefer curated Q&A.
#     - For general supply-chain / 3PL questions, consult both Q&A and doc_kb.
#     - Returns list of hits sorted by score desc.
#     """
#     q = (query or "").strip()
#     if not q:
#         return []

#     lower_q = q.lower()
#     brand_query = "shipcube" in lower_q or "ship cube" in lower_q

#     results: List[Dict] = []

#     # 1) General supply-chain: query doc_kb first
#     if not brand_query:
#         results.extend(_search_doc_kb(q, top_k=top_k))

#     # 2) Always query curated Q&A as well
#     results.extend(_search_qna(q, top_k=top_k, score_threshold=score_threshold))

#     if not results:
#         return []

#     # 3) Sort by score descending
#     results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
#     return results

def get_retrieval_answer(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.2,
    tag: Optional[str] = None,
) -> List[Dict]:
    """
    Combined retrieval on the unified global_kb index (PDF + FAQ).

    Rules:
    - If the question clearly mentions ShipCube (brand_query):
        → prefer FAQ docs (source == "faq") if any exist.
    - Otherwise:
        → allow both PDFs and FAQ docs, but still keep a score threshold.
    """
    q = (query or "").strip()
    if not q:
        return []

    lower_q = q.lower()
    brand_query = "shipcube" in lower_q or "ship cube" in lower_q

    # 1) Get raw hits from the unified KB
    raw_hits = _search_global_kb(q, top_k=top_k * 3)  # query a bit more, we'll filter
    if not raw_hits:
        return []

    # 2) Apply some logic based on brand_query
    faq_hits = [h for h in raw_hits if h.get("source") == "faq"]
    pdf_hits = [h for h in raw_hits if h.get("source") != "faq"]

    results: List[Dict] = []

    if brand_query:
        # For ShipCube-specific questions, strongly prefer FAQ entries.
        if faq_hits:
            results = faq_hits
        else:
            # If no FAQ hits at all, fall back to whatever we found (PDFs)
            results = raw_hits
    else:
        # General supply-chain / logistics questions → use everything
        results = raw_hits

    # 3) Apply a score threshold
    filtered = [r for r in results if r.get("score", 0.0) >= score_threshold]
    if not filtered:
        return []

    # 4) Sort by score descending and cut to top_k
    filtered.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return filtered[:top_k]

# ---------------------------------------------------------------------
# Fallback LLM (you already had a simple stub)
# ---------------------------------------------------------------------

def get_ai_response(prompt: str) -> str:
    """
    Generate a reply using a local LLaMA model via Ollama.
    This is used both for:
      - summarising PDF doc_kb chunks (RAG)
      - pure generative fallback answers
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are ShipCube AI, a helpful logistics and 3PL assistant. "
                "Answer clearly and concisely. If you are given a 'Context:' "
                "section, you MUST base your answer only on that context."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        res = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=False,
        )
        content = res.get("message", {}).get("content", "").strip()
        if not content:
            return "I couldn't understand that — could you please rephrase or ask a more specific question?"
        return content
    except Exception as e:
        # Log the error so you can see it in the Flask console
        print("[ollama] generation error:", e)
        return "Sorry, the model couldn't produce a useful answer right now."
