# utils/ai_model.py
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

QNA_JSON_PATH = DATA_DIR / "qna.json"
GLOBAL_KB_DIR = DATA_DIR / "global_kb"    # unified KB (PDF chunks + FAQ)

# MUST match embed model used when building the KB (build_global_kb.py)
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
_global_vectordb: Optional[LCFAISS] = None
_model: Optional[SentenceTransformer] = None
_llm: Optional[ChatGoogleGenerativeAI] = None
QNA: List[Dict] = []

# cached FAQ embeddings (numpy array) and parallel question list
_QNA_EMBEDDINGS: Optional[np.ndarray] = None
_QNA_QUESTIONS: List[str] = []

# ---------------------------------------------------------------------
# Lazy loaders and initialisation
# ---------------------------------------------------------------------
def _ensure_qna_loaded():
    global QNA
    if QNA:
        return
    if not QNA_JSON_PATH.exists():
        print("[ai_model] qna.json not found at", QNA_JSON_PATH)
        QNA = []
        return
    with QNA_JSON_PATH.open("r", encoding="utf-8") as fh:
        QNA = json.load(fh)

def _ensure_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _ensure_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.2,
            max_output_tokens=512,
        )
    return _llm

def _ensure_qna_embeddings():
    """
    Encode FAQ questions once into normalized embeddings using the SAME model as the rest
    of the system. Stored as numpy array for fast dot-product (cosine) comparisons.
    """
    global _QNA_EMBEDDINGS, _QNA_QUESTIONS
    _ensure_qna_loaded()
    if _QNA_EMBEDDINGS is not None:
        return
    if not QNA:
        _QNA_EMBEDDINGS = None
        _QNA_QUESTIONS = []
        return
    questions = []
    for item in QNA:
        q = (item.get("question") or item.get("Question") or "").strip()
        questions.append(q)
    model = _ensure_model()
    embs = model.encode(questions, normalize_embeddings=True)
    _QNA_EMBEDDINGS = np.array(embs, dtype=float)
    _QNA_QUESTIONS = questions

def _ensure_global_kb_loaded():
    """
    Load the unified FAISS built by build_global_kb.py.
    """
    global _global_vectordb
    if _global_vectordb is not None:
        return
    if not GLOBAL_KB_DIR.exists():
        print("[ai_model] global_kb directory not found:", GLOBAL_KB_DIR)
        _global_vectordb = None
        return
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        _global_vectordb = LCFAISS.load_local(
            str(GLOBAL_KB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[ai_model] loaded FAISS index from", GLOBAL_KB_DIR)
    except Exception as e:
        print("[ai_model] error loading FAISS index:", e)
        _global_vectordb = None

# ---------------------------------------------------------------------
# FAQ helpers
# ---------------------------------------------------------------------
def get_top_faq(limit: int = 6) -> List[Dict]:
    _ensure_qna_loaded()
    if not isinstance(QNA, list):
        return []
    return QNA[:limit]

def get_items_for_tag(tag: str, limit: int = 10) -> List[Dict]:
    _ensure_qna_loaded()
    if not QNA:
        return []
    tag_lower = (tag or "").lower()
    items = []
    for q in QNA:
        dept = (q.get("Departments") or "").lower()
        if not dept:
            dept = (q.get("tag") or q.get("context") or "").lower()
        if dept == tag_lower:
            items.append(q)
    return items[:limit]

# ---------------------------------------------------------------------
# Semantic FAQ match (primary)
# ---------------------------------------------------------------------
def semantic_faq_match(query: str, top_k: int = 3, threshold: float = 0.66) -> Optional[Tuple[Dict, float]]:
    """
    Return best FAQ item + similarity if above threshold. Similarity is cosine in [-1,1].
    """
    if not query:
        return None
    _ensure_qna_embeddings()
    if _QNA_EMBEDDINGS is None or len(_QNA_EMBEDDINGS) == 0:
        return None

    model = _ensure_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=float)[0]

    sims = np.dot(_QNA_EMBEDDINGS, q_emb)  # dot product since normalized => cosine
    top_idx = np.argsort(-sims)[:top_k]
    best_i = int(top_idx[0])
    best_score = float(sims[best_i])
    if best_score >= threshold:
        return QNA[best_i], best_score
    return None

# ---------------------------------------------------------------------
# Lexical fallback (rare): very conservative lexical heuristic
# ---------------------------------------------------------------------
_LEXICAL_TOKEN_RE = re.compile(r"[^a-z0-9\s]", re.I)
def lexical_faq_match(query: str, min_coverage: float = 0.6) -> Optional[Dict]:
    """
    Lightweight lexical fallback: compute token overlap coverage. Returns FAQ item if coverage
    of user's tokens within FAQ question >= min_coverage (conservative).
    This is a fallback only — semantic matching is preferred.
    """
    _ensure_qna_loaded()
    q_raw = (query or "").strip().lower()
    if not q_raw or not QNA:
        return None
    q_norm = _LEXICAL_TOKEN_RE.sub(" ", q_raw)
    user_tokens = [t for t in q_norm.split() if len(t) > 1]
    if not user_tokens:
        return None
    user_set = set(user_tokens)

    best = None
    best_cov = 0.0
    for item in QNA:
        faq_q = (item.get("question") or item.get("Question") or "").strip().lower()
        if not faq_q:
            continue
        faq_norm = _LEXICAL_TOKEN_RE.sub(" ", faq_q)
        faq_tokens = [t for t in faq_norm.split() if len(t) > 1]
        if not faq_tokens:
            continue
        inter = user_set.intersection(set(faq_tokens))
        coverage = len(inter) / len(user_set)
        if coverage > best_cov:
            best_cov = coverage
            best = item
    if best and best_cov >= min_coverage:
        return best
    return None

# ---------------------------------------------------------------------
# Unified KB retrieval helpers
# ---------------------------------------------------------------------
def _search_global_kb(query: str, top_k: int = 5) -> List[Dict]:
    _ensure_global_kb_loaded()
    if _global_vectordb is None:
        return []
    docs_scores = _global_vectordb.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, dist in docs_scores:
        sim = 1.0 / (1.0 + float(dist))
        meta = doc.metadata or {}
        src = meta.get("source", "global_kb")
        faq_question = meta.get("question") or meta.get("Question")
        results.append({
            "question": faq_question,
            "answer": doc.page_content.strip(),
            "score": float(sim),
            "source": src,
            "metadata": meta,
        })
    return results

def get_retrieval_answer(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.18,
    tag: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve top-K relevant docs from the unified KB.
    - For brand-specific queries (contains 'shipcube'), prefer FAQ docs if any.
    - Apply score threshold and return sorted top_k.
    """
    q = (query or "").strip()
    if not q:
        return []
    lower_q = q.lower()
    brand_query = "shipcube" in lower_q or "ship cube" in lower_q

    raw_hits = _search_global_kb(q, top_k=top_k * 3)
    if not raw_hits:
        return []

    faq_hits = [h for h in raw_hits if h.get("source") == "faq"]
    if brand_query and faq_hits:
        candidates = faq_hits
    else:
        candidates = raw_hits

    filtered = [r for r in candidates if r.get("score", 0.0) >= score_threshold]
    if not filtered:
        return []
    filtered.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return filtered[:top_k]

# ---------------------------------------------------------------------
# LLM / Gemini wrapper and summariser
# ---------------------------------------------------------------------
def get_ai_response(prompt: str) -> str:
    llm = _ensure_llm()
    system_msg = (
        "You are ShipCube AI, a helpful logistics and 3PL assistant. "
        "Answer concisely and base responses on provided context when present."
    )
    full_prompt = system_msg + "\n\n" + prompt
    try:
        res = llm.invoke(full_prompt)
        content = (getattr(res, "content", "") or "").strip()
        if not content:
            return "I couldn't understand that — could you please rephrase?"
        return content
    except Exception as e:
        print("[ai_model] LLM error:", e)
        return "Sorry, I couldn't produce an answer right now."

def summarise_context(context: str) -> str:
    """
    Summarise a long context into ~100 words for multi-turn RAG. If summariser fails, returns empty string.
    """
    if not context:
        return ""
    llm = _ensure_llm()
    system_msg = (
        "You are an expert summarizer. Summarize the given chat history in <= 100 words. "
        "Do not add new information; only compress what's present."
    )
    full_prompt = system_msg + "\n\n" + context
    try:
        res = llm.invoke(full_prompt)
        content = (getattr(res, "content", "") or "").strip()
        return content
    except Exception as e:
        print("[ai_model] summariser error:", e)
        return ""

# ---------------------------------------------------------------------
# High-level: generate answer from retrieval (RAG)
# ---------------------------------------------------------------------
def generate_answer_from_retrieval(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.18,
    max_context_tokens: int = 1200,
    concise_word_limit: int = 70,
    conversation_summary: str = "",
) -> Dict:
    """
    End-to-end RAG:
    0) Semantic FAQ match (embedding).
    1) Lexical FAQ fallback.
    2) Retrieval from unified KB -> build context -> call LLM instructing it to only use context.
    Returns dict: {query, answer, sources, used_context}
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "answer": "Please provide a question.", "sources": [], "used_context": ""}

    # 0) semantic FAQ
    sem = semantic_faq_match(query, top_k=3, threshold=0.67)
    if sem:
        item, score = sem
        q_text = item.get("Question") or item.get("question") or ""
        a_text = item.get("Answer") or item.get("answer") or ""
        raw = f"Q: {q_text}\nA: {a_text}"
        return {
            "query": query,
            "answer": a_text,
            "sources": [{"id": 1, "source": "faq_semantic", "score": float(score)}],
            "used_context": raw,
        }

    # 1) lexical fallback (conservative)
    lex = lexical_faq_match(query, min_coverage=0.65)
    if lex:
        q_text = lex.get("Question") or lex.get("question") or ""
        a_text = lex.get("Answer") or lex.get("answer") or ""
        raw = f"Q: {q_text}\nA: {a_text}"
        return {
            "query": query,
            "answer": a_text,
            "sources": [{"id": 1, "source": "faq_lexical", "score": 1.0}],
            "used_context": raw,
        }

    # 2) retrieval: incorporate conversation_summary (if provided) to help multi-turn
    retrieval_query = (conversation_summary + "\n\n" + query).strip() if conversation_summary else query
    hits = get_retrieval_answer(retrieval_query, top_k=top_k, score_threshold=score_threshold)
    if not hits:
        # fallback: ask LLM to answer without context
        prompt_noctx = (
            "Context:\n\n"
            "None\n\n"
            f"User question:\n{query}\n\n"
            "Instructions: If you don't know, say 'I don't know'. Keep the answer concise."
        )
        ans = get_ai_response(prompt_noctx)
        return {"query": query, "answer": ans, "sources": [], "used_context": ""}

    # Build context from top hits (truncate to limit)
    sources = []
    context_parts = []
    chars_used = 0
    max_chars = max_context_tokens * 4  # approx char tokenization
    for i, h in enumerate(hits, start=1):
        snippet = (h.get("answer") or "").strip()
        header = f"[{i}] Source: {h.get('source','unknown')} (score={h.get('score',0):.3f})\n"
        block = header + snippet + "\n\n"
        if chars_used + len(block) > max_chars:
            remaining = max(0, max_chars - chars_used - 10)
            if remaining <= 0:
                break
            snippet_short = snippet[:remaining].rsplit(" ", 1)[0]
            block = header + snippet_short + "...\n\n"
            context_parts.append(block)
            chars_used += len(block)
            sources.append({"id": i, "source": h.get("source"), "score": float(h.get("score", 0.0)), "snippet": snippet_short})
            break
        context_parts.append(block)
        chars_used += len(block)
        sources.append({"id": i, "source": h.get("source"), "score": float(h.get("score", 0.0)), "snippet": snippet[:300]})

    context_text = "".join(context_parts).strip()

    # Compose RAG prompt instructing the LLM to use ONLY the context
    prompt = (
        "Context:\n\n"
        f"{context_text}\n\n"
        "User question:\n"
        f"{query}\n\n"
        "Instructions:\n"
        "- Base your answer ONLY on the Context above.\n"
        f"- Keep the answer concise (~{concise_word_limit} words or less).\n"
        "- If the context doesn't contain enough info to answer, say 'I don't know' and list which sources (by id) are most relevant.\n"
        "- Provide a 1–2 sentence actionable answer.\n"
    )

    answer = get_ai_response(prompt)

    # try to detect citations like [1] in answer and map to sources
    cited_ids = re.findall(r"\[(\d+)\]", answer)
    cited_ids = sorted({int(x) for x in cited_ids if x.isdigit()})
    cited_sources = [s for s in sources if s["id"] in cited_ids]
    if not cited_sources:
        cited_sources = sources[: min(2, len(sources))]

    return {
        "query": query,
        "answer": answer,
        "sources": cited_sources,
        "used_context": context_text,
    }
