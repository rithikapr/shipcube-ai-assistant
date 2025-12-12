# utils/ai_model.py
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"

QNA_JSON_PATH = DATA_DIR / "qna.json"
GLOBAL_KB_DIR = DATA_DIR / "global_kb"    # unified KB from build_global_kb.py

# MUST match EMBED_MODEL_NAME in build_global_kb.py
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

_global_vectordb: Optional[LCFAISS] = None
_model: Optional[SentenceTransformer] = None
_llm: Optional[ChatGoogleGenerativeAI] = None
QNA: List[Dict] = []  # exported for other modules
QNA: List[Dict] = []  # exported for other modules

# cached semantic embeddings for FAQ questions
_QNA_EMBEDDINGS = None   # type: ignore
_QNA_QUESTIONS: List[str] = []

# ---------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------
def _ensure_qna_embeddings():
    """
    Encode all FAQ questions once into embeddings, cached globally.
    """
    global _QNA_EMBEDDINGS, _QNA_QUESTIONS

    _ensure_qna_loaded()
    if _QNA_EMBEDDINGS is not None:
        return

    if not QNA:
        _QNA_EMBEDDINGS = None
        _QNA_QUESTIONS = []
        return

    questions: List[str] = []
    for item in QNA:
        q_text = (item.get("question") or item.get("Question") or "").strip()
        questions.append(q_text)

    model = _ensure_model()
    # normalised embeddings → cosine similarity is just dot product
    _QNA_EMBEDDINGS = model.encode(questions, normalize_embeddings=True)
    _QNA_QUESTIONS = questions

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


def _ensure_qna_loaded():
    """Load QNA JSON once into QNA list."""
    global QNA
    if QNA:
        return

    if not QNA_JSON_PATH.exists():
        print("[QNA] qna.json not found at", QNA_JSON_PATH)
        QNA = []
        return

    with QNA_JSON_PATH.open("r", encoding="utf-8") as f:
        QNA = json.load(f)


def _ensure_global_kb_loaded():
    """
    Load the unified LangChain FAISS index built by build_global_kb.py
    from data/global_kb (PDF chunks + FAQ).
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

# ---------------------------------------------------------------------
# FAQ helpers
# ---------------------------------------------------------------------

def get_top_faq(limit: int = 6) -> List[Dict]:
    """Return top N FAQ entries from qna.json."""
    _ensure_qna_loaded()
    if not isinstance(QNA, list):
        return []
    return QNA[:limit]

def direct_faq_match(query: str) -> Optional[Dict]:
    """
    Generic lexical matcher over FAQ questions.

    Idea: if most of the *user's* words appear in a FAQ question,
    we treat it as a direct FAQ-style query and answer from qna.json.
    No domain-specific hardcoding.
    """
    _ensure_qna_loaded()
    q_raw = (query or "").strip()
    if not q_raw or not QNA:
        return None

    # 🔹 If query has "Context: ..." appended, strip it off.
    lower_all = q_raw.lower()
    idx = lower_all.find("context:")
    if idx != -1:
        q_raw = q_raw[:idx].strip()

    def norm_tokens(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return text.split()

    q_tokens = norm_tokens(q_raw)
    if len(q_tokens) == 0:
        return None

    q_set = set(q_tokens)

    best_item: Optional[Dict] = None
    best_score: float = 0.0

    for item in QNA:
        faq_q_raw = (item.get("question") or item.get("Question") or "").strip()
        if not faq_q_raw:
            continue

        faq_tokens = norm_tokens(faq_q_raw)
        if not faq_tokens:
            continue

        faq_set = set(faq_tokens)
        inter = q_set & faq_set
        if not inter:
            continue

        coverage_query = len(inter) / len(q_set)
        coverage_faq = len(inter) / len(faq_set)

        score = 0.7 * coverage_query + 0.3 * coverage_faq
        score -= 0.01 * len(faq_set)

        if score > best_score:
            best_score = score
            best_item = item

    # loosen threshold a bit to be safe
    if best_item is not None and best_score >= 0.55:
        return best_item

    return None

def get_items_for_tag(tag: str, limit: int = 10) -> List[Dict]:
    """
    Return FAQ items whose Departments field matches the given tag
    (about / warehouse / logistics / finance), case-insensitive.
    """
    _ensure_qna_loaded()
    if not QNA:
        return []

    tag_lower = (tag or "").lower()
    items: List[Dict] = []

    for q in QNA:
        # Use Departments as main category
        dept = (q.get("Departments") or "").lower()

        # Optional fallback if you ever add tag/context later
        if not dept:
            dept = (q.get("tag") or q.get("context") or "").lower()

        if dept == tag_lower:
            items.append(q)

    return items[:limit]

# ---------------------------------------------------------------------
# Unified global_kb retrieval (PDF + FAQ)
# ---------------------------------------------------------------------

def _search_global_kb(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search the unified global_kb index (PDF + FAQ).
    Returns list of dicts with question, answer, score, source, metadata.
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
        faq_question = meta.get("question") or meta.get("Question")

        results.append(
            {
                "question": faq_question,               # for FAQ docs; None for PDFs
                "answer": doc.page_content.strip(),     # raw chunk text
                "score": float(sim),
                "source": src,                          # "faq" or PDF path
                "metadata": meta,
            }
        )
    return results

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
    # pdf_hits = [h for h in raw_hits if h.get("source") != "faq"]  # kept for clarity

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
# LLM wrapper (Gemini)
# ---------------------------------------------------------------------

def get_ai_response(prompt: str) -> str:
    """
    Generate a reply using Gemini.
    This is used both for:
      - summarising PDF / FAQ chunks (RAG)
      - pure generative fallback answers
    """
    llm = _ensure_llm()

    # We'll compress system + user into a single text prompt
    system_msg = (
        "You are ShipCube AI, a helpful logistics and 3PL assistant. "
        "Answer clearly and concisely. If you are given a 'Context:' "
        "section, you MUST base your answer only on that context. "
        "Make sure the response is concise and presentable."
    )

    full_prompt = system_msg + "\n\n" + prompt

    try:
        res = llm.invoke(full_prompt)
        # langchain-google-genai returns an object with .content
        content = (getattr(res, "content", "") or "").strip()
        if not content:
            return (
                "I couldn't understand that — could you please rephrase or "
                "ask a more specific question?"
            )
        return content
    except Exception as e:
        print("[genai] generation error:", e)
        return "Sorry, the model couldn't produce a useful answer right now."

# ---------------------------------------------------------------------
# Summariser used for chat history / context compression
# ---------------------------------------------------------------------

def summarise_context(context: str) -> str:
    """
    Summarise a long context string into < 100 words
    for storing / passing into RAG as chat history.
    """
    llm = _ensure_llm()

    system_msg = (
        "You are an expert summarizer. I need to pass this to a RAG model, so context is required. "
        "Summarize the given context in less than 100 words. If context is too large, trim from the earlier "
        "context, that is, the lines from the start. "
        "Summarize as you store context information for your user for a chat. "
        "Don't add anything extra than the context like I can do this or that. If no context, leave blank response."
    )

    full_prompt = system_msg + "\n\n" + (context or "")

    try:
        res = llm.invoke(full_prompt)
        content = (getattr(res, "content", "") or "").strip()
        # If strictly no content, return empty (as per instructions)
        return content
    except Exception as e:
        print("[genai] summarization error:", e)
        # For summariser, an empty string is safer than an error message
        return ""

# ---------------------------------------------------------------------
# Higher-level: retrieval + Gemini summarisation
# ---------------------------------------------------------------------

def generate_answer_from_retrieval(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.2,
    max_context_tokens: int = 1500,
    concise_word_limit: int = 70,
) -> Dict:
    """
    0) Try direct FAQ match on qna.json.
    1) Otherwise run retrieval (PDF + FAQ) and build a concise context.
    2) Call Gemini to summarise that context.
    """

    # ---------- 0) DIRECT FAQ MATCH ----------
    direct = direct_faq_match(query)
    if direct:
        q_text = direct.get("Question") or direct.get("question") or ""
        a_text = direct.get("Answer") or direct.get("answer") or ""
        raw = f"Q: {q_text}\nA: {a_text}"

        return {
            "query": query,
            "answer": a_text,
            "sources": [{
                "id": 1,
                "source": "faq_direct",
                "score": 1.0,
                "snippet": raw[:300] + ("..." if len(raw) > 300 else ""),
            }],
            "used_context": raw,
        }

    # ---------- 1) VECTOR RETRIEVAL (RAG) ----------
    hits = get_retrieval_answer(query, top_k=top_k, score_threshold=score_threshold)
    if not hits:
        prompt_noctx = (
            "Context:\n\n"
            "User question: {}\n\n"
            "Instructions: You do NOT have context. Answer concisely and clearly. "
            "If you don't know, say 'I don't know' or ask for clarification."
        ).format(query)
        ans = get_ai_response(prompt_noctx)
        return {"query": query, "answer": ans, "sources": [], "used_context": ""}

    # ---------- 2) BUILD CONTEXT & CALL GEMINI ----------
    sources = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.get("answer") or "").strip()
        snippet_short = (
            snippet if len(snippet) <= 300
            else snippet[:300].rsplit(" ", 1)[0] + "..."
        )
        sources.append(
            {
                "id": i,
                "source": h.get("source", "unknown"),
                "score": float(h.get("score", 0.0)),
                "snippet": snippet_short,
            }
        )

    max_chars = max_context_tokens * 4
    context_parts = []
    chars_used = 0

    for s in sources:
        idx = s["id"] - 1
        full_text = (hits[idx].get("answer") or "").strip()
        header = f"[{s['id']}] Source: {s['source']} (score={s['score']:.3f})\n"
        block = header + full_text + "\n\n"

        if chars_used + len(block) > max_chars:
            remaining = max(0, max_chars - chars_used - 10)
            if remaining <= 0:
                break
            snippet = full_text[:remaining].rsplit(" ", 1)[0]
            block = header + snippet + "...\n\n"
            context_parts.append(block)
            chars_used += len(block)
            break

        context_parts.append(block)
        chars_used += len(block)

    context_text = "".join(context_parts).strip()

    prompt = (
        "Context:\n\n"
        f"{context_text}\n\n"
        "User question:\n"
        f"{query}\n\n"
        "Instructions:\n"
        "- Base your answer ONLY on the Context above.\n"
        f"- Keep the answer concise and clear (around {concise_word_limit} words or less).\n"
        "- If the context doesn't contain enough info to answer, say \"I don't know\" "
        "and list the most relevant sources.\n"
        "- Provide a 1–2 sentence actionable answer.\n"
    )

    answer = get_ai_response(prompt)

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



def contextualize_query(chat_history_str, latest_question):
    """
        Rewrites the latest question based on chat history to make it standalone.
        #TODO: CAUTION: This can incur additional LLM costs and need to add a upper limit.
    """
    
    if not chat_history_str:
        return latest_question

    contextualize_q_prompt = PromptTemplate.from_template(
        """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        
        Chat History:
        {chat_history}
        
        Follow Up Input: {question}
        
        Standalone Question:"""
    )
    
    chain = contextualize_q_prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "chat_history": chat_history_str,
        "question": latest_question
    })
    
    return response
