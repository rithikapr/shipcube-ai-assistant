# utils/ai_model.py
import os
import json
import numpy as np
from typing import List, Optional, Dict

from sentence_transformers import SentenceTransformer, util

# Optional HF generator
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception:
    pipeline = None

# CONFIG - tune as needed
QNA_PATH = os.environ.get('QNA_PATH', 'data/qna.json')
INDEX_PATH = os.environ.get('INDEX_PATH', 'data/qna.faiss.index')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
MODEL_GEN = os.environ.get('MODEL_NAME', 'gpt2')  # fallback generator; replace if you have better model

# Load QnA
try:
    with open(QNA_PATH, 'r', encoding='utf-8') as f:
        QNA = json.load(f)
except Exception:
    QNA = []

# Prepare embedder
_embedder = SentenceTransformer(EMBED_MODEL)

# Compute embeddings (cache in memory)
_q_texts = [ (q.get('question','') + " \n" + q.get('context','')) for q in QNA ]
_q_embeddings = None
if _q_texts:
    _q_embeddings = _embedder.encode(_q_texts, convert_to_numpy=True)
    # normalize for cosine
    norms = np.linalg.norm(_q_embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    _q_embeddings = _q_embeddings / norms

# Try to use FAISS if available for faster search & persistent index
_USE_FAISS = False
try:
    import faiss
    if _q_embeddings is not None:
        d = _q_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine
        index.add(_q_embeddings.astype('float32'))
        _USE_FAISS = True
except Exception:
    # faiss not available or failed; will use util.semantic_search fallback
    index = None
    _USE_FAISS = False


# Retrieval function returns list[ {question, answer, tag, context, score, idx} ]
def get_retrieval_answer(query: str, top_k: int = 5, score_threshold: float = 0.45, tag: Optional[str] = None) -> List[Dict]:
    """
    Returns list of hits sorted by score desc (score is cosine similarity 0..1).
    If no hits above threshold, returns empty list.
    If tag provided, it will restrict results to items where item['tag'] or item['context'] equals tag (case-insensitive)
    """
    if not query:
        return []

    q_emb = _embedder.encode([query], convert_to_numpy=True)
    # normalize
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    hits = []
    if _USE_FAISS and index is not None:
        D, I = index.search(q_emb.astype('float32'), top_k)
        # D are inner products (cosine) because we normalized
        D = D[0].tolist()
        I = I[0].tolist()
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(QNA):
                continue
            item = QNA[idx]
            item_tag = (item.get('tag') or item.get('context') or '').lower()
            if tag and tag.lower() not in item_tag:
                # skip if tag filter present and not match
                continue
            if score >= score_threshold:
                hits.append({
                    'question': item.get('question'),
                    'answer': item.get('answer'),
                    'tag': item.get('tag'),
                    'context': item.get('context'),
                    'score': float(score),
                    'idx': int(idx)
                })
    else:
        # fallback: sentence_transformers semantic_search
        candidates = util.semantic_search(q_emb, _q_embeddings, top_k=top_k)[0]
        for c in candidates:
            idx = int(c['corpus_id'])
            score = float(c['score'])  # cosine-like
            item = QNA[idx]
            item_tag = (item.get('tag') or item.get('context') or '').lower()
            if tag and tag.lower() not in item_tag:
                continue
            if score >= score_threshold:
                hits.append({
                    'question': item.get('question'),
                    'answer': item.get('answer'),
                    'tag': item.get('tag'),
                    'context': item.get('context'),
                    'score': float(score),
                    'idx': idx
                })

    # sort desc
    hits = sorted(hits, key=lambda x: x.get('score',0), reverse=True)
    return hits


# Generator fallback (lazy)
_generator = None
def _init_generator():
    global _generator
    if _generator is None and pipeline is not None:
        try:
            tok = AutoTokenizer.from_pretrained(MODEL_GEN)
            model = AutoModelForCausalLM.from_pretrained(MODEL_GEN)
            _generator = pipeline('text-generation', model=model, tokenizer=tok)
        except Exception as e:
            print("Generator init failed:", e)
            _generator = None

def get_ai_response(prompt: str) -> str:
    if not prompt:
        return "Please provide a question."
    _init_generator()
    if _generator:
        sys_prompt = "You are ShipCube AI, a helpful logistics assistant. Answer concisely and factually."
        full = sys_prompt + "\nUser: " + prompt + "\nAssistant:"
        try:
            out = _generator(full, max_new_tokens=150, do_sample=True, temperature=0.6, top_p=0.95)
            txt = out[0]['generated_text']
            # Trim to the assistant's response part if system prompt repeated
            # Return full text for now
            return txt
        except Exception as e:
            print("Generation failed:", e)
            return "Sorry, the model failed to generate a response."
    # no generator available
    return "No generator model available — please install transformers and a model, or add better QnA data."

# helper to return top N items for an icon (no scores)
def get_items_for_tag(tag: str, limit: int = 12):
    tag_low = (tag or '').lower()
    items = []
    for it in QNA:
        t = (it.get('tag') or it.get('context') or '').lower()
        if tag_low == 'about' or tag_low == '' :
            # about -> general items (no tag) OR everything
            pass
        if tag_low and tag_low not in t:
            continue
        items.append({'question': it.get('question'), 'answer': it.get('answer'), 'tag': it.get('tag')})
        if len(items) >= limit:
            break
    # if empty and tag!='about', try partial matching
    if not items and tag_low and tag_low != 'about':
        for it in QNA:
            if tag_low in (it.get('question') or '').lower() or tag_low in (it.get('answer') or '').lower():
                items.append({'question': it.get('question'), 'answer': it.get('answer'), 'tag': it.get('tag')})
                if len(items) >= limit:
                    break
    return items
