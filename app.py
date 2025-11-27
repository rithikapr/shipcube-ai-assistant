import os
import json
import re
import sqlite3
import time
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, g, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
#from utils.ai_model import get_ai_response,  QNA
from utils.ai_model import get_ai_response, get_retrieval_answer, get_top_faq, get_items_for_tag, generate_answer_from_retrieval
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

DB_PATH = os.environ.get('DB_PATH', 'data/shipcube.db')
ANON_HISTORY_LIMIT = 100
USER_HISTORY_LIMIT = 1000

# Where your PDF-based KB is stored (created by build_doc_kb_langchain.py)
DOC_KB_DIR = Path("data/doc_kb")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Prometheus metrics -------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

ASK_REQUESTS = Counter(
    "ask_requests_total",
    "Total /ask endpoint calls"
)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET', 'shipcube_dev_secret')

# --- DB helpers --------------------------------------------------------------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at REAL DEFAULT (strftime('%s','now'))
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NULL,
        is_anonymous INTEGER DEFAULT 0,
        role TEXT NOT NULL,   -- 'user' or 'assistant'
        message TEXT NOT NULL,
        ts REAL DEFAULT (strftime('%s','now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    # client_orders table: only a subset of columns (rest still exist if imported)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS client_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID TEXT,
        MerchantName TEXT,
        CustomerName TEXT,
        OrderID TEXT,
        TransactionStatus TEXT,
        TrackingID TEXT,
        Carrier TEXT,
        CarrierService TEXT,
        FinalInvoiceAmt REAL,
        City TEXT,
        ZipCode TEXT,
        DestinationCountry TEXT,
        OrderInsertTimestamp TEXT
    )
    """)
    db.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# initialize DB at start
with app.app_context():
    init_db()

# --- NEW: load PDF doc_kb FAISS index once ----------------------------------
DOC_KB_STORE = None
DOC_KB_READY = False

def load_doc_kb():
    """Load LangChain FAISS doc_kb index from disk."""
    global DOC_KB_STORE, DOC_KB_READY
    if DOC_KB_READY:
        return DOC_KB_STORE

    if not DOC_KB_DIR.exists():
        print("[doc_kb] directory not found:", DOC_KB_DIR)
        DOC_KB_READY = False
        return None

    try:
        print("[doc_kb] loading FAISS index from", DOC_KB_DIR)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        DOC_KB_STORE = LCFAISS.load_local(
            str(DOC_KB_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        DOC_KB_READY = True
        print("[doc_kb] loaded successfully.")
    except Exception as e:
        print("[doc_kb] failed to load:", e)
        DOC_KB_STORE = None
        DOC_KB_READY = False

    return DOC_KB_STORE


def doc_kb_retrieve(query: str, k: int = 3):
    """
    Retrieve top-k chunks from PDF doc KB for a query.
    Returns list of (page_content, metadata).
    """
    store = load_doc_kb()
    if not store or not query:
        return []

    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception as e:
        print("[doc_kb] similarity search error:", e)
        return []

    cleaned = []
    for doc, score in results:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        # simple heuristic: skip extremely short / noisy chunks
        if len(text) < 40:
            continue
        cleaned.append((text, doc.metadata, score))

    return cleaned


def build_doc_kb_answer(query: str):
    """
    Use PDF KB chunks + LLM (get_ai_response) to build a nice answer.
    Returns text answer or None if no good chunks.
    """
    hits = doc_kb_retrieve(query, k=3)
    if not hits:
        return None, None

    # you can play with score thresholds later
    # For now we just use the top 2 chunks.
    top_chunks = hits[:2]
    context_parts = []
    sources = []
    for text, meta, score in top_chunks:
        source_file = meta.get("source_file") or meta.get("source") or "document"
        context_parts.append(text)
        sources.append(source_file)

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are ShipCube AI, a supply chain and 3PL assistant. "
        "Use ONLY the context below to answer the user's question in a clear, concise way.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer strictly based on the context above. If the context is not enough, say you are not sure."
    )

    try:
        answer = get_ai_response(prompt)
    except Exception as e:
        print("[doc_kb] get_ai_response error:", e)
        return None, None

    # join sources (de-duplicate)
    sources_str = ", ".join(sorted(set(sources)))
    return answer, sources_str

@app.before_request
def start_timer():
    # store start time on Flask's `g` object
    g.start_time = time.time()


@app.after_request
def record_request_data(response):
    try:
        latency = time.time() - getattr(g, "start_time", time.time())
        endpoint = request.endpoint or "unknown"

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    except Exception as e:
        print("Metrics error:", e)
    return response

# --- auth routes ------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template('login.html',
                                   error='Provide username and password',
                                   mode='register')
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )
            db.commit()
            # ✅ show success message on login page
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('login.html',
                                   error='Username already taken',
                                   mode='register')
    return render_template('login.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = get_db()
        row = db.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if row and check_password_hash(row['password_hash'], password):
            session['user'] = {'id': row['id'], 'username': username}
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials', mode='login')
    return render_template('login.html', mode='login')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# --- history helpers --------------------------------------------------------
def append_chat_to_history(role, message, user_session):
    """Persist message to session or DB depending on login state."""
    if user_session:
        user_id = user_session['id']
        db = get_db()
        db.execute(
            "INSERT INTO chats (user_id, is_anonymous, role, message) VALUES (?, 0, ?, ?)",
            (user_id, role, message)
        )
        db.execute("""
          DELETE FROM chats WHERE id IN (
            SELECT id FROM chats WHERE user_id=? ORDER BY ts DESC LIMIT -1 OFFSET ?
          )
        """, (user_id, USER_HISTORY_LIMIT))
        db.commit()
    else:
        hist = session.get('anon_history', [])
        hist.append({'role': role, 'message': message, 'ts': time.time()})
        if len(hist) > ANON_HISTORY_LIMIT:
            hist = hist[-ANON_HISTORY_LIMIT:]
        session['anon_history'] = hist


def get_history_for_current_user():
    user = session.get('user')
    if user:
        db = get_db()
        rows = db.execute(
            "SELECT role, message, ts FROM chats WHERE user_id=? ORDER BY ts ASC",
            (user['id'],)
        ).fetchall()
        return [{'role': r['role'], 'message': r['message'], 'ts': r['ts']} for r in rows]
    else:
        return session.get('anon_history', [])


# --- helper: find order in client_orders -----------------------------------
ORDER_RE = re.compile(r'\b(\d{5,12})\b')   # naive numeric candidate (5-12 digits)


def find_order_in_db(token: str):
    db = get_db()
    t = (token or "").strip()

    # Build variants to handle things like "219195797" vs "219195797.0"
    variants = {t}

    if re.fullmatch(r"\d+", t):
        # pure digits in query, but DB often has ".0"
        variants.add(f"{t}.0")
    elif re.fullmatch(r"\d+\.0", t):
        # query has ".0", DB may have plain integer
        variants.add(t.split(".")[0])

    var_list = list(variants)
    # Ensure we always have 2 elements for the IN (?, ?) placeholders
    while len(var_list) < 2:
        var_list.append(var_list[0])

    q = """
      SELECT
        order_id,
        store_orderid,
        tracking_id,
        customername,
        merchant_name,
        transaction_status,
        final_invoice_amt,
        city,
        zip_code,
        carrier,
        carrier_service
      FROM client_orders
      WHERE order_id IN (?, ?)
         OR store_orderid IN (?, ?)
         OR tracking_id = ?
         OR invoice_number = ?
      LIMIT 1
    """

    row = db.execute(
        q,
        (
            var_list[0], var_list[1],   # order_id variants
            var_list[0], var_list[1],   # store_orderid variants
            t,                          # tracking_id exact
            t,                          # invoice_number exact
        ),
    ).fetchone()

    return dict(row) if row else None

# --- main app endpoints -----------------------------------------------------
@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'ok': True, 'history': get_history_for_current_user()})

@app.route("/faq/top", methods=["GET"])
# def faq_top():
#     """
#     Return 5–6 top FAQ Q&As for the right-hand pane.
#     Non-category specific, just a small curated list from QNA.
#     """
#     items = []
#     try:
#         if isinstance(QNA, list):
#             for q in QNA[:6]:
#                 items.append({
#                     "question": q.get("question", ""),
#                     "answer": q.get("answer", ""),
#                 })
#     except Exception as e:
#         print("faq_top error:", e)
#     return jsonify({"ok": True, "items": items})
@app.route("/faq/top", methods=["GET"])
def faq_top():
    items = []
    try:
        for q in get_top_faq(6):
            items.append({
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
            })
    except Exception as e:
        print("faq_top error:", e)
    return jsonify({"ok": True, "items": items})

@app.route("/faq/<tag>", methods=["GET"])
def faq_by_tag(tag):
    """
    Return FAQ items filtered by tag (about, warehouse, logistics, storage, finance).
    """
    items = []
    try:
        faqs = get_items_for_tag(tag, limit=10)
        for q in faqs:
            items.append({
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
            })
    except Exception as e:
        print("faq_by_tag error:", e)
    return jsonify({"ok": True, "items": items})


@app.route('/ask', methods=['POST'])
def ask():
    ASK_REQUESTS.inc()   # count each /ask call
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    icon = data.get('icon')
    user_session = session.get('user')

    if not query and not icon:
        return jsonify({'ok': False, 'response': 'Please send a question.'}), 400

    # Build the effective question text (for history only)
    effective_query = (f"[{icon}] " if icon else "") + query if query else ''
    append_chat_to_history('user', effective_query, user_session)

    # ---------- ORDER / TRACKING HANDLING (early exit) ----------
    candidate = None
    m = ORDER_RE.search(query)
    if m:
        candidate = m.group(1)

    explicit = re.search(r'order\s+(\d{5,12})', query, re.IGNORECASE)
    if explicit:
        candidate = explicit.group(1)

    if candidate:
        # Not logged in → ask to login, do NOT hit PDFs / RAG
        if not user_session:
            msg = (
                "I detect an order or tracking number. "
                "Please log in to view order-specific details."
            )
            append_chat_to_history('assistant', msg, user_session)
            return jsonify({
                'ok': True,
                'response': {
                    'question': None,
                    'answer': msg,
                    'source': 'auth_required'
                }
            })

        # Logged in → look up in client_orders
        order = find_order_in_db(candidate)
        if order:
            text = (
                f"Order {order.get('order_id') or order.get('store_orderid') or candidate}: "
                f"status {order.get('transaction_status') or 'unknown'}. "
                f"Customer: {order.get('customername') or 'N/A'}. "
                f"Merchant: {order.get('merchant_name') or 'N/A'}. "
                f"Carrier: {order.get('carrier') or order.get('carrier_service') or 'N/A'}. "
                f"Amount: {order.get('final_invoice_amt') or 'N/A'}. "
                f"Destination: {order.get('city') or 'N/A'}, "
                f"Zip: {order.get('zip_code') or 'N/A'}."
            )
            append_chat_to_history('assistant', text, user_session)
            return jsonify({
                'ok': True,
                'response': {
                    'question': f"Order {candidate}",
                    'answer': text,
                    'source': 'order_db'
                }
            })
        else:
            msg = f"No order found for {candidate}."
            append_chat_to_history('assistant', msg, user_session)
            return jsonify({
                'ok': True,
                'response': {
                    'question': None,
                    'answer': msg,
                    'source': 'order_not_found'
                }
            })
    # ---------- END ORDER HANDLING ----------

    # ---------- GLOBAL RAG + LLM ----------
    try:
        rag_res = generate_answer_from_retrieval(
            query,
            top_k=3,
            score_threshold=0.25,
        )
    except Exception as e:
        print("RAG generation error:", e)
        rag_res = None

    if rag_res and rag_res.get("answer"):
        answer_text = rag_res["answer"]
        # build a compact source string for client display
        src_list = [
            f"{s['source']} (id={s['id']}, score={s['score']:.3f})"
            for s in rag_res.get("sources", [])
        ]
        src_str = ", ".join(src_list) if src_list else "global_kb"
        append_chat_to_history('assistant', answer_text, user_session)
        return jsonify({
            'ok': True,
            'response': {
                'question': query,
                'answer': answer_text,
                'source': src_str,
            }
        })


    # Fallback to LLM
    ai_resp = get_ai_response(query)
    append_chat_to_history('assistant', ai_resp, user_session)
    return jsonify({
        'ok': True,
        'response': {
            'question': None,
            'answer': ai_resp,
            'source': 'generated'
        }
    })

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)