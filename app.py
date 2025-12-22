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
from typing import List, Dict, Optional
from werkzeug.security import generate_password_hash, check_password_hash
from utils.ai_model import (
    get_ai_response,
    get_top_faq,
    get_items_for_tag    
)
from utils.rag_pipeline import shipcube_agent 
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


DB_PATH = os.environ.get('DB_PATH', 'data/shipcube.db')
ANON_HISTORY_LIMIT = 100
USER_HISTORY_LIMIT = 1000
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

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
# NOTE: replace with a real secret in production via env var FLASK_SECRET
app.secret_key = os.environ.get('FLASK_SECRET', 'shipcube_dev_secret')

# session-based conversation summary helpers (per-session, not global)
def _summary_key():
    user = session.get('user')
    return f"summary_user_{user['id']}" if user else f"summary_anon_{session.get('_id', 'guest')}"

def get_session_summary():
    return session.get(_summary_key(), "")

def set_session_summary(text: str):
    session[_summary_key()] = text

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

    # client_orders table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS client_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipping_label_id TEXT,
    order_date TEXT,
    order_number TEXT,
    quantity_shipped INTEGER,
    order_id TEXT,
    carrier TEXT,
    shipping_method TEXT,
    tracking_number TEXT,
    created_at TEXT,
    to_name TEXT,
    final_amount REAL,
    zip TEXT,
    state TEXT,
    country TEXT,
    size_dimensions TEXT,   -- Length x Width x Height (in)
    weight_oz REAL,
    tpl_customer TEXT,      -- 3PL Customer
    warehouse TEXT)""")
#    order_tags TEXT 

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at REAL DEFAULT (strftime('%s','now'))
    )
    """)

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NULL,
        is_anonymous INTEGER DEFAULT 0,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        message_id TEXT,
        ts REAL DEFAULT (strftime('%s','now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NULL,
        message_id TEXT NOT NULL,
        rating INTEGER,
        question TEXT,
        answer TEXT,
        model_name TEXT,
        ts REAL DEFAULT (strftime('%s','now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
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

@app.before_request
def start_timer():
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
            # store minimal user object in session
            session['user'] = {'id': row['id'], 'username': username}
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials', mode='login')
    return render_template('login.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Receive thumbs up/down for a specific assistant response.
    Payload expected (JSON):
      {
        "message_id": "msg-173314234",
        "rating": 1 or -1,
        "question": "...",
        "answer": "...",
        "model": "gemini-2.5-flash-lite"
      }
    """
    data = request.get_json(silent=True) or {}
    msg_id = data.get('message_id')
    rating = data.get('rating')

    if msg_id is None or rating not in [1, -1]:
        return jsonify({'ok': False, 'error': 'Invalid payload'}), 400

    user = session.get('user')
    user_id = str(user['id']) if user else session.get('_id')

    question = data.get('question')
    answer = data.get('answer')
    model_name = data.get('model')

    db = get_db()
    db.execute("""
        INSERT INTO feedback (user_id, message_id, rating, question, answer, model_name)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, msg_id, rating, question, answer, model_name))
    db.commit()

    return jsonify({'ok': True})

# --- history helpers --------------------------------------------------------
def make_message_id() -> str:
    return f"msg-{int(time.time() * 1000)}"

def append_chat_to_history(role, message, user_session, message_id=None):
    """
    Save every message to the chats table.
    If the user is not logged in, store user_id = session token (string) and is_anonymous = 1.
    `user_session` expected to be either:
      - dict with 'id' and optional 'is_guest'
      - None (will be treated as anonymous and a session token created if missing)
    """
    db = get_db()

    if user_session:
        user_id = str(user_session["id"])
        is_anon = 0 if not user_session.get('is_guest', False) else 1
    else:
        # Ensure session token exists so we can correlate anonymous chats
        if '_id' not in session:
            session['_id'] = os.urandom(8).hex()
        user_id = session.get('_id')
        is_anon = 1

    if message_id is None:
        message_id = make_message_id()

    db.execute(
        """
        INSERT INTO chats (user_id, is_anonymous, role, message, message_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, is_anon, role, message, message_id),
    )

    # Optional: trim history per user / anon
    if user_session and not user_session.get('is_guest', False):
        # logged-in user trimming by integer id still works because user_id stored as TEXT
        db.execute(
            """
            DELETE FROM chats
            WHERE id IN (
              SELECT id FROM chats
              WHERE user_id = ?
              ORDER BY ts DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (str(user_session['id']), USER_HISTORY_LIMIT),
        )
    else:
        # anonymous trimming by session token
        db.execute(
            """
            DELETE FROM chats
            WHERE id IN (
              SELECT id FROM chats
              WHERE user_id = ?
              ORDER BY ts DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (session.get('_id'), ANON_HISTORY_LIMIT),
        )

    db.commit()
    return message_id

def get_history_for_current_user():
    user = session.get('user')
    if user:
        db = get_db()
        rows = db.execute(
            "SELECT role, message, ts FROM chats WHERE user_id=? ORDER BY ts ASC",
            (str(user['id']),)
        ).fetchall()
        return [{'role': r['role'], 'message': r['message'], 'ts': r['ts']} for r in rows]
    else:
        # return anonymous history for current session token
        token = session.get('_id')
        if not token:
            return []
        db = get_db()
        rows = db.execute(
            "SELECT role, message, ts FROM chats WHERE user_id=? ORDER BY ts ASC",
            (token,)
        ).fetchall()
        return [{'role': r['role'], 'message': r['message'], 'ts': r['ts']} for r in rows]

# --- helper: find order in client_orders -----------------------------------
ORDER_RE = re.compile(r'\b(\d{9})\b')   # naive numeric candidate (9 digits)


# --- main app endpoints -----------------------------------------------------
@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'ok': True, 'history': get_history_for_current_user()})

@app.route("/faq/top", methods=["GET"])
def faq_top():
    user = session.get('user')
    is_logged_in = user is not None

    items = []
    try:
        for q in get_top_faq(6):
            question = q.get("question") or q.get("Question") or ""
            ans = q.get("answer") or q.get("Answer") or ""

            dept = (
                q.get("Departments")
                or q.get("department")
                or q.get("tag")
                or q.get("context")
                or ""
            ).strip().lower()

            requires_login = bool(
                q.get("RequiresLogin") or q.get("requires_login")
            )

            if not is_logged_in and requires_login:
                ans = (
                    "This answer contains detailed pricing information. "
                    "Please create an account and log in to view the exact rates."
                )

            items.append({
                "question": question,
                "answer": ans,
            })
    except Exception as e:
        print("faq_top error:", e)
    return jsonify({"ok": True, "items": items})

@app.route("/faq/<tag>", methods=["GET"])
def faq_by_tag(tag):
    """
    Return FAQ items filtered by category.
    Category comes from the Departments field in qna.json.
    """
    user = session.get('user')
    is_logged_in = user is not None

    items = []
    try:
        faqs = get_items_for_tag(tag, limit=10)
        for q in faqs:
            question = q.get("question") or q.get("Question") or ""
            ans = q.get("answer") or q.get("Answer") or ""

            dept = (
                q.get("Departments")
                or q.get("department")
                or q.get("tag")
                or q.get("context")
                or ""
            ).strip().lower()

            requires_login = bool(
                q.get("RequiresLogin") or q.get("requires_login")
            )

            if not is_logged_in and requires_login:
                ans = (
                    "This FAQ includes specific pricing and rate details. "
                    "Please log in to your ShipCube account to see full pricing."
                )

            items.append({
                "question": question,
                "answer": ans,
            })
    except Exception as e:
        print("faq_by_tag error:", e)
    return jsonify({"ok": True, "items": items})


def markdown_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)



@app.route('/ask', methods=['POST'])
def ask():
    ASK_REQUESTS.inc()
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    icon = data.get('icon')

    if not query and not icon:
        return jsonify({'ok': False, 'response': 'Please send a question.'}), 400

    if 'user' in session:
        user_obj = session['user']
        user_id = user_obj['id']
    else:
        if '_id' not in session:
            session['_id'] = os.urandom(4).hex()
        user_id = session['_id']
        user_obj = {'id': user_id, 'is_guest': True}

    
    effective_query = (f"[{icon}] " if icon else "") + query
    append_chat_to_history('user', effective_query, user_obj)
    print(f"[Agent] Effective Query: {effective_query}")

    chat_history_str = get_session_history(user_id)
    response_data = shipcube_agent.process_query(query, chat_history_str, user_obj)

    answer_text = response_data.get('answer')
    print(f"[Agent] Answer: {answer_text}")
    append_chat_to_history('assistant', answer_text, user_obj)

    answer_text = markdown_bold_to_html(answer_text)

    return jsonify({
        'ok': True,
        'response': {
            'question': query,
            'answer': answer_text,
            'source': response_data['source']
        }
    })

def get_session_history(user_id, limit=10):
    """
    Fetches the last 'limit' messages for a specific user_id and formats them as a string for the AI model.
    user_id may be an integer (for logged-in) or a session token string (for anonymous).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT role, message 
        FROM chats 
        WHERE user_id = ? 
        ORDER BY ts DESC 
        LIMIT ?
    """
    cursor.execute(query, (str(user_id), limit))
    rows = cursor.fetchall()
    conn.close()

    rows = rows[::-1]  # oldest -> newest

    formatted_history = []
    for role, msg in rows:
        display_role = "Human" if role == "user" else "AI"
        formatted_history.append(f"{display_role}: {msg}")

    return "\n".join(formatted_history)

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# Invoice route with login check
@app.route('/invoice')
def invoice():
    user = session.get('user')
    if not user:
        flash("Please log in to access invoices.", "warning")
        return redirect(url_for('login'))

    # TODO: placeholder – implement invoice download logic later
    return render_template('invoice.html', user=user)


def get_session_history(user_id, limit=10):
    """
        Fetches the last 'limit' messages for a specific user_id and formats them as a string for the AI model.

    """

    conn = sqlite3.connect('data/shipcube.db')
    cursor = conn.cursor()
    
    query = """
        SELECT role, message 
        FROM chats 
        WHERE user_id = ? 
        ORDER BY ts DESC 
        LIMIT ?
    """
    cursor.execute(query, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    
    rows = rows[::-1]
    
    formatted_history = []
    for role, msg in rows:
        display_role = "Human" if role == "user" else "AI"
        formatted_history.append(f"{display_role}: {msg}")
        
    return "\n".join(formatted_history)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
