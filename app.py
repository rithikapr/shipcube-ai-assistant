# app.py
import os, json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from utils.ai_model import get_retrieval_answer, get_ai_response, get_items_for_tag, QNA

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET', 'shipcube_dev_secret')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            session['user'] = {'username': username}
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Enter username and password')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session.get('user'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    """
    Behavior:
     - icon-only => return tag preview list + hint (type='tag_list')
     - icon+query => try tag-restricted retrieval -> global -> generator
     - query-only => global retrieval -> generator
     - order-related queries require login (returns login_required)
    """
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    icon = data.get('icon')
    user_session = session.get('user')

    if not query and not icon:
        return jsonify({'ok': False, 'response': 'Please send a question or click an icon.'}), 400

    # ICON-only: preview suggestions/hints
    if icon and not query:
        items = get_items_for_tag(icon, limit=12)
        if not items and icon.lower() == 'about':
            items = [{'question': q.get('question'), 'answer': q.get('answer')} for q in QNA[:12]]
        return jsonify({
            'ok': True,
            'response': {
                'type': 'tag_list',
                'tag': icon,
                'items': items,
                'hint': f"You're viewing '{icon}'. Click a suggestion or type a question to search within this context."
            }
        })

    # ICON + QUERY: prefer tag-restricted retrieval
    if icon and query:
        tag_hits = get_retrieval_answer(query, top_k=6, score_threshold=0.10, tag=icon)
        if tag_hits:
            top = tag_hits[0]
            return jsonify({'ok': True, 'response': {'question': top.get('question'), 'answer': top.get('answer'), 'tag': top.get('tag'), 'source': 'retrieval_tag'}})

        global_hits = get_retrieval_answer(query, top_k=6, score_threshold=0.05, tag=None)
        if global_hits:
            top = global_hits[0]
            return jsonify({'ok': True, 'response': {'question': top.get('question'), 'answer': top.get('answer'), 'tag': top.get('tag'), 'source': 'retrieval_global'}})

        ai = get_ai_response(query)
        return jsonify({'ok': True, 'response': {'question': None, 'answer': ai, 'source': 'generated'}})

    # QUERY-only (no icon)
    if query:
        # order checks
        if 'order' in query.lower() or 'my order' in query.lower():
            if not user_session:
                return jsonify({'ok': False, 'login_required': True, 'response': 'Please log in to view order details.'}), 401
            return jsonify({'ok': True, 'response': {'question': None, 'answer': f"Hello {user_session['username']}, demo cannot access live orders here."}})

        global_hits = get_retrieval_answer(query, top_k=6, score_threshold=0.20, tag=None)
        if global_hits:
            # return a few alternatives so frontend can show multiple cards if desired
            resp_hits = [{'question': h['question'], 'answer': h['answer'], 'tag': h.get('tag'), 'score': h.get('score')} for h in global_hits[:4]]
            return jsonify({'ok': True, 'response': {'type': 'hits_list', 'hits': resp_hits, 'source': 'retrieval_global'}})

        ai = get_ai_response(query)
        return jsonify({'ok': True, 'response': {'question': None, 'answer': ai, 'source': 'generated'}})

    return jsonify({'ok': False, 'response': 'Unhandled request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
