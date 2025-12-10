# ShipCube AI – Logistics / 3PL Chat Assistant

ShipCube AI is a conversational assistant for a 3PL / logistics platform, built with Flask, LangChain, FAISS, and Google Gemini.  
It allows users to ask questions about ShipCube (warehouses, logistics, finance, etc.), browse curated FAQ, and (for logged-in users) access order-specific information and 
restricted content like pricing or invoices.

Features
# Web chat interface
  - Single chat box with typing indicator and avatars.
  - Right-hand FAQ panel with category filters.
  - Click an FAQ to pre-fill the chat box.

# Authentication
  - User registration and login using `users` table in SQLite.
  - Session-based auth (Flask sessions).
  - Logged-in users can:
    - See protected content (e.g., detailed pricing answers).
    - Access the Get Invoice page (placeholder for now).
    - Get order-specific responses from the `client_orders` table.

# RAG (Retrieval-Augmented Generation)
  - Unified FAISS vector store (`data/global_kb`) built from:
    - FAQ JSON (`data/qna.json`)
    - PDF documentation (`data/pdfs`)
  - Uses `sentence-transformers/all-mpnet-base-v2` for embeddings.
  - Retrieval + answer generation handled by:
    - generate_answer_from_retrieval(...) in `utils/ai_model.py`
    - Google Gemini (`gemini-2.5-flash-lite`) via `langchain-google-genai`.

# Order detection & pricing guardrails
  - Automatically detects possible order IDs / tracking IDs in user queries and fetches from `client_orders` (if logged in).
  - For pricing-related queries, non-logged-in users get a generic message asking them to log in to see detailed charges.

# Feedback collection
  - Each AI response shows “Was this helpful?” with 👍/👎 buttons.
  - Feedback is posted to `/feedback` and stored in the `feedback` table with:
    - `message_id`
    - `rating` (1 or -1)
    - question, answer, model name.

# Metrics
  - Prometheus metrics exposed at `/metrics`:
    - `http_requests_total` (by method, endpoint, status)
    - `http_request_duration_seconds` (latency histogram)
    - `ask_requests_total` (total `/ask` calls)

---

# Tech Stack

- Backend: Python, Flask
- Frontend: HTML, CSS, vanilla JS (`static/js/chat.js`)
- Database: SQLite (`data/shipcube.db`)
- LLM: Google Gemini (`ChatGoogleGenerativeAI`)
- Embeddings / RAG:
  - `sentence-transformers/all-mpnet-base-v2`
  - `langchain_community.vectorstores.FAISS`
  - `langchain_community.embeddings.HuggingFaceEmbeddings`
- Monitoring: `prometheus_client`

---

Project Structure (key files)

project_root/
├─ app.py                   # Flask app, routes, DB init, /ask logic
├─ utils/
│  └─ ai_model.py           # LLM wrapper, FAQ helpers, RAG pipeline
├─ build_global_kb.py       # Builds unified FAISS index from PDFs + FAQ
├─ data/
│  ├─ shipcube.db           # SQLite DB (created at runtime)
│  ├─ qna.json              # FAQ knowledge base
│  ├─ pdfs/                 # Source PDFs
│  └─ global_kb/            # FAISS vector store (PDF + FAQ)
├─ templates/
│  ├─ index.html            # Main chat UI
│  ├─ invoice.html          # Invoice page (extends layout)
│  ├─ login.html            # Login / register page (not shown above)
│  └─ dashboard.html        # Simple dashboard placeholder
└─ static/
   ├─ css/style.css         # Styles
   ├─ js/chat.js            # Chat + FAQ + feedback logic
   └─ images/shipcube_logo.png

Flowchart
                           ┌────────────────────── ───┐
                           │   User (Web Browser)     │
                           │ Chat UI + FAQ Sidebar    │
                           └───────────────┬──────────┘
                                           │
                                           ▼
                                 Front-End Logic (chat.js)
                                 - Captures query
                                 - Shows typing indicator
                                 - Sends /ask & renders reply
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Flask Backend (app.py)│
                               └───────────┬───────────┘
                                           │
                     ┌──────────────────────────────────────────────────┐
                     │  Intelligent Routing in `/ask`                   │
                     │  • Pricing Guard (login required)                │
                     │  • Order Detection → DB lookup (`client_orders`) │
                     │  • Small-talk → Friendly Gemini reply            │
                     │  • Otherwise go to RAG pipeline                  │
                     └──────────────────────────────────────────────────┘
                                           │
                                           ▼
                          RAG Pipeline (utils/ai_model.py)
     ┌────────────────────────────────────────────────────────────────────────┐
     │ 1️⃣ Direct FAQ match? → Gemini summarised FAQ answer                    │
     │ 2️⃣ If not: Vector search in FAISS (PDF + FAQ global_kb)                │
     │ 3️⃣ Build short context from best matches                               │
     │ 4️⃣ Pass “Context + Question” → Gemini (Google GenAI API)               │
     │ 5️⃣ Return concise answer + top sources                                 │
     └─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                ┌────────────────────────────────────────────┐
                │ Save chat + summary to SQLite (chats)      │
                │ Return answer + source + message_id        │
                └────────────────────────────────────────────┘
                                           │
                                           ▼
            UI Appends Assistant Message + 👍/👎 Feedback Buttons
                                           │
                                           ▼
                    Feedback → /feedback → Stored in SQLite


──────────────────────────────────────────────────────────────────────────────
Supporting Components
──────────────────────────────────────────────────────────────────────────────
• DB Initialization: users, client_orders, chats, feedback  
• Authentication: login/register, access to invoices & pricing data  
• build_global_kb.py: offline FAISS index creation (PDF + FAQ)  
• Prometheus Metrics: /metrics (requests, latency, /ask count)

