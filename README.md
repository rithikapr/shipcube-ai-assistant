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

<img width="449" height="436" alt="image" src="https://github.com/user-attachments/assets/2fef526f-ca9a-4139-8e49-bdb757893774" />

Supporting Components
• DB Initialization: users, client_orders, chats, feedback  
• Authentication: login/register, access to invoices & pricing data  
• build_global_kb.py: offline FAISS index creation (PDF + FAQ)  
• Prometheus Metrics: /metrics (requests, latency, /ask count)

