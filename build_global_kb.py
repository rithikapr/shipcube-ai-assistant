# build_global_kb.py  (new file – you can reuse the old name if you like)

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

PDF_DIR = Path("data/pdfs")
KB_DIR = Path("data/global_kb")   # NEW: unified index
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QNA_JSON = Path("data/qna.json")

def load_qna_docs_from_json():
    """Load data/qna.json and convert each item to a short Document."""
    if not QNA_JSON.exists():
        print(f"[faq] No qna.json found at {QNA_JSON}, skipping FAQ docs.")
        return []

    with QNA_JSON.open("r", encoding="utf-8") as f:
        qna_list = json.load(f)

    docs = []
    for item in qna_list:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue

        text = f"Q: {q}\nA: {a}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "faq",
                    "question": q,
                },
            )
        )

    print(f"[faq] Loaded {len(docs)} FAQ docs from qna.json")
    return docs


def load_all_pdfs(pdf_dir: Path):
    docs = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"[load-pdf] {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        for d in pdf_docs:
            d.metadata.setdefault("source_type", "pdf")
            d.metadata.setdefault("source_file", pdf_path.name)
        docs.extend(pdf_docs)
    print(f"Total PDF pages loaded: {len(docs)}")
    return docs


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load sources
    raw_pdf_docs = load_all_pdfs(PDF_DIR)          # pages from all PDFs
    faq_docs = load_qna_docs_from_json()           # curated FAQ from qna.json

    if not raw_pdf_docs and not faq_docs:
        print("[kb] No PDFs or FAQ entries found. Nothing to build.")
        return

    # 2. Split only the PDF docs (FAQ docs are already short)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "?", "!", " "],
    )
    print("[split] Splitting PDF pages into chunks...")
    pdf_chunks = splitter.split_documents(raw_pdf_docs)
    print(f"[split] Total PDF pages: {len(raw_pdf_docs)}, chunks: {len(pdf_chunks)}")

    # Remove super-short chunks that are probably just headings or noise
    cleaned_chunks = []
    for d in pdf_chunks:
        text = d.page_content.strip()
        # Skip if it's too short or only a few words (tune these thresholds as you like)
        if len(text) < 40 or len(text.split()) < 5:
            continue
        cleaned_chunks.append(d)

    print(f"[clean] Kept {len(cleaned_chunks)} chunks after removing tiny/heading-only chunks.")
    pdf_chunks = cleaned_chunks

    # 3. Combine PDF chunks + FAQ docs into a single corpus
    all_docs = pdf_chunks + faq_docs
    print(f"[kb] Total docs to embed (PDF + FAQ): {len(all_docs)}")

    # 4. Build vector store
    print("[embed] Building unified vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = FAISS.from_documents(all_docs, embeddings)

    # 5. Save
    print(f"[save] Saving vector store to {KB_DIR}")
    vectordb.save_local(str(KB_DIR))
    print("[kb] Done.")
   
    

if __name__ == "__main__":
    main()
