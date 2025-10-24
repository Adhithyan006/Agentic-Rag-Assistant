# Agentic RAG Assistant

Small local repository for a Retrieval-Augmented Generation (RAG) assistant built in Python.

Contents:
- `main.py`, `real_rag.py`, `rag_system.py` - core Python modules
- `documents/` - sample text documents used to build the vector DB
- `chroma_db/` - local Chroma DB (excluded from git via .gitignore)
- `index.html` - simple HTML frontend

Quick start
1. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

2. Set required environment variables (openAI key etc.) in a local `.env` file — do not commit secrets to git.

3. Run the app:

   python main.py

Notes
- The `chroma_db/` directory contains local database files and is intentionally excluded from version control.
- Add a LICENSE if you want to publish this repository publicly.
