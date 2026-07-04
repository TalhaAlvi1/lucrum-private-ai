# Lucrum Private AI

**A fully offline, RAG-powered desktop assistant built with Streamlit — upload documents, build a local vector store, and chat with your files without any data leaving the machine.**

<img width="1539" height="805" alt="image" src="https://github.com/user-attachments/assets/9f4f141f-bd31-4da5-9b84-5023506c93e8" />


## Features

- **Offline RAG chat** — upload up to 5 files (PDF / CSV / TXT), auto-chunked and embedded into a local ChromaDB vector store
- **Local LLM inference** — runs quantized GGUF models via `llama-cpp-python`, CPU-only, no external API calls
- **Branded model selector** — Lucrum Pro / Thinking / Instant / Auto dropdown maps internally to different local models, so no third-party model names are exposed in the UI
- **Semantic search + source attribution** — `sentence-transformers` embeddings with cosine similarity, responses cite the source document
- **Streaming chat responses** with persistent session history
- **Configurable system prompt** — identity, tone, and feature descriptions driven by `properties.json`
- **Native desktop windows** — `pywebview` wraps the Streamlit app and the CRM into standalone app windows
- **Integrated CRM** — bundled EspoCRM instance (Docker) launched and managed alongside the AI assistant
- **One-command orchestration** — `main.py` starts/stops both the AI server and CRM container together

## Architecture
<img width="963" height="654" alt="image" src="https://github.com/user-attachments/assets/cececa1a-cb1f-4f30-864e-3521ed0f8a51" />


## Tech Stack

`Streamlit` · `llama-cpp-python` · `ChromaDB` · `sentence-transformers` · `PyPDF2` · `pandas` · `pywebview` · `customtkinter` · `EspoCRM (Docker)`

## Project Structure

```
lucrum-private-ai/
├── lucrum_ai.py          # Main Streamlit RAG chat app
├── main.py               # Orchestrator: starts/stops AI + CRM servers
├── helpers.py            # Server lifecycle + UI helper functions
├── document_handler.py   # Document query / retrieval logic
├── help_window.py        # Desktop help & troubleshooting window
├── window_ai.py          # pywebview wrapper for the AI app
├── window_crm.py         # pywebview wrapper for the CRM
├── lucrum_crm.py         # CRM Docker container launcher
├── espo.py               # EspoCRM Docker start/stop utility
├── properties.json       # System prompt & branding configuration
├── config/
│   └── model_config.json # Model name → internal model mapping
├── docs/                 # Architecture & UI diagrams (used in this README)
├── avatar.png / icon.ico / Benne-Regular.otf   # Branding assets
└── requirements.txt
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/<your-username>/lucrum-private-ai.git
cd lucrum-private-ai
pip install -r requirements.txt
```

**2. Add a local LLM**

Download a GGUF-quantized model (e.g. from Hugging Face) and place it in the project root. Update the filenames in `load_model()` inside `lucrum_ai.py` to match your model file(s).

**3. (Optional) Set up the CRM**

The CRM requires Docker and a `docker-compose.yml` for EspoCRM placed under `docker/`. Skip this step if you only need the AI chat assistant.

**4. Run**

Full app (AI + CRM, native windows):
```bash
python main.py
```

Streamlit AI assistant only:
```bash
streamlit run lucrum_ai.py
```

- AI Assistant → `http://127.0.0.1:8501`
- CRM → `http://127.0.0.1:8080`

## Usage

1. Select a model from the sidebar dropdown
2. Upload up to 5 PDF/CSV/TXT files
3. Wait for processing (chunking + embedding into the vector store)
4. Ask questions — answers are generated from your documents, with sources cited

## ▶️ Demo

https://github.com/user-attachments/assets/1303d5ff-647d-442e-833e-d250af7ccb63


## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---
