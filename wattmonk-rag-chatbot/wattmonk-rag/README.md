# ⚡ Wattmonk RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **FastAPI** (backend) and **Streamlit** (frontend), powered by **Anthropic Claude**. Answers questions about Wattmonk company info and NEC 2017 solar PV code.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│               STREAMLIT FRONTEND  (port 8501)                    │
│  • Chat UI with message history                                   │
│  • Source attribution badges                                      │
│  • Retrieved chunks viewer                                        │
│  • Session stats & debug panel                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │  POST /chat  (JSON)
┌───────────────────────────▼─────────────────────────────────────┐
│               FASTAPI BACKEND  (port 8000)                       │
│                                                                   │
│  RAG Pipeline:                                                    │
│  1. Intent Classification  → wattmonk | nec | both | general     │
│  2. Chunk Retrieval        → keyword scoring, top-K chunks        │
│  3. Context Injection      → chunks + history → system prompt     │
│  4. Claude API Call        → claude-sonnet-4-20250514             │
│  5. Response + Attribution → reply + source label                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTPS
┌───────────────────────────▼─────────────────────────────────────┐
│               ANTHROPIC API  (claude-sonnet-4-20250514)          │
└─────────────────────────────────────────────────────────────────┘
```

### Knowledge Base Sources
| Source | Content | Chunks |
|--------|---------|--------|
| `wattmonk` | Company info, services, Zippy, stats, culture | ~10 chunks |
| `nec` | NEC 2017 Article 690, 705, 250 (solar PV) | ~16 chunks |
| `general` | Solar workflow, acronyms, project types | ~4 chunks |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Anthropic API key ([get one here](https://console.anthropic.com/))

### 1. Clone / Extract the project
```bash
cd wattmonk-rag
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Install and run the **Backend** (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Backend will be available at: `http://localhost:8000`  
API docs (Swagger): `http://localhost:8000/docs`

### 4. Install and run the **Frontend** (Streamlit)
Open a new terminal:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Frontend will be available at: `http://localhost:8501`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | API status + knowledge sources |
| `GET` | `/knowledge-base/stats` | KB character counts, chunk counts |
| `POST` | `/chat` | Main RAG chat endpoint |
| `POST` | `/retrieve?query=...` | Debug: see chunk retrieval results |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### POST /chat — Request
```json
{
  "message": "What is the maximum voltage for residential PV systems?",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

### POST /chat — Response
```json
{
  "reply": "According to NEC 2017 Article 690.7...",
  "intent": "nec",
  "chunks_used": 3,
  "retrieved_chunks": [
    {
      "source": "nec",
      "score": 8.0,
      "preview": "690.7 MAXIMUM VOLTAGE: The maximum voltage of PV system..."
    }
  ],
  "source_label": "NEC 2017"
}
```

---

## 🔧 RAG Pipeline Details

### 1. Intent Classification
Keywords are scored against two sets:
- **NEC keywords**: "690", "ampacity", "overcurrent", "grounding", "voltage", etc.
- **Wattmonk keywords**: "wattmonk", "planset", "proposal", "zippy", "pto", etc.

### 2. Chunk Retrieval
- Knowledge base text is split into **800-character chunks** with **100-character overlap**
- Each chunk is scored by counting keyword matches from the user query
- Top-5 highest-scoring chunks are retrieved

### 3. Context Injection
Retrieved chunks are injected into the Claude system prompt:
```
RETRIEVED CONTEXT FROM KNOWLEDGE BASE:

[Source: NEC | Relevance Score: 8]
690.7 MAXIMUM VOLTAGE: The maximum voltage of PV system dc circuits...

---

[Source: NEC | Relevance Score: 5]
690.8 CIRCUIT SIZING AND CURRENT...
```

### 4. Response Generation
- Model: `claude-sonnet-4-20250514`
- Last 8 messages of conversation history are included
- Max tokens: 1000

---

## 🚢 Deployment

### Deploy on Railway
```bash
# Backend
railway new
railway add --service backend
cd backend && railway up

# Frontend
railway add --service frontend
cd frontend && railway up
```

### Deploy on Streamlit Cloud
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `frontend/app.py` as main file
4. Add `API_URL` to Streamlit secrets

### Deploy on Heroku
```bash
# Backend
heroku create wattmonk-rag-api
heroku config:set ANTHROPIC_API_KEY=your_key
cd backend && git push heroku main

# Frontend
heroku create wattmonk-rag-ui
heroku config:set API_URL=https://wattmonk-rag-api.herokuapp.com
cd frontend && git push heroku main
```

### Deploy with Docker
```bash
# Build and run both services
docker-compose up --build
```

---

## 📁 Project Structure

```
wattmonk-rag/
├── backend/
│   ├── main.py              # FastAPI app + RAG pipeline + knowledge base
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── requirements.txt
├── .env.example             # Environment variable template
└── README.md
```

---

## 🎯 Features

### Core (MVP)
- [x] Multi-context query handling (Wattmonk / NEC / General)
- [x] Source attribution on every response
- [x] Conversation memory (last 8 messages)
- [x] Graceful error handling with user feedback
- [x] RAG chunk retrieval with relevance scoring
- [x] FastAPI with auto-generated Swagger docs

### UI Features
- [x] Suggested questions for first-time users
- [x] Retrieved chunks viewer (toggle on/off)
- [x] Session statistics (query count, last intent)
- [x] API health indicator in sidebar
- [x] Debug mode toggle
- [x] Clear chat button
- [x] Knowledge base explorer in sidebar

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | — | Your Anthropic API key |
| `PORT` | No | `8000` | FastAPI server port |
| `API_URL` | No | `http://localhost:8000` | Backend URL for Streamlit |

---

## 📝 Submission

**GitHub Repository:** [your-repo-url]  
**Deployed Application:** [your-deployed-url]  
**Demo Video:** [your-video-url]

Built for: AI Intern Assignment — Wattmonk Technologies
