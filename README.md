# ⚡ Wattmonk RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** based AI chatbot that intelligently 
answers questions from two specialized knowledge sources — Wattmonk company 
information and NEC 2017 Solar PV electrical code standards.

Built as part of the **Wattmonk AI Intern Assignment**.

---

## 🚀 Live Demo
> Deployed URL: https://rag-chatbot-drpbfvjzayswttlnw6zzsa.streamlit.app/

---

## 🧠 What is RAG?
RAG (Retrieval-Augmented Generation) is an AI technique that retrieves 
relevant context from a knowledge base before generating a response. 
This allows the AI to answer questions grounded in specific documents 
rather than relying only on its training data.

---

## 🏗️ Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Frontend     | Streamlit                         |
| Backend      | FastAPI + Uvicorn                 |
| LLM          | Llama 3.3 70B via Groq API        |
| RAG Pipeline | Custom keyword-based retrieval    |
| HTTP Client  | httpx                             |
| Config       | python-dotenv                     |
| Deployment   | Docker + Docker Compose           |

---

## 📚 Knowledge Base

| Source         | Content                                      |
|----------------|----------------------------------------------|
| Wattmonk Info  | Company overview, all 6 services, Zippy tech, market stats |
| NEC 2017       | Article 690 (Solar PV), voltage rules, circuit sizing, overcurrent protection |
| General Solar  | Installation workflow, acronyms, project types |

---

## ⚙️ RAG Pipeline

1. **Intent Classification** — Detects if query is about Wattmonk, NEC, or General
2. **Text Chunking** — Knowledge base split into 800-char chunks with 100-char overlap
3. **Chunk Retrieval** — Top-5 most relevant chunks retrieved via keyword scoring
4. **Context Injection** — Retrieved chunks injected into the LLM system prompt
5. **Response Generation** — Llama 3.3 70B generates a grounded, cited response

---

## ✨ Features

- 💬 Multi-turn conversation with memory
- 🎯 Automatic intent detection (Wattmonk / NEC / General)
- 📚 Source attribution badge on every response
- 🔍 Retrieved chunks viewer to inspect RAG results
- 📊 Session statistics in sidebar
- 🌐 REST API with auto-generated Swagger docs at `/docs`
- 🐳 Docker + Docker Compose support

---

## 🗂️ Project Structure

\`\`\`
wattmonk-rag/
├── backend/
│   ├── main.py              # FastAPI app + RAG pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
\`\`\`

---

## 🚀 Quick Start

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/ykhushbu941/RAG-Chatbot.git
cd wattmonk-rag
\`\`\`

### 2. Set up environment variables
\`\`\`bash
cp .env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY
\`\`\`

### 3. Run the Backend
\`\`\`bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
\`\`\`

### 4. Run the Frontend
\`\`\`bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
\`\`\`

### 5. Open in browser
\`\`\`
http://localhost:8501
\`\`\`

---

## 🐳 Run with Docker

\`\`\`bash
docker-compose up --build
\`\`\`

---

## 🔑 Environment Variables

| Variable       | Required | Description              |
|----------------|----------|--------------------------|
| GROQ_API_KEY   | ✅ Yes   | Get free key at console.groq.com |
| API_URL        | No       | Backend URL (default: http://localhost:8000) |

---

## 📡 API Endpoints

| Method | Endpoint    | Description                        |
|--------|-------------|------------------------------------|
| GET    | /           | Health check                       |
| GET    | /health     | API status + model info            |
| POST   | /chat       | Main RAG chat endpoint             |
| POST   | /retrieve   | Debug — inspect chunk retrieval    |
| GET    | /docs       | Swagger UI (auto-generated)        |

---

## 🙋 Author
**Khushbu Yadav**  
ykhushbu941@gmail.com | (https://www.linkedin.com/in/khushbu-yadav-500174317/)
```

---

## GitHub Topics/Tags to Add
When uploading, add these tags to make your repo discoverable:
```
rag  fastapi  streamlit  llm  groq  llama  solar  python  
chatbot  retrieval-augmented-generation  nec  ai  machine-learning
