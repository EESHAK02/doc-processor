# DocAgent - Intelligent Multi-Document Processor

A production-grade agentic document processing system built with **LangGraph**, **Groq Llama 3.3 70B**, and **Langfuse** observability. Upload invoices, expense reports, or booking confirmations, or even your resume - the agent classifies, extracts, validates, and lets you chat with your documents.

---

## Live Demo

> URL: [Chat with my agent here](https://doc-processor-x2dd.onrender.com)
- Note : No login required, may take a minute to wake up (Render free tier limits on inactivity)

---

## Langfuse Observability

Every document processing run and chat message is traced in your Langfuse dashboard:
- Full node-by-node traces (load - classify - extract - validate - respond)
- LLM call inputs/outputs with latency
- Per-user session tracking
- Anomaly rates and risk distributions

Go to https://cloud.langfuse.com to view your traces.

---

## Example Questions

After uploading documents:
- *"What is the total amount on the invoice?"*
- *"Are there any anomalies across all documents?"*
- *"Who is the traveler on the booking confirmation?"*
- *"Summarize all documents in one paragraph"*

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph 0.2+ |
| LLM | Groq Llama 3.3 70B (free tier) |
| Document loading | LangChain PyPDFLoader |
| Observability | Langfuse cloud (free tier) |
| UI | Gradio 4 |
| Backend | FastAPI + Uvicorn |
| Deployment | Render (free tier) |

---

## Quick Start (Local)

### 1. Clone & install
```bash
git clone https://github.com/EESHAK02/doc-processor
cd doc-processor
pip install -r requirements.txt
```

### 2. Set environment variables

Edit .env with your keys

You need:
- **Groq API key** - free at https://console.groq.com
- **Langfuse keys** - free at https://cloud.langfuse.com (create a project, copy Public + Secret keys)

### 3. Run
```bash
python main.py
```

Open http://localhost:7860 in your browser.

---

## Deploy to Render (Free Public Link)

1. Push this repo to GitHub
2. Go to https://render.com -> New -> Web Service
3. Connect your GitHub repo
4. Set these environment variables in Render dashboard:
   - `GROQ_API_KEY`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_HOST` → `https://cloud.langfuse.com`
5. Build command: `pip install -r requirements.txt`
6. Start command: `python main.py`
7. Deploy - Render gives you a public `https://your-app.onrender.com` URL


