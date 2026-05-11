# Ibe AI 🧠

### *Wisdom, instantly.*

## Live Demo
**API:** https://ibe-ai.onrender.com
**Swagger Docs:** https://ibe-ai.onrender.com/docs

Ibe is a production-ready Agentic RAG API built with FastAPI and Google Gemini. Upload an Excel file of insurance policy data and ask questions in plain English — Ibe's agent decides how to answer using the right tool for every query. 

**Ibe** — *wisdom/intellect* in Igala, spoken in Kogi State, Nigeria. Named to reflect the system's purpose: turning raw data into instant, grounded knowledge.


## What Makes Ibe Different

Most RAG systems do one thing: retrieve chunks and send them to an LLM. Ibe goes further with a **three-tool agentic architecture** — the agent reads your query, decides which tool is most appropriate, runs it, and returns a structured response with confidence scoring and source citations.

| Query Type | Tool Used | Example |
|---|---|---|
| Client lookups, policy details | `SemanticSearchTool` | *"Tell me about Adaeze Okonkwo's policy"* |
| Numbers, averages, statistics | `AnalyticsTool` | *"What is the average claim amount for Life insurance?"* |
| Overviews, patterns, groups | `SummaryTool` | *"Summarise all high-risk clients in Lagos"* |

---

## Architecture

```
Excel Upload (POST /ingest)
        │
        ▼
┌─────────────────────────────────────┐
│  Data Validator + Row Processor     │
│  Each row → rich text document      │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  Gemini           Pandas
  Embeddings       DataFrame
        │             │
        ▼             │
   ChromaDB ◄─────────┘
   Vector Store    (held in memory
                    for analytics)

User Query (POST /chat)
        │
        ▼
┌─────────────────────────────────────┐
│  Ibe Agent — Router                 │
│  Gemini decides: which tool?        │
└──────┬──────────┬───────────────────┘
       │          │           │
       ▼          ▼           ▼
  Semantic    Analytics    Summary
  Search      Tool         Tool
  Tool        │            │
  │           ▼            ▼
  │      Pre-computed   Top-8
  │      DataFrame      Semantic
  │      Stats          Results
       │          │           │
       └──────────┴───────────┘
                  │
                  ▼
        Structured Response
        {answer, tool_used,
         confidence, sources}
```

---

## Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/ingest` | Upload Excel file and index all records | Yes |
| POST | `/chat` | Ask Ibe a question | Yes |
| DELETE | `/chat/history/{session_id}` | Clear conversation memory | Yes |
| GET | `/health` | System health + record count | No |
| GET | `/docs` | Swagger UI | No |
| GET | `/redoc` | ReDoc documentation | No |

---

## Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/JaneAjodo/ibe-ai.git
cd ibe-ai/part_2

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Build and run
docker build -t ibe-ai .
docker run -p 8000:8000 --env-file .env ibe-ai

# 4. Open docs
open http://localhost:8000/docs
```

### Option B — Local Python

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run
uvicorn src.main:app --reload --port 8000

# 5. Open docs
open http://localhost:8000/docs
```

---

## Usage

### Step 1 — Ingest data

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: ibe-secret-key-2025" \
  -F "file=@sample_data.xlsx"
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully ingested 100 policy records from 'sample_data.xlsx'",
  "records_ingested": 100,
  "collection_name": "ibe_insurance_policies",
  "timestamp": "2025-04-05T10:00:00"
}
```

---

### Step 2 — Ask questions

**Semantic query — client lookup:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: ibe-secret-key-2025" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Adaeze Okonkwo policy", "session_id": "demo"}'
```

**Analytical query — statistics:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: ibe-secret-key-2025" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the average claim amount for Life insurance policies?", "session_id": "demo"}'
```

**Summary query — overview:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: ibe-secret-key-2025" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise all high risk clients in Lagos", "session_id": "demo"}'
```

**Follow-up question (conversation memory):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: ibe-secret-key-2025" \
  -H "Content-Type: application/json" \
  -d '{"query": "What about their claim status?", "session_id": "demo"}'
```

---

### Response Structure

```json
{
  "answer": "Based on the policy records, Adaeze Okonkwo holds policy IBE-2023-0002...",
  "tool_used": "semantic_search",
  "confidence": "high",
  "sources": [
    {
      "policy_id": "IBE-2023-0002",
      "client_name": "Adaeze Okonkwo",
      "policy_type": "Health",
      "region": "Lagos",
      "relevance_score": 0.912
    }
  ],
  "session_id": "demo",
  "timestamp": "2025-04-05T10:01:00"
}
```

---

## Sample Questions to Try

| Category | Question |
|---|---|
| Client lookup | `"Find the policy for Babatunde Ikenna"` |
| Claim details | `"Which policies have overdue claims?"` |
| Analytics | `"What is the total premium revenue?"` |
| Analytics | `"Which region has the highest average risk score?"` |
| Analytics | `"How many claims are currently pending?"` |
| Summary | `"Summarise all Business insurance policies"` |
| Summary | `"Give me an overview of policies in Kano"` |
| Follow-up | `"What are their claim amounts?"` *(after any previous query)* |

---

## Project Structure

```
part_2/
├── src/
│   ├── main.py              # FastAPI app, middleware, routers
│   ├── core/
│   │   ├── config.py        # All settings and environment config
│   │   ├── logging.py       # Structured logging setup
│   │   └── security.py      # API key authentication
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── routers/
│   │   ├── ingest.py        # POST /ingest — Excel ingestion
│   │   ├── chat.py          # POST /chat — Agentic Q&A
│   │   └── health.py        # GET /health — System status
│   ├── tools/
│   │   ├── agent.py         # Ibe Agent — router + orchestrator
│   │   └── agent_tools.py   # SemanticSearch, Analytics, Summary tools
│   └── utils/
│       ├── vector_store.py  # ChromaDB operations + Gemini embeddings
│       └── memory.py        # Conversation history per session
├── sample_data.xlsx         # Ibe AI sample dataset — 100 Nigerian insurance policy records
├── Dockerfile               # Container setup
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST endpoints, Swagger docs, validation |
| LLM | Gemini 1.5 Flash | Routing, tool execution, answer generation |
| Embeddings | Google embedding-001 | Semantic vector generation |
| Vector Store | ChromaDB | Local vector persistence + similarity search |
| Data Processing | Pandas | DataFrame operations for analytics |
| Auth | API Key Header | Request authentication |
| Container | Docker | Reproducible deployment |
| Config | python-dotenv | Environment variable management |

---

## Production Upgrade Path

Ibe is architecturally production-ready. The clear upgrade path for scale:

| Current (Local) | Production (GCP) |
|---|---|
| ChromaDB local | BigQuery Vector Search |
| Gemini AI Studio key | Vertex AI managed Gemini |
| In-memory conversation | Redis / Firestore session store |
| Single FastAPI process | Cloud Run auto-scaling |
| .env file | GCP Secret Manager |
| Manual logging | Cloud Logging + Monitoring |

---

## Security

- All `/ingest` and `/chat` endpoints require `X-API-Key` header
- API key configurable via `IBE_API_KEY` environment variable
- User inputs sanitised before entering prompts
- LLM grounded strictly to ingested data — no external knowledge injection
- `.env` file gitignored — keys never committed to source control

---

## Known Limitations & Trade-offs

- **ChromaDB is local** — resets when the container restarts. Production deployment should use a persistent vector store or mount a volume.
- **Analytics tool uses pre-computed stats** — for very large datasets (10k+ records), a proper SQL layer would be more efficient than pandas in-memory operations.
- **Conversation memory is in-process** — sessions are lost on restart. Redis would solve this in production.
- **No streaming** — responses are synchronous. FastAPI streaming with `StreamingResponse` would improve perceived latency for long answers.

---

## Get Your Gemini API Key

1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API Key**
3. Copy and paste into your `.env` file


---

*Built by Jane Ajodo — AI/ML Engineer*  
*GitHub: [github.com/JaneAjodo](https://github.com/JaneAjodo)*
