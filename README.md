# RAGForge

**Ask questions about your documents. Get answers grounded strictly in what they actually say.**

RAGForge is a retrieval-augmented generation system built to avoid the two things that make most RAG demos unreliable: shallow single-method retrieval, and answers that quietly hallucinate past the source document.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [What makes the retrieval good](#what-makes-the-retrieval-good)
- [Evaluation](#evaluation)
- [Latency & reliability controls](#latency--reliability-controls)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Design constraints](#design-constraints)

---

## Why this exists

Most "chat with your PDF" projects wire up an embedding model, a vector store, and an LLM, and call it done. That works until:

- the answer needs an exact term or number the embedding model paraphrased away
- a follow-up question depends on the previous one ("what about the second point?")
- the model just answers from its own training data instead of the document
- a free-tier LLM API rate-limits you mid-conversation

RAGForge is built around solving those four problems specifically, not around demoing embeddings.

---

## Architecture

```
                    ┌─────────────────┐
                    │   PDF Upload     │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  Load & Chunk    │  (cached per thread)
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │    Embed +       │
                    │  Build Indexes   │  Chroma (dense) + BM25 (sparse)
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  Contextualize   │  rewrites follow-up questions
                    │    Question      │  using conversation history
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Retrieval │  Ensemble(vector, BM25)
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │    Rerank        │  FlashRank cross-encoder
                    │  (top-k → top-3) │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │    Generate      │  Groq LLM, retry on rate limits
                    │  (grounded only) │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │     Answer       │
                    └─────────────────┘
```

The whole flow is modeled as a **LangGraph state machine** with per-thread checkpointing, not a linear script. Each stage caches independently — re-asking a question against a document you've already processed skips redundant embedding, retrieval, and reranking work entirely.

---

## What makes the retrieval good

**Hybrid search, not just vector similarity.**
Every query hits two retrievers in parallel: a dense one (Chroma, sentence embeddings) and a sparse one (BM25), merged through an `EnsembleRetriever`. Vector search misses exact keyword/number matches; BM25 misses semantic paraphrases. Running both closes the gap either one leaves alone.

**Reranking before generation, not top-k-and-done.**
Hybrid results get passed through a FlashRank cross-encoder before anything reaches the LLM — reordering by actual relevance to the question rather than trusting the retriever's own ranking, then trimming to the top 3. Smaller, cleaner, more relevant context beats a large noisy one.

**Follow-ups get rewritten before retrieval, not after.**
"What about the second one?" only makes sense with history. Before retrieval runs, recent turns are used to rewrite ambiguous questions into standalone ones, resolving pronouns and references — so multi-turn conversations don't quietly degrade retrieval quality.

**MCP support for direct Claude integration.**
The backend exposes an MCP server (`FastMCP`), so the same retrieval pipeline is callable as a tool directly from Claude or any other MCP-compatible client — not locked behind the HTTP API.

**Rate limits handled, not just hoped around.**
LLM calls detect 429s specifically and retry with a reduced token budget and backoff, instead of failing the whole request the first time a free-tier limit gets hit.

**Traced end-to-end, not a black box.**
Every run is instrumented with LangSmith — retrieval hits, rerank scores, and generation calls are individually inspectable per thread.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Stateful, checkpointed pipeline per conversation thread |
| Retrieval | Chroma + BM25 (`EnsembleRetriever`) | Dense + sparse hybrid search |
| Reranking | FlashRank (`ms-marco-TinyBERT-L-2-v2`) | Cross-encoder relevance scoring post-retrieval |
| Embeddings | `all-MiniLM-L6-v2` | Fast, local, no external embedding API cost |
| LLM | Groq (`qwen/qwen3.8-27b`) | Low-latency inference |
| Backend | FastAPI | Concurrency-limited, global exception handling |
| Frontend | Streamlit | Fast to iterate, custom-themed |
| Integration | MCP server | Direct tool access from Claude / MCP clients |
| Observability | LangSmith | Full pipeline tracing |
| Evaluation | Ragas | Faithfulness, context precision/recall, answer relevancy scoring |

---

## Getting started

```bash
git clone https://github.com/Shreyansh-awasthi/RAGForge.git
cd RAGForge
pip install -r requirements.txt
```

Create `.env`:
```env
GROQ_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=rag-api
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Run the backend:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Run the frontend:
```bash
streamlit run app.py
```

Open the Streamlit URL, upload a PDF, ask a question.

---

## Project structure

```
RAGForge/
├── main.py            # FastAPI backend — upload, query, health endpoints
├── Rag.py             # LangGraph pipeline — retrieval, reranking, generation
├── app.py             # Streamlit frontend
├── requirements.txt
└── .env               # not committed — see .gitignore
```

---

## Evaluation

Retrieval and answer quality aren't just assumed — there's a separate evaluation script that runs the pipeline through [Ragas](https://github.com/explodinggradients/ragas) against a fixed set of question/document pairs, scoring:

- **Context precision / recall** — whether the hybrid retriever + reranker are actually surfacing the right chunks
- **Faithfulness** — whether the generated answer is actually supported by retrieved context, not hallucinated
- **Answer relevancy** — whether the answer addresses what was actually asked

This runs independently of the live API — it's a correctness check on the pipeline itself, not something that executes on every user query.

---

## Latency & reliability controls

Every stage that could stall or slow the pipeline down has an explicit limit rather than running unbounded:

- **Hard request timeout.** Each query is wrapped in `asyncio.wait_for` — if generation hangs (a stuck Groq call, a slow embedding step), the request fails cleanly with a 504 instead of hanging the connection indefinitely.
- **Concurrency cap.** A semaphore limits how many heavy queries (embedding + retrieval + generation) run at once. Extra requests queue briefly instead of competing for RAM/CPU and slowing everything down at once.
- **Context is trimmed before generation.** Retrieved chunks are capped (reranked down to the top 3, then truncated to a fixed character limit) before hitting the LLM — smaller prompts mean faster generation, not just cleaner answers.
- **Rate-limit backoff with a reduced retry budget.** On a Groq 429, the retry uses a smaller `max_tokens` and a short backoff instead of resending the same expensive request and hitting the limit again immediately.
- **Caching skips repeated work entirely.** Parsed documents, chunks, retrievers, and answers are all cached per thread (LRU-capped, not unbounded) — re-asking a question, or asking a new question against an already-processed document, skips embedding and reranking instead of redoing it from scratch.

---

## Design constraints


Deliberate limits, not oversights:

- **Answers are strictly document-grounded.** If it's not in the file, the system says so instead of guessing.
- **Bounded memory, not unbounded caching.** Document, chunk, retriever, and answer caches are LRU-capped — old sessions get evicted, not accumulated forever.
- **Capped concurrency.** A limited number of heavy queries run at once; extra requests queue briefly instead of exhausting server resources.
- **No silent failures.** A global exception handler ensures one bad request returns a clean error instead of taking the whole server down.
