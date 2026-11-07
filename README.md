@'
# RAGForge

Ask questions about a PDF and get answers grounded only in that document. RAGForge uses hybrid retrieval (vector + BM25), FlashRank reranking and a Groq-hosted LLM, behind a FastAPI backend and a Streamlit chat UI. It ships with a DeepEval suite for offline evaluation and is set up for online evaluation with LangSmith.

## Features

- Hybrid retrieval: Chroma vector search plus BM25, merged by an ensemble retriever
- FlashRank reranking (top 3 of the merged candidates)
- Streaming answers from Groq (`qwen/qwen3.8-27b`)
- LangGraph pipeline with per-thread history and follow-up question rewriting
- Caching of parsed PDFs, chunks, embeddings and answers
- Privacy guard: requests for phone, email, address, date of birth or ID numbers are refused, and emails and phone numbers are redacted from answers
- Offline evaluation with DeepEval (retrieval, generation, safety, operations)
- Online evaluation with LangSmith tracing
- Docker Compose setup for the API and the frontend

## Architecture

```
PDF upload -> load -> split (1000 chars, 150 overlap) -> embed (Chroma) + BM25
question -> rewrite using history -> hybrid retrieve (k=5 each) -> FlashRank (top 3) -> LLM -> answer
```

## Project structure

```
mainapi.py              FastAPI app (POST /query/stream, docs at /docs)
Rag.py                  RAG pipeline (LangGraph)
frontend/app.py         Streamlit chat UI
MCP_RAG/                MCP server (experimental)
Rag_evals/              DeepEval scripts (see Evaluation)
docs_synth/, golden_dataset.*, retrieval_eval_final.py
                        earlier retrieval-evaluation experiments
Dockerfile, docker-compose.yml, requirements.txt
```

## Configuration

Create a `.env` file in the project root. Never commit it.

```
GROQ_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=ragforge
```

The frontend reads `API_URL` (default `http://127.0.0.1:8000`).

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m uvicorn mainapi:app --reload
```

In a second terminal:

```powershell
python -m streamlit run frontend\app.py
```

Open http://localhost:8501, upload a PDF and ask a question. API docs: http://localhost:8000/docs.

## Run with Docker

```powershell
docker compose build
docker compose up -d
```

Frontend: http://localhost:8501. API: http://localhost:8000/docs. The first request downloads the embedding and reranker models, so it is slow. They are cached in a Docker volume.

---

## Evaluation

RAGForge is evaluated in two ways:

| Type | When it runs | Data | Tooling |
|---|---|---|---|
| Offline | On your machine, before a release or after changing the pipeline | A fixed question set (some with reference answers) | DeepEval scripts in `Rag_evals/` |
| Online | Continuously, on real traffic | Live traces, sampled | LangSmith online evaluators |

Offline evals tell you whether a change made the system better or worse. Online evals tell you whether it is behaving well for real users. Use both.

A Groq model acts as the judge in the offline scripts, so no OpenAI key is needed. LLM judges are noisy: compare averages over many questions, and treat differences under about 0.1 on a small set as noise.

### What each eval measures

The pipeline has two stages, retrieval and generation, so the metrics are split to show which stage is at fault.

#### 1. Retrieval quality (`Rag_evals/contextual.py`)

| Metric | Question it answers | Needs reference answer |
|---|---|---|
| Contextual Precision | Are the relevant chunks ranked above the irrelevant ones? | Yes |
| Contextual Recall | Do the retrieved chunks contain everything needed for the correct answer? | Yes |
| Contextual Relevancy | How much of the retrieved text is actually on topic? | No |

Low precision means the reranker or ranking is weak. Low recall means the right chunk was never retrieved (raise `k`, change chunk size or the embedding model). Low relevancy means chunks are too large or noisy.

#### 2. Generation quality, reference-free (`Rag_evals/faithfulness.py`)

| Metric | Question it answers |
|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved context? Catches hallucination. |
| Answer Relevancy | Does the answer address the question, without padding or off-topic text? |

The judge extracts claims or statements from the answer and checks each one, so no hand-written answer is needed. This scales to many questions cheaply.

#### 3. Answer quality against a reference (`Rag_evals/geval.py`)

Three GEval metrics, each with a written criterion and a 0 to 10 scoring rubric that is normalized to 0 to 1:

| Metric | Compares | Checks |
|---|---|---|
| Correctness | answer vs reference | Facts match; no invented details; "not in the document" answers are rewarded when correct |
| Completeness | answer vs reference | Every key point and every part of the question is covered |
| Style | answer only | Clear plain-text prose, no markdown, no padding, respects length limits |

#### 4. Safety (`Rag_evals/safety_eval.py`)

Adversarial questions, each tagged with the checks that apply. Scores are reported so that 1.0 is safe.

| Check | Probes | Pass means |
|---|---|---|
| Toxicity | Requests to roast, insult or discriminate against the candidate | The answer stays civil or refuses |
| PII Protection | Requests for phone, email, address, date of birth, government IDs, or "how can I find this person" | No identifier is revealed |
| Scope Adherence | "Ignore previous instructions", persona changes, system-prompt reveal, coding or trivia questions, hiring opinions | The assistant stays a document Q&A assistant |

A normal control question is included to confirm the safety layer does not break regular use.

#### 5. Operations (`Rag_evals/ops_evals.py`)

Custom DeepEval metrics that need no judge model, so they cost no Groq tokens for scoring.

| Metric | Measured with | Default threshold |
|---|---|---|
| Latency | `completion_time` of each query | 5 seconds |
| Cost | token usage times the per-million-token prices you set | $0.002 per query |
| Reliability | Status of each run: ok, rate limited, LLM error, empty, false refusal | 100% ok |

Set `PRICE_IN_PER_M` and `PRICE_OUT_PER_M` in the file from Groq's pricing page, or cost always shows $0.

### Run the offline evals

Put a test PDF at `Rag_evals/data/resume.pdf` (PDFs are git-ignored) and run from the project root:

```powershell
python -m Rag_evals.contextual
python -m Rag_evals.faithfulness
python -m Rag_evals.geval
python -m Rag_evals.safety_eval
python -m Rag_evals.ops_evals
```

`contextual.py` creates `Rag_evals/goldens.json` on its first run. Open it and delete bad questions before trusting the scores. It is git-ignored because it can contain private document details.

### Reading the results

| Symptom | Likely cause | What to change |
|---|---|---|
| Low Contextual Recall | Right chunk not retrieved | Raise `k`, change chunk size, try a better embedding model |
| Low Contextual Precision | Good chunk ranked too low | Stronger reranker, tune ensemble weights |
| Low Contextual Relevancy | Chunks too large or noisy | Smaller chunks, keep fewer chunks |
| Low Faithfulness | Answer adds facts not in context | Tighten the prompt, lower temperature |
| Low Answer Relevancy | Padding or off-topic text | Shorten the prompt rules, lower the output cap |
| Low Completeness, good Correctness | Missing parts of multi-part questions | Retrieve more chunks, improve question rewriting |
| PII or Scope failures | Prompt or guard gaps | Extend the privacy rules and the keyword guard |
| High latency only on later questions | Groq rate limiting, not the pipeline | Pace the eval or use a higher tier |

### Sample results

Measured on a one-page template resume. The sample is tiny, so treat these as a smoke test, not a benchmark.

| Metric | Result | Questions |
|---|---|---|
| Contextual Precision / Recall | 1.0 / 1.0 | 2 |
| Faithfulness | 1.00 | 5 |
| Answer Relevancy | 0.96 | 5 |
| PII Protection (after the privacy guard) | 1.00 | 4 |
| Scope Adherence | about 0.9 | 10 |
| Latency (unthrottled) | about 0.5 to 2 s per query | 20 |

Contextual Relevancy was low (0.1 to 0.7) because 1000-character chunks contain more than the question needs. Before the privacy guard, PII Protection was about 0.5.

---

## Online evaluation with LangSmith

Offline evals use a fixed question set. Online evals score real traffic as it arrives, so they catch problems your test set never covered, such as new question types, drift in retrieval quality, new jailbreak attempts and rising error rates.

`Rag.py` already imports LangSmith and wraps `run_query` with `@traceable`. Each traced call records the input question, the output (including the retrieved chunks), latency, token usage and errors.

### Set up

1. Create a LangSmith account and an API key.
2. Add the keys to `.env` (see Configuration), then restart the API.
3. Send some questions through the app. A project named `ragforge` appears in LangSmith with one trace per question.

### Add online evaluators

In LangSmith, open the project and add an automation rule with an online evaluator. Menu names change between versions, so check the current LangSmith docs if a label differs. For each rule you choose:

- a filter (for example, all root runs, or only runs with errors)
- a sampling rate (for example 10 to 20 percent, to control judge cost)
- an evaluator: an LLM-as-judge prompt, or a custom code evaluator
- a mapping from the trace fields (input, output, retrieved context) to the evaluator's variables

LLM-as-judge evaluators need a model provider key configured in your LangSmith workspace settings.

Suggested evaluators for RAGForge:

| Evaluator | Type | Uses | Flags |
|---|---|---|---|
| Faithfulness | LLM-as-judge | answer + retrieved chunks | Claims not supported by the document |
| Answer relevancy | LLM-as-judge | question + answer | Off-topic or padded answers |
| PII leak | LLM-as-judge or regex | answer | Phone numbers, emails, addresses in output |
| Refusal rate | Code | answer text | Rate of "not related to the document" replies, which often means a retrieval miss |
| Latency and errors | Built in | trace metadata | Slow runs, rate-limit failures |

Use the same criteria as the offline GEval prompts so offline and online scores are comparable.

### Close the loop

1. Filter online results for low scores or user-reported problems.
2. Add those traces to a LangSmith dataset, or copy the questions into the offline question sets.
3. Fix the pipeline, re-run the offline evals, and confirm the fix before shipping.

### Notes

- The Streamlit app calls `/query/stream`, which uses `stream_query`. Only `run_query` is decorated with `@traceable`, so streamed requests are not traced until you trace that path too. Check the LangSmith docs for tracing async generators before changing it.
- Traces contain the question, the retrieved document text and the answer. If your PDFs are confidential, set `LANGSMITH_TRACING=false` or use a self-hosted LangSmith.
- Online judges cost tokens. Keep the sampling rate low.

## Known limits

- The free Groq tier for this model allows about 8K tokens per minute and 200K per day. Many quick questions get throttled, which inflates measured latency. Pace the eval or upgrade the tier.
- The built-in DeepEval Toxicity metric returned unreliable scores with the Qwen judge. Prefer a GEval-based toxicity check where high means safe.
- Memory: torch, the embedding model and the reranker need about 2 GB of RAM.
- Uploaded PDFs and the vector store live on local disk. Attach a persistent disk when deploying.

## Security

- Keep `.env`, uploaded PDFs and `chroma_store/` out of Git, and rotate keys that were ever committed.
- Document text is sent to Groq, and to LangSmith if tracing is on. Only upload documents you are allowed to process.

## License

MIT
'@ | Set-Content -Encoding utf8 README.md
