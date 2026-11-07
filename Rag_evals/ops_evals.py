import asyncio, time, statistics
from langchain_core.callbacks import get_usage_metadata_callback
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
import asyncio, time, statistics, uuid
from Rag import (run_query, BUSY_MESSAGE, NOT_RELEVANT_MESSAGE,
                 DOC_CACHE, CHUNK_CACHE, RETRIEVER_STORE, _lru_get, _lru_set,
                 MAX_THREADS_CACHED)

PDF = r"C:\Users\ayush\OneDrive\Desktop\simple_minimalist_resume.pdf"

QUESTIONS = [
    # original 5
    "What technical tools are listed?",
    "Summarize the work experience and education.",
    "What are the core competencies?",
    "Which role reduced project delivery times by 15%?",
    "How do the responsibilities in the two roles differ?",

    # short factual lookups
    "What is the job title in the most recent role?",
    "What degree is listed in the education section?",
    "Is Microsoft Office Suite mentioned?",
    "Is CRM experience listed?",
    "What does the professional summary say?",

    # medium questions
    "List the soft skills mentioned in the document.",
    "What did the candidate do to improve operational processes?",
    "Describe the leadership experience mentioned.",
    "What achievements are listed with numbers or percentages?",
    "What administrative responsibilities are mentioned?",

    # longer, multi-part and length-limited questions
    "In under 30 words, summarize the candidate's profile.",
    "Compare the two roles and say which one shows more leadership.",
    "Give a brief overview of the skills, experience and education together.",
    "Which skills in the skills section match the responsibilities in the experience section?",
    "Explain in two sentences what kind of job this candidate is suited for, based only on the document.",
]

# EDIT from Groq's pricing page for your model (USD per 1M tokens)
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.0, 0.0

# Thresholds
MAX_SECONDS = 5.0
MAX_COST_PER_QUERY = 0.002


class LatencyMetric(BaseMetric):
    def __init__(self, threshold=MAX_SECONDS):
        self.threshold = threshold
    def measure(self, tc: LLMTestCase):
        t = tc.completion_time
        self.score = min(1.0, self.threshold / t) if t else 0.0   # 1.0 = within budget
        self.success = t <= self.threshold
        self.reason = f"{t:.2f}s (limit {self.threshold}s)"
        return self.score
    async def a_measure(self, tc): return self.measure(tc)
    def is_successful(self): return self.success
    @property
    def __name__(self): return "Latency"


class CostMetric(BaseMetric):
    def __init__(self, threshold=MAX_COST_PER_QUERY):
        self.threshold = threshold
    def measure(self, tc: LLMTestCase):
        c = tc.token_cost or 0.0
        self.score = 1.0 if c <= self.threshold else self.threshold / c
        self.success = c <= self.threshold
        self.reason = f"${c:.5f} (limit ${self.threshold})"
        return self.score
    async def a_measure(self, tc): return self.measure(tc)
    def is_successful(self): return self.success
    @property
    def __name__(self): return "Cost"


class ReliabilityMetric(BaseMetric):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
    def measure(self, tc: LLMTestCase):
        status = (tc.metadata or {}).get("status", "unknown")
        self.score = 1.0 if status == "ok" else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"status = {status}"
        return self.score
    async def a_measure(self, tc): return self.measure(tc)
    def is_successful(self): return self.success
    @property
    def __name__(self): return "Reliability"


def classify(ans, err):
    if err: return "exception"
    if ans == BUSY_MESSAGE: return "rate_limited"
    if ans.startswith("Something went wrong"): return "llm_error"
    if not ans.strip(): return "empty"
    if ans == NOT_RELEVANT_MESSAGE: return "false_refusal"
    return "ok"


WARM_TID = "ops-warm"
_warm = {}

async def warm_up():
    await run_query(PDF, "What technical tools are listed?", WARM_TID)
    for name, store in (("doc", DOC_CACHE), ("chunk", CHUNK_CACHE), ("ret", RETRIEVER_STORE)):
        _warm[name] = _lru_get(store, WARM_TID)

def seed_cache(tid):
    for name, store in (("doc", DOC_CACHE), ("chunk", CHUNK_CACHE), ("ret", RETRIEVER_STORE)):
        if _warm.get(name) is not None:
            _lru_set(store, tid, _warm[name], MAX_THREADS_CACHED)


async def run_case(q):
    ans, err = "", ""
    tid = str(uuid.uuid4())
    seed_cache(tid)                      # PDF, chunks and embeddings already cached
    t0 = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        try:
            ans = (await run_query(PDF, q, tid)).get("retrieval", "") or ""
        except Exception as e:
            err = str(e)
    sec = time.perf_counter() - t0
    tin = sum(u.get("input_tokens", 0) for u in cb.usage_metadata.values())
    tout = sum(u.get("output_tokens", 0) for u in cb.usage_metadata.values())
    cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
    return LLMTestCase(
        input=q, actual_output=ans,
        completion_time=sec, token_cost=cost,
        metadata={"status": classify(ans, err), "tin": tin, "tout": tout},
    )


async def build_cases():
    await warm_up()                     
    return [await run_case(q) for q in QUESTIONS]

if __name__ == "__main__":
    cases = asyncio.run(build_cases())
    metrics = [LatencyMetric(), CostMetric(), ReliabilityMetric()]

    results = {m.__name__: [] for m in metrics}
    for tc in cases:
        print("\nQ:", tc.input)
        for m in metrics:
            m.measure(tc)
            results[m.__name__].append((m.score, m.success))
            print(f"  {m.__name__:12s} {'PASS' if m.success else 'FAIL'} | {m.reason}")

    secs = [c.completion_time for c in cases]
    s = sorted(secs)
    tin = sum(c.metadata["tin"] for c in cases)
    tout = sum(c.metadata["tout"] for c in cases)

    print("\n===== SUMMARY =====")
    for name, vals in results.items():
        print(f"{name:12s} pass rate {sum(ok for _, ok in vals)/len(vals):.0%}")
    print(f"Latency mean {statistics.mean(secs):.2f}s | p50 {s[len(s)//2]:.2f}s | "
          f"p95 {s[min(len(s)-1, int(0.95*len(s)))]:.2f}s | max {max(secs):.2f}s")
    print(f"Tokens: {tin} in, {tout} out  (avg {tin/len(cases):.0f} in, {tout/len(cases):.0f} out per query)")

    bad = [c for c in cases if c.metadata["status"] != "ok"]
    if bad:
        print("\nNon-ok runs:")
        for c in bad:
            print(f"  [{c.metadata['status']}] {c.input}")

    if PRICE_IN_PER_M == 0 and PRICE_OUT_PER_M == 0:
        print("\nSet PRICE_IN_PER_M / PRICE_OUT_PER_M to get a real cost.")