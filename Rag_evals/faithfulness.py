import asyncio, json, os, re, time, statistics
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from Rag import run_query, NOT_RELEVANT_MESSAGE, BUSY_MESSAGE

PDF = r"C:\Users\ayush\OneDrive\Desktop\simple_minimalist_resume.pdf"
GOLDENS = "Rag_evals/goldens.json"

# Used only if goldens.json doesn't exist yet
FALLBACK_QUESTIONS = [
    "What technical tools are listed?",
    "Summarize the work experience and education.",
]

# Always added, whether or not goldens.json exists
EXTRA_QUESTIONS = [
    # needs two sections at once
    "Which role reduced project delivery times by 15%, and what education is listed for the candidate?",
    # needs comparison across both roles
    "How do the responsibilities in the two listed roles differ, and which one shows more leadership?",
    # not answerable from the document (tests for made-up facts)
    "What is the candidate's current salary and expected salary?",
    # length limit plus two sections
    "In under 25 words, list the core competencies and the technical tools.",
]


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, name="qwen/qwen3.8-27b"):
        self.name, self.llm = name, ChatGroq(model=name, temperature=0)

    def load_model(self): return self.llm
    def get_model_name(self): return self.name

    @staticmethod
    def _clean(t):
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        return t.replace("```json", "").replace("```", "").strip()

    def generate(self, prompt, schema=None):
        if schema:
            prompt += f"\nReturn ONLY JSON matching: {schema.model_json_schema()}"
        for i in range(4):
            try:
                out = self._clean(self.llm.invoke(prompt).content)
                if not schema:
                    return out
                m = re.search(r"\{.*\}", out, flags=re.S)
                return schema.model_validate(json.loads(m.group(0) if m else out))
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5 * (i + 1))
                elif i == 3:
                    raise
        raise RuntimeError("Judge failed after retries")

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def load_questions():
    if os.path.exists(GOLDENS):
        qs = [g["input"] for g in json.load(open(GOLDENS, encoding="utf-8"))]
    else:
        print("goldens.json not found, using fallback questions.")
        qs = list(FALLBACK_QUESTIONS)
    return qs + [q for q in EXTRA_QUESTIONS if q not in qs]


async def build_cases(questions):
    cases = []
    for q in questions:
        r = await run_query(PDF, q)
        ans = r.get("retrieval", "")
        if ans in (NOT_RELEVANT_MESSAGE, BUSY_MESSAGE) or not r.get("retrieved_docs"):
            print("Skipping (refusal/busy/no retrieval):", q)
            continue
        cases.append(LLMTestCase(
            input=q,
            actual_output=ans,
            retrieval_context=[d.page_content for d in r["retrieved_docs"]],
        ))
    return cases


if __name__ == "__main__":
    judge = GroqJudge()
    cases = asyncio.run(build_cases(load_questions()))

    kw = dict(model=judge, include_reason=True, async_mode=False)
    metrics = [FaithfulnessMetric(**kw), AnswerRelevancyMetric(**kw)]

    scores = {m.__class__.__name__: [] for m in metrics}
    for case in cases:
        print("\nQ:", case.input)
        print("A:", case.actual_output[:300])
        for m in metrics:
            m.measure(case)
            scores[m.__class__.__name__].append(m.score)
            print(f"  {m.__class__.__name__}: {m.score:.2f} | {m.reason}")

    print("\n===== AVERAGES =====")
    for name, vals in scores.items():
        if vals:
            print(f"{name:24s} {statistics.mean(vals):.2f}  (n={len(vals)})")