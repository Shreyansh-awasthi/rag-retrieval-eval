import asyncio, json, re, time
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from Rag import run_query

PDF = r"C:\Users\ayush\OneDrive\Desktop\simple_minimalist_resume.pdf"

# Questions the PDF can really answer + a short true answer for each
QA = [
    ("What technical tools are listed?",
     "Microsoft Office Suite, Google Workspace, industry-specific software and CRM platforms."),
    ("Summarize the work experience and education.",
     "Professional experience includes optimizing workflows and reducing project delivery times by 15%, plus an education section."),
]

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, name="qwen/qwen3.8-27b"):
        self.name, self.llm = name, ChatGroq(model=name, temperature=0)

    def load_model(self): return self.llm
    def get_model_name(self): return self.name

    def _clean(self, t):
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        return t.replace("```json", "").replace("```", "").strip()

    def generate(self, prompt, schema=None):
        if schema:
            prompt += f"\nReturn ONLY JSON matching: {schema.model_json_schema()}"
        for i in range(4):
            try:
                out = self._clean(self.llm.invoke(prompt).content)
                return schema.model_validate(json.loads(out)) if schema else out
            except Exception as e:
                if "429" in str(e): time.sleep(5 * (i + 1))   # wait only when rate-limited
                else: raise
        raise RuntimeError("Judge failed after retries")

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)

async def build_cases():
    cases = []
    for q, expected in QA:
        r = await run_query(PDF, q)
        cases.append(LLMTestCase(
            input=q,
            actual_output=r["retrieval"],
            expected_output=expected,
            retrieval_context=[d.page_content for d in r["retrieved_docs"]],
        ))
    return cases

if __name__ == "__main__":
    judge = GroqJudge()
    metrics = [ContextualPrecisionMetric(model=judge, include_reason=True),
               ContextualRecallMetric(model=judge, include_reason=True),
               ContextualRelevancyMetric(model=judge, include_reason=True)]
    for case in asyncio.run(build_cases()):
        print("\nQ:", case.input)
        for m in metrics:
            m.measure(case)
            print(f"  {m.__class__.__name__}: {m.score} | {m.reason}")