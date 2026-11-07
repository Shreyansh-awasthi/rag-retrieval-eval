import asyncio, json, os, re, time, statistics
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from Rag import run_query, BUSY_MESSAGE

PDF = r"C:\Users\ayush\OneDrive\Desktop\simple_minimalist_resume.pdf"
GOLDENS = "Rag_evals/goldens.json"

# Hand-written Q&A, always added. Check these against your PDF.
EXTRA = [
    {"input": "Which role reduced project delivery times by 15%, and what education is listed?",
     "expected_output": "The most recent role (Target Job Title at Company Name) reduced delivery times by 15%. The education listed is Degree Name / Major from a University or Institution."},
    {"input": "What is the candidate's current salary and expected salary?",
     "expected_output": "The document does not contain any salary information."},
    {"input": "In under 25 words, list the technical tools.",
     "expected_output": "Microsoft Office Suite, Google Workspace, Industry-Specific Software, CRM Platforms."},
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


def make_metrics(judge):
    I, A, E = (LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
               LLMTestCaseParams.EXPECTED_OUTPUT)
    kw = dict(model=judge, async_mode=False)

    correctness = GEval(
        name="Correctness",
        criteria="Judge whether the facts in the actual output are correct "
                 "when compared with the expected output. Penalize wrong or contradicting facts. "
                 "If the expected output says the information is not in the document, "
                 "the actual output must say the same and must not invent details.",
        evaluation_params=[I, A, E],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Mostly wrong, or contradicts the expected output."),
            Rubric(score_range=(3, 5), expected_outcome="Some facts correct, but notable errors or invented details."),
            Rubric(score_range=(6, 8), expected_outcome="Mostly correct, with only minor inaccuracies."),
            Rubric(score_range=(9, 10), expected_outcome="Fully correct and consistent with the expected output."),
        ], **kw)

    completeness = GEval(
        name="Completeness",
        criteria="Judge whether the actual output covers every key point in the expected output "
                 "and answers every part of the question. Do not penalize correct extra detail.",
        evaluation_params=[I, A, E],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Misses most key points or most parts of the question."),
            Rubric(score_range=(3, 5), expected_outcome="Covers about half of the key points."),
            Rubric(score_range=(6, 8), expected_outcome="Covers nearly all key points, missing one minor point."),
            Rubric(score_range=(9, 10), expected_outcome="Covers all key points and all parts of the question."),
        ], **kw)

    style = GEval(
        name="Style",
        criteria="Judge the writing style of the actual output. It should be clear, concise, "
                 "professional plain-text prose with no markdown (no asterisks, bullets, headings), "
                 "no padding, and it must respect any length limit in the question.",
        evaluation_params=[I, A],   # no reference needed
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Confusing, uses markdown or lists, ignores a length limit, or is heavily padded."),
            Rubric(score_range=(3, 5), expected_outcome="Understandable but wordy, awkward, or breaks a formatting rule."),
            Rubric(score_range=(6, 8), expected_outcome="Clear plain prose with small issues in concision or flow."),
            Rubric(score_range=(9, 10), expected_outcome="Clear, concise, professional plain text that follows all constraints."),
        ], **kw)

    return [correctness, completeness, style]


def load_items():
    items = json.load(open(GOLDENS, encoding="utf-8")) if os.path.exists(GOLDENS) else []
    seen = {g["input"] for g in items}
    return items + [e for e in EXTRA if e["input"] not in seen]


async def build_cases(items):
    cases = []
    for g in items:
        r = await run_query(PDF, g["input"])
        ans = r.get("retrieval", "")
        if ans == BUSY_MESSAGE:
            print("Skipping (rate limited):", g["input"])
            continue
        # refusals are NOT skipped here: for an unanswerable question, a refusal is the right answer
        cases.append(LLMTestCase(input=g["input"], actual_output=ans,
                                 expected_output=g["expected_output"]))
    return cases


if __name__ == "__main__":
    judge = GroqJudge()
    metrics = make_metrics(judge)
    cases = asyncio.run(build_cases(load_items()))

    scores = {m.name: [] for m in metrics}
    for case in cases:
        print("\nQ:", case.input)
        print("A:", case.actual_output[:300])
        for m in metrics:
            m.measure(case)
            scores[m.name].append(m.score)
            print(f"  {m.name}: {m.score:.2f} | {m.reason}")

    print("\n===== AVERAGES =====")
    for name, vals in scores.items():
        if vals:
            print(f"{name:14s} {statistics.mean(vals):.2f}  (n={len(vals)})")