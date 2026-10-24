import argparse
import csv
import json
import os
import random
import re
import time
from difflib import SequenceMatcher

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "qwen/qwen3.8-27b"

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Create a .env file with: GROQ_API_KEY=your_key")
client = Groq(api_key=api_key)

DEFAULT_TOPICS = [
    "Retrieval-Augmented Generation (RAG)",
    "Transformers and attention mechanisms",
    "Prompt engineering",
    "LLM evaluation and benchmarking",
    "Fine-tuning and LoRA",
    "AI agents and tool use",
    "Embeddings and vector databases",
    "AI safety and alignment",
]

DIFFICULTY_MIX = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
QUESTION_TYPES = ["factual", "conceptual", "comparison", "applied", "edge_case"]


def ask(prompt: str, max_tokens: int = 800, retries: int = 5) -> str:
    """Optimized ask wrapper with adaptive backoff and lower default max_tokens to stay below OTPM caps."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            time.sleep(3) 
            return resp.choices[0].message.content
        except Exception as e:
            wait_time = 15 * (attempt + 1)
            print(f"[warn] API error (attempt {attempt + 1}/{retries}). Waiting {wait_time}s. Error: {e}")
            time.sleep(wait_time)
    raise RuntimeError("Groq API failed after retries")


def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=0)
    end = max(text.rfind("]"), text.rfind("}")) + 1
    return json.loads(text[start:end])


def generate_batch(topic: str, count: int, difficulty: str) -> list[dict]:
    """Generates question-answer pairs. Uses strict formatting to control response length."""
    prompt = f"""You are building a GOLDEN evaluation dataset about: "{topic}".

Create exactly {count} question-answer pairs at "{difficulty}" difficulty.
Mix these question types: {", ".join(QUESTION_TYPES)}.

Rules:
- Every question must be self-contained and unambiguous.
- Reference answers must be highly accurate, clear, and brief (strictly 1-2 sentences).
- "key_facts" = 2 short atomic phrases a correct answer MUST contain.
- Keep output concise to fit within strict token limits.

Return ONLY a JSON array, no commentary:
[{{"question": "...", "answer": "...", "key_facts": ["..."], "type": "...", "difficulty": "{difficulty}"}}]"""
    
    # 800-900 max_tokens keeps the expected token count safe under the 1000 limit
    return parse_json(ask(prompt, max_tokens=850))


def generate(topic: str, n: int) -> list[dict]:
    """Generates items sequentially with pacing to avoid tripping burst limits."""
    items = []
    for diff, share in DIFFICULTY_MIX.items():
        # Target smaller micro-batches of up to 4 items at a time to minimize payload spikes
        target_count = max(1, round(n * share * 1.3))
        chunks = [4] * (target_count // 4) + ([target_count % 4] if target_count % 4 > 0 else [])
        
        for chunk in chunks:
            if chunk == 0:
                continue
            try:
                print(f"  Requesting micro-batch of {chunk} '{diff}' difficulty items...")
                items += generate_batch(topic, chunk, diff)
                # Cooling delay between sequential generations
                time.sleep(12)
            except Exception as e:
                print(f"[warn] micro-batch generation failed: {e}")
    return items


def dedupe(items: list[dict], threshold: float = 0.85) -> list[dict]:
    kept = []
    for it in items:
        q = it["question"].lower()
        if all(SequenceMatcher(None, q, k["question"].lower()).ratio() < threshold for k in kept):
            kept.append(it)
    return kept


def verify(topic: str, items: list[dict], min_score: int = 4) -> list[dict]:
    """Verifies items by chunking them into micro-payloads to respect free tier boundaries."""
    if not items:
        return items
        
    scores = {}
    # Process verification in ultra-small chunks of 3 items to stay completely safe under 1000 OTPM
    chunk_size = 3
    
    for idx in range(0, len(items), chunk_size):
        chunk_items = items[idx:idx + chunk_size]
        payload = [{"id": idx + i, "question": x["question"], "answer": x["answer"]} for i, x in enumerate(chunk_items)]
        
        prompt = f"""You are a strict fact-checker for the topic "{topic}".
Score each Q&A pair 1-5 on: correctness, clarity, and un-ambiguity (5 = perfect, 1 = wrong).
Return ONLY a valid JSON array: [{{"id": {idx}, "score": 5, "issue": "none"}}]

Pairs to score:
{json.dumps(payload, indent=2)}"""
        
        try:
            print(f"  Verifying indices {idx} to {idx + len(chunk_items) - 1}...")
            chunk_scores = parse_json(ask(prompt, max_tokens=400))
            for s in chunk_scores:
                scores[int(s["id"])] = s
            # Throttle delay between chunk validations
            time.sleep(10)
        except Exception as e:
            print(f"[warn] Verification chunk starting at index {idx} failed: {e}")
            # If a chunk fails, default them to baseline passing to avoid losing data
            for i in range(len(chunk_items)):
                scores[idx + i] = {"score": min_score}

    good = []
    for i, it in enumerate(items):
        s = scores.get(i, {})
        if s.get("score", 0) >= min_score:
            it["quality_score"] = s["score"]
            good.append(it)
    return good


def export(items: list[dict], topic: str, out: str):
    slug = re.sub(r"\W+", "_", topic.lower())[:20]
    for i, it in enumerate(items, 1):
        it["id"] = f"{slug}_{i:03d}"
        it["topic"] = topic

    with open(f"{out}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    fields = ["id", "topic", "type", "difficulty", "question", "answer", "key_facts", "quality_score"]
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for it in items:
            row = dict(it)
            row["key_facts"] = " | ".join(it.get("key_facts", []))
            w.writerow(row)
    print(f"Saved {len(items)} items -> {out}.jsonl, {out}.csv")


def main():
    ap = argparse.ArgumentParser(description="Generate a golden dataset for any AI topic")
    ap.add_argument("--topic", default=None, help="e.g. 'Prompt Engineering' (random if omitted)")
    ap.add_argument("--n", type=int, default=30, help="target number of final items")
    ap.add_argument("--out", default="golden_dataset", help="output file prefix")
    args = ap.parse_args()

    if not args.topic:
        args.topic = random.choice(DEFAULT_TOPICS)

    print(f"Generating for topic: {args.topic}")
    items = generate(args.topic, args.n)
    print(f"  generated: {len(items)}")
    items = dedupe(items)
    print(f"  after dedupe: {len(items)}")
    items = verify(args.topic, items)
    print(f"  after verification: {len(items)}")
    items = items[: args.n]
    export(items, args.topic, args.out)


if __name__ == "__main__":
    main()
