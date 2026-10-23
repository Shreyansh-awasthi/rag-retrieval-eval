import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "qwen/qwen3.8-27b"
KS = [1, 3, 5, 10]
MAXK = max(KS)
JUDGE_FILE = "judgments.json"

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Create a .env file with: GROQ_API_KEY=your_key")
client = Groq(api_key=api_key)


def ask(prompt, max_tokens=3000, retries=3, temperature=0.2):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = r.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except Exception as e:
            print(f"[warn] API error ({attempt + 1}/{retries}): {e}")
            time.sleep(5 * (attempt + 1))
    return ""


def parse_json_obj(text):
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return None

def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_golden(path, limit):
    if not os.path.exists(path):
        raise SystemExit(f"Golden dataset not found: {path}")
    items = []
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["key_facts"] = [x.strip() for x in (row.get("key_facts") or "").split("|") if x.strip()]
                items.append(row)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    items = [it for it in items if it.get("question") and it.get("answer")]
    for it in items:
        it.setdefault("key_facts", [])
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("Golden dataset is empty. Run golden_dataset.py first.")
    return items

def generate_synthetic_corpus(golden, out_dir, group=4):
    os.makedirs(out_dir, exist_ok=True)
    topic = golden[0].get("topic", "artificial intelligence")
    print(f"No docs found -> writing a synthetic knowledge base about '{topic}' into ./{out_dir}")
    print("(For a real evaluation, put your own .txt/.md files in ./docs)")
    for g in range(0, len(golden), group):
        qs = "\n".join("- " + x["question"] for x in golden[g:g + group])
        prompt = f"""Write a detailed, factual reference article (about 300 words) about {topic}.
It must naturally contain the information needed to answer these questions:
{qs}

Also include several related facts that are NOT answers to those questions.
Do not mention the questions. Plain prose only: no headings, no lists, no markdown."""
        text = ask(prompt, max_tokens=1500, temperature=0.5)
        if text:
            with open(os.path.join(out_dir, f"article_{g // group + 1:02d}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
        print(f"  article {g // group + 1}/{math.ceil(len(golden) / group)}", end="\r")
    print()


def load_corpus(docs_dir, golden):
    def read_dir(d):
        out = []
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if p.lower().endswith((".txt", ".md")):
                with open(p, encoding="utf-8", errors="ignore") as f:
                    out.append((os.path.basename(p), f.read()))
        return out

    docs = read_dir(docs_dir) if os.path.isdir(docs_dir) else []
    if not docs:
        synth = "docs_synth"
        if not read_dir(synth):
            generate_synthetic_corpus(golden, synth)
        docs = read_dir(synth)
    if not docs:
        raise SystemExit("No corpus available. Put .txt/.md files in ./docs")
    return docs


def chunk_doc(text, size, overlap):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > size:
            chunks.append(" ".join(cur))
            keep, kn = [], 0
            for t in reversed(cur):
                tw = len(t.split())
                if kn + tw > overlap:
                    break
                keep.insert(0, t)
                kn += tw
            cur, n = keep, kn
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(docs, size, overlap):
    chunks = []
    for name, text in docs:
        for c in chunk_doc(text, size, overlap):
            chunks.append({"id": len(chunks), "text": c, "source": name, "hash": md5(c)})
    return chunks

STOP = set("a an the of to in and or is are was were be been for on with as by at from that this it its "
           "into than then so such can which how what why when who does do did not no".split())


def tok(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class BM25:
    name = "BM25"

    def __init__(self, chunks, k1=1.5, b=0.75):
        docs = [tok(c["text"]) for c in chunks]
        self.k1, self.b, self.N = k1, b, len(docs)
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, self.N)
        self.tf, df = [], defaultdict(int)
        for d in docs:
            counts = defaultdict(int)
            for w in d:
                counts[w] += 1
            self.tf.append(counts)
            for w in counts:
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, q):
        s = np.zeros(self.N)
        qs = tok(q)
        for i in range(self.N):
            tf, dl = self.tf[i], self.len[i]
            for w in qs:
                f = tf.get(w)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def search(self, q, k, qx=None):
        return list(np.argsort(-self.scores(qx or q))[:k])


class Dense:
    def __init__(self, chunks, model_name):
        from sentence_transformers import SentenceTransformer
        self.name = "Dense"
        self.model = SentenceTransformer(model_name)
        self.prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
        self.emb = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)

    def scores(self, q):
        qe = self.model.encode([self.prefix + q], normalize_embeddings=True)[0]
        return self.emb @ qe

    def search(self, q, k, qx=None):
        return list(np.argsort(-self.scores(qx or q))[:k])


class Hybrid:
    """Reciprocal Rank Fusion of BM25 + Dense."""
    name = "Hybrid (RRF)"

    def __init__(self, bm25, dense, pool=30, c=60):
        self.bm25, self.dense, self.pool, self.c = bm25, dense, pool, c

    def search(self, q, k, qx=None):
        fused = defaultdict(float)
        for r in (self.bm25, self.dense):
            for rank, i in enumerate(r.search(q, self.pool, qx)):
                fused[i] += 1.0 / (self.c + rank + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


class Rerank:
    """First stage retrieves a wide pool; a cross-encoder re-scores (question, chunk) pairs."""

    def __init__(self, base, chunks, pool=20, model_name="BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.name = base.name + " + Rerank"
        self.base, self.chunks, self.pool = base, chunks, pool
        self.ce = CrossEncoder(model_name)

    def search(self, q, k, qx=None):
        cand = self.base.search(q, self.pool, qx)
        sc = self.ce.predict([(q, self.chunks[i]["text"]) for i in cand], show_progress_bar=False)
        return [cand[j] for j in np.argsort(-np.array(sc))[:k]]


class Hyde:
    """HyDE: LLM writes a hypothetical answer; it is added to the query for first-stage retrieval."""

    def __init__(self, base):
        self.name = base.name + " + HyDE"
        self.base, self.cache = base, {}

    def search(self, q, k, qx=None):
        if q not in self.cache:
            self.cache[q] = ask(f"Write a short, factual 2-3 sentence passage that answers:\n{q}", max_tokens=400)
        return self.base.search(q, k, qx=q + " " + self.cache[q])


# =====================================================================
# LLM judging of retrieved chunks (pooled relevance labels)
# =====================================================================
def load_judgments():
    if os.path.exists(JUDGE_FILE):
        try:
            with open(JUDGE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_judgments(j):
    with open(JUDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(j, f)


def judge_batch(item, chunks, ids):
    passages = "\n\n".join(f"[{n + 1}] {chunks[i]['text'][:900]}" for n, i in enumerate(ids))
    facts = "; ".join(item["key_facts"]) if item["key_facts"] else "(none given)"
    prompt = f"""You are grading search results for a question-answering system.

Question: {item['question']}
Reference answer: {item['answer']}
Key facts: {facts}

Grade EACH numbered passage:
2 = contains the key information needed to answer the question
1 = related or partially helpful, but not enough to answer
0 = irrelevant

Return ONLY JSON like {{"grades": {{"1": 2, "2": 0}}}}

Passages:
{passages}"""
    for _ in range(2):
        obj = parse_json_obj(ask(prompt, max_tokens=3000, temperature=0.0))
        if obj and "grades" in obj:
            out = {}
            for n, i in enumerate(ids):
                try:
                    out[i] = int(obj["grades"].get(str(n + 1), 0))
                except Exception:
                    out[i] = 0
            return out
    return None


def grade_pool(golden, chunks, rankings):
    judgments = load_judgments()
    grades = {}
    for qi, item in enumerate(golden):
        qh = md5(item["question"])
        pool = []
        for name in rankings:
            for i in rankings[name][qi]:
                if i not in pool:
                    pool.append(i)
        g = {}
        todo = []
        for i in pool:
            key = f"{qh}:{chunks[i]['hash']}"
            if key in judgments:
                g[i] = judgments[key]
            else:
                todo.append(i)
        for b in range(0, len(todo), 8):
            batch = todo[b:b + 8]
            res = judge_batch(item, chunks, batch)
            if res is None:
                continue
            for i, v in res.items():
                g[i] = v
                judgments[f"{qh}:{chunks[i]['hash']}"] = v
        grades[qi] = g
        save_judgments(judgments)
        print(f"  graded question {qi + 1}/{len(golden)}", end="\r")
    print()
    return grades


def metrics_for(ranked, g):
    rel = {i for i, v in g.items() if v >= 2}
    first = next((r + 1 for r, i in enumerate(ranked) if i in rel), None)
    m = {"mrr": 1.0 / first if first else 0.0}
    ideal = sorted(g.values(), reverse=True)
    for k in KS:
        top = ranked[:k]
        hits = sum(1 for i in top if i in rel)
        m[f"hit@{k}"] = 1.0 if hits else 0.0
        m[f"precision@{k}"] = hits / k
        m[f"recall@{k}"] = hits / len(rel)
        dcg = sum(g.get(i, 0) / math.log2(r + 2) for r, i in enumerate(top))
        idcg = sum(v / math.log2(r + 2) for r, v in enumerate(ideal[:k]))
        m[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return m, first


def evaluate(rankings, grades, n_q):
    results, firsts = {}, {}
    valid = [qi for qi in range(n_q) if any(v >= 2 for v in grades.get(qi, {}).values())]
    for name, per_q in rankings.items():
        rows, firsts[name] = [], {}
        for qi in valid:
            m, first = metrics_for(per_q[qi], grades[qi])
            rows.append(m)
            firsts[name][qi] = first
        results[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    return results, firsts, valid


def print_table(results):
    cols = ["hit@1", "hit@5", "prec@5", "prec@10", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"]
    alias = {"prec@5": "precision@5", "prec@10": "precision@10"}
    w = max(len(n) for n in results) + 2
    print("\n" + "Retriever".ljust(w) + "".join(c.rjust(11) for c in cols))
    print("-" * (w + 11 * len(cols)))
    for name, m in results.items():
        print(name.ljust(w) + "".join(f"{m.get(alias.get(c, c), 0):11.3f}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description="Evaluate and improve retrieval with your golden dataset")
    ap.add_argument("--golden", default="golden_dataset.jsonl")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--chunk-size", type=int, default=120, help="words per chunk")
    ap.add_argument("--overlap", type=int, default=30, help="overlap in words")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N golden questions")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2", help="e.g. BAAI/bge-small-en-v1.5")
    ap.add_argument("--hyde", action="store_true", help="also test HyDE query expansion (extra LLM calls)")
    args = ap.parse_args()

    golden = load_golden(args.golden, args.limit)
    print(f"Golden questions: {len(golden)}")
    docs = load_corpus(args.docs, golden)
    chunks = build_chunks(docs, args.chunk_size, args.overlap)
    print(f"Corpus: {len(docs)} docs -> {len(chunks)} chunks")

    retrievers = [BM25(chunks)]
    try:
        dense = Dense(chunks, args.embedder)
        hybrid = Hybrid(retrievers[0], dense)
        reranked = Rerank(hybrid, chunks)
        retrievers += [dense, hybrid, reranked]
        if args.hyde:
            retrievers.append(Hyde(reranked))
    except ImportError:
        print("[warn] sentence-transformers not installed -> only BM25 will run.")
        print("       pip install sentence-transformers")

    print("Retrieving...")
    rankings = {}
    for r in retrievers:
        rankings[r.name] = {qi: r.search(it["question"], MAXK) for qi, it in enumerate(golden)}

    print("Grading retrieved chunks with the LLM (cached in judgments.json)...")
    grades = grade_pool(golden, chunks, rankings)

    results, firsts, valid = evaluate(rankings, grades, len(golden))
    skipped = len(golden) - len(valid)
    print(f"\nEvaluated {len(valid)}/{len(golden)} questions.")
    if skipped:
        print(f"{skipped} question(s) had NO relevant chunk anywhere in the corpus (skipped). "
              f"Add docs that cover them.")
    if not valid:
        raise SystemExit("No gradable questions. Check the corpus / API key.")

    print_table(results)
    best = max(results, key=lambda n: results[n].get("ndcg@5", 0))
    print(f"\nBest retriever by nDCG@5: {best}")

    with open("retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("retrieval_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["retriever", "question", "key_facts", "rank_of_first_relevant", "top1_chunk", "top1_grade", "best_relevant_chunk"])
        n_fail = 0
        for qi in valid:
            first = firsts[best][qi]
            if first == 1:
                continue
            ranked = rankings[best][qi]
            g = grades[qi]
            rel = [i for i, v in g.items() if v >= 2]
            w.writerow([best, golden[qi]["question"], " | ".join(golden[qi]["key_facts"]),
                        first or f">{MAXK}", chunks[ranked[0]]["text"][:300], g.get(ranked[0], 0),
                        chunks[rel[0]]["text"][:300]])
            n_fail += 1
    print(f"Saved {n_fail} misses for '{best}' -> retrieval_failures.csv, all scores -> retrieval_results.json")

    print("\nHow to improve further:")
    print("  - Low hit@5 for every retriever  -> your docs don't cover the questions, or chunks are too big/small (--chunk-size)")
    print("  - Dense >> BM25                  -> questions are paraphrased; keep embeddings in the mix")
    print("  - Hybrid + Rerank best           -> use it in production, retrieve ~20 and rerank down to 5")
    print("  - Try a stronger embedder        -> --embedder BAAI/bge-small-en-v1.5")


if __name__ == "__main__":
    main()
