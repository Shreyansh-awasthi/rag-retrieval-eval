# Golden Dataset + Retrieval Evaluation

Tools to build a golden Q&A dataset for any AI topic and evaluate RAG retrieval
(BM25, dense embeddings, hybrid RRF, cross-encoder reranking).

## Setup
    pip install -r requirements.txt
    Create a .env file containing: GROQ_API_KEY=your_key

## Usage
    python golden_dataset.py --topic "Prompt engineering" --n 30
    python retrieval_eval_final.py --embedder BAAI/bge-small-en-v1.5

Put your own .txt/.md files in ./docs to evaluate on real documents.
Metrics: Hit@k, Precision@k, Recall@k, MRR, nDCG@k.
