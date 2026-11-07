FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/uploaded_docs /app/chroma_store /models

EXPOSE 8000 8501

# default = API; compose overrides this for the frontend
CMD ["python", "-m", "uvicorn", "mainapi:app", "--host", "0.0.0.0", "--port", "8000"]