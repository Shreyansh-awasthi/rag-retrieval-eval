from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import asyncio
import logging
import traceback
import inspect
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-api")

logger.info("LANGSMITH_TRACING=%s", os.getenv("LANGSMITH_TRACING"))
logger.info("LANGSMITH_PROJECT=%s", os.getenv("LANGSMITH_PROJECT"))
logger.info("LANGSMITH_API_KEY set=%s", bool(os.getenv("LANGSMITH_API_KEY")))


RAG_IMPORT_ERROR = None
run_query = None
stream_query = None
try:
    try:
        from .Rag import run_query, stream_query
    except (ModuleNotFoundError, ImportError):
        from Rag import run_query, stream_query
except Exception as e:
    RAG_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    logger.error("Failed to import Rag.py: %s", RAG_IMPORT_ERROR)
    traceback.print_exc()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT_SECONDS = 90
MAX_FILE_SIZE_MB = 20
UPLOAD_DIR = "uploaded_docs"

try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception as e:
    logger.error("Could not create upload directory %s: %s", UPLOAD_DIR, e)

MAX_CONCURRENT_QUERIES = 3
query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)


@app.middleware("http")
async def catch_all_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error("Unhandled error in middleware for %s: %s", request.url.path, e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error: {type(e).__name__}: {e}"},
        )
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s %s completed in %.2fs", request.method, request.url.path, elapsed)



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.info("HTTPException on %s: %s - %s", request.url.path, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}: {exc}"},
    )


@app.post("/query/stream")
async def query_stream(
    file: UploadFile = File(...),
    question: str = Form(...),
    thread_id: str = Form(...),
):
    if RAG_IMPORT_ERROR:
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline failed to load at startup: {RAG_IMPORT_ERROR}",
        )

    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    safe_filename = f"{thread_id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(saved_path, "wb") as f:
            f.write(contents)
    except PermissionError as e:
        logger.exception("Permission denied saving uploaded file")
        raise HTTPException(status_code=500, detail=f"Permission denied saving file: {e}")
    except OSError as e:
        logger.exception("OS error saving uploaded file (disk full? path issue?)")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    except Exception as e:
        logger.exception("Unexpected error saving uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    acquired = False
    try:
        try:
            await asyncio.wait_for(query_semaphore.acquire(), timeout=30)
            acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Server is busy handling other requests. Please try again shortly.",
            )

        async def token_generator():
            nonlocal acquired
            try:
                async for chunk in stream_query(saved_path, question, thread_id):
                    yield chunk
            except Exception as e:
                logger.exception("stream_query failed for thread_id=%s", thread_id)
                yield f"\n[Error: {type(e).__name__}: {e}]"
            finally:
                if acquired:
                    query_semaphore.release()
                    acquired = False

        return StreamingResponse(token_generator(), media_type="text/plain")

    except HTTPException:
        if acquired:
            query_semaphore.release()
        raise
    except Exception as e:
        if acquired:
            query_semaphore.release()
        logger.exception("Unexpected error before streaming for thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
async def health():
    status = "ok" if RAG_IMPORT_ERROR is None else "degraded"
    return {
        "status": status,
        "rag_pipeline_loaded": RAG_IMPORT_ERROR is None,
        "rag_import_error": RAG_IMPORT_ERROR,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))