import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USER_AGENT'] = 'ClaudeMCP'
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

from mcp.server.fastmcp import FastMCP
from Rag import run_query

mcp = FastMCP("rag-server")

@mcp.tool()
async def query_pdf(file_path: str, question: str, thread_id: str = "default") -> str:
    """Answer a question about a PDF document using RAG"""
    result = await run_query(file_path, question, thread_id)
    return result.get("retrieval", "") or "No answer could be generated."

if __name__ == "__main__":
    mcp.run(transport="stdio")
