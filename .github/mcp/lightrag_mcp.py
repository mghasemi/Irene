import os
import asyncio
from functools import partial
import fitz
from fastmcp import FastMCP
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

mcp = FastMCP("Local Repo Research Graph")

# 1. Dynamically find the directory where this script (.github/mcp/) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set the workspace folder to be exactly next to the script
WORKING_DIR = os.path.join(SCRIPT_DIR, "workspace")

# Create the directory if it doesn't exist yet
os.makedirs(WORKING_DIR, exist_ok=True)

# 1. Define the custom OpenRouter LLM function
async def openrouter_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model="google/gemini-3.1-flash-lite-preview", # Replace with your preferred OpenRouter model
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        **kwargs
    )

# 2. Initialize LightRAG with the custom function
# Create embedding function by parameterizing the already-instantiated openai_embed
# openai_embed is already an EmbeddingFunc, so we reconfigure it with OpenRouter params
async def openrouter_embed_func(texts, **kwargs):
    """Wrapper around openai_embed.func configured for OpenRouter API."""
    return await openai_embed.func(
        texts=texts,
        model="google/gemini-embedding-001",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        embedding_dim=1536,  # Explicitly request dimension reduction
        **kwargs
    )

openrouter_embed_wrapper = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    model_name="google/gemini-embedding-001",
    send_dimensions=False,
    func=openrouter_embed_func,
)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openrouter_complete,
    llm_model_name="google/gemini-3.1-flash-lite-preview",

    # Use OpenRouter's OpenAI-compatible embeddings endpoint.
    embedding_func=openrouter_embed_wrapper,
    embedding_batch_num=5,  # Process multiple items at once for efficiency
    # Increase timeout for OpenRouter's slower API service
    default_embedding_timeout=600,
    default_llm_timeout=600,
    embedding_func_max_async=1,
    llm_model_max_async=1,
    max_parallel_insert=1,
)

_rag_init_error = None
try:
    # LightRAG v1.4+ requires explicit storage initialization before insert/query.
    asyncio.run(rag.initialize_storages())
except Exception as e:
    _rag_init_error = str(e)


def _check_rag_ready() -> str | None:
    if _rag_init_error:
        return f"LightRAG initialization failed: {_rag_init_error}"
    return None


def _check_runtime_requirements() -> str | None:
    # Both generation and embeddings route via OpenRouter.
    if not os.environ.get("OPENROUTER_API_KEY"):
        return "Missing required environment variable: OPENROUTER_API_KEY"
    return None


def _clear_workspace() -> str:
    """Clear the entire workspace to remove duplicates and start fresh."""
    import shutil
    try:
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR, exist_ok=True)
        # Reinitialize storages
        asyncio.run(rag.initialize_storages())
        return f"Workspace cleared. {WORKING_DIR} is now empty and ready for fresh ingestion."
    except Exception as e:
        return f"Error clearing workspace: {str(e)}"


async def _reset_document_by_hash(doc_id: str) -> str:
    """Remove a document by ID from the graph and status tracking (async)."""
    try:
        await rag.adelete_by_doc_id(doc_id)
        return f"Removed document {doc_id} and all its entities/relations from the knowledge graph."
    except Exception as e:
        return f"Error removing document {doc_id}: {str(e)}"


def reset_document_by_hash(doc_id: str) -> str:
    """Sync wrapper to remove a document by ID."""
    loop = always_get_an_event_loop()
    return loop.run_until_complete(_reset_document_by_hash(doc_id))


def _summarize_track_result(track_id: str) -> tuple[bool, str]:
    """Return (ok, message) based on document statuses for a track_id."""
    loop = always_get_an_event_loop()
    docs = loop.run_until_complete(rag.aget_docs_by_track_id(track_id))

    if not docs:
        return False, f"Insertion started with track_id={track_id}, but no document status records were found."

    processed = 0
    failed = []
    pending_like = 0
    for doc_id, doc in docs.items():
        status = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
        if status == DocStatus.PROCESSED.value:
            processed += 1
        elif status == DocStatus.FAILED.value:
            failed.append(f"{doc_id}: {doc.error_msg or 'unknown error'}")
        else:
            pending_like += 1

    total = len(docs)
    if processed == total:
        return True, f"Successfully ingested document(s). track_id={track_id}, processed={processed}/{total}."

    details = []
    details.append(f"Ingestion did not fully complete. track_id={track_id}, processed={processed}/{total}")
    if pending_like:
        details.append(f"pending_or_processing={pending_like}")
    if failed:
        details.append("failed=" + "; ".join(failed[:3]))

    return False, ". ".join(details) + "."

@mcp.tool()
def ingest_file(filepath: str) -> str:
    """
    Ingests a specific local file into the RAG knowledge graph.
    Supports code (.py), notes (.md, .txt, .tex, .rst), and research papers (.pdf).
    """
    ready_error = _check_rag_ready()
    if ready_error:
        return ready_error

    requirements_error = _check_runtime_requirements()
    if requirements_error:
        return requirements_error

    expanded_path = os.path.expanduser(filepath)
    
    if not os.path.exists(expanded_path):
        return f"Error: File not found at {expanded_path}"
        
    supported_text_exts = ('.py', '.md', '.txt', '.tex', '.rst')
    content = ""
    
    try:
        # Route 1: Handle plain text and code files
        if expanded_path.lower().endswith(supported_text_exts):
            with open(expanded_path, "r", encoding="utf-8") as f:
                content = f.read()
                
        # Route 2: Handle PDF files
        elif expanded_path.lower().endswith('.pdf'):
            with fitz.open(expanded_path) as pdf_doc:
                # Extract text from each page and combine it
                content = "\n".join([page.get_text() for page in pdf_doc])
                
        else:
            return f"Error: Unsupported file extension. Please use {supported_text_exts} or .pdf"
        
        # Insert text into the LightRAG graph
        if content.strip():
            track_id = rag.insert(content, file_paths=expanded_path)
            ok, message = _summarize_track_result(track_id)
            if ok:
                return f"{message} File: {os.path.basename(expanded_path)}"
            return f"Failed to ingest {os.path.basename(expanded_path)}. {message}"
        else:
            return f"Warning: Read {os.path.basename(expanded_path)}, but no text could be extracted (it may be an image-only PDF)."
            
    except Exception as e:
        return f"Failed to ingest file: {str(e)}"
        
@mcp.tool()
def ingest_directory(directory_path: str) -> str:
    """
    Ingests local directories containing code (.py), notes (.md, .txt, .tex), 
    and research papers (.pdf) into the RAG knowledge graph.
    """
    ready_error = _check_rag_ready()
    if ready_error:
        return ready_error

    requirements_error = _check_runtime_requirements()
    if requirements_error:
        return requirements_error

    expanded_path = os.path.expanduser(directory_path)
    
    if not os.path.exists(expanded_path):
        return f"Error: Directory not found at {expanded_path}"
        
    supported_text_exts = ('.py', '.md', '.txt', '.tex', '.rst')
    ingested_count = 0
    error_list = []
    
    for root, _, files in os.walk(expanded_path):
        for file in files:
            filepath = os.path.join(root, file)
            content = ""
            
            try:
                # Route 1: Handle plain text and code files
                if file.endswith(supported_text_exts):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                # Route 2: Handle PDF files
                elif file.lower().endswith('.pdf'):
                    with fitz.open(filepath) as pdf_doc:
                        # Extract text from each page and combine it
                        content = "\n".join([page.get_text() for page in pdf_doc])
                        
                else:
                    continue # Skip unsupported file types
                
                # If we successfully extracted text, insert it into LightRAG
                if content.strip():
                    track_id = rag.insert(content, file_paths=filepath)
                    ok, message = _summarize_track_result(track_id)
                    if ok:
                        ingested_count += 1
                    else:
                        error_list.append(f"{file}: {message}")
                    
            except Exception as e:
                error_list.append(f"{file}: {str(e)}")
                    
    report = f"Successfully ingested {ingested_count} files from {directory_path}."
    if error_list:
        report += f"\nEncountered errors reading {len(error_list)} files: {', '.join(error_list[:5])}..."
            
    return report

@mcp.tool()
def clear_workspace() -> str:
    """
    Clears the entire workspace to remove duplicate documents and start fresh ingestion.
    WARNING: This deletes all graphs, vectors, and status records. Use only to reset for clean re-ingestion.
    """
    ready_error = _check_rag_ready()
    if ready_error:
        return ready_error
    
    return _clear_workspace()

@mcp.tool()
def reset_document(doc_id: str) -> str:
    """
    Remove a specific document by its ID from the knowledge graph and all its entities/relations.
    Useful for removing duplicates or corrupted ingestion results.
    
    Args:
        doc_id: The document ID to remove (e.g., "doc-c9dbc0090794fc1e61767ae1f6d6a8a5")
    """
    ready_error = _check_rag_ready()
    if ready_error:
        return ready_error
    
    return reset_document_by_hash(doc_id)

@mcp.tool()
def query_research_graph(query: str, mode: str = "hybrid") -> str:
    """
    Queries the LightRAG knowledge base.
    
    Args:
        query: The research question (e.g., "How does the SAGE certificate implementation relate to the moment problem notes?")
        mode: The search mode. Use "local" for specific entity details, "global" for broader concepts, or "hybrid" for both.
    """
    valid_modes = ["naive", "local", "global", "hybrid"]
    if mode not in valid_modes:
        mode = "hybrid"

    ready_error = _check_rag_ready()
    if ready_error:
        return ready_error
        
    try:
        # Query the graph
        result = rag.query(query, param=QueryParam(mode=mode))
        return result
    except Exception as e:
        return f"Error querying LightRAG: {str(e)}"

if __name__ == "__main__":
    mcp.run()
