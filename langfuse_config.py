import os
from functools import wraps
from langfuse import Langfuse

# Initialize Langfuse client (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env)
def get_langfuse_client():
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

def create_trace(name: str, user_id: str = None, metadata: dict = None):
    """Create a new Langfuse trace for a full document processing run."""
    lf = get_langfuse_client()
    return lf.trace(
        name=name,
        user_id=user_id or "anonymous",
        metadata=metadata or {},
    )

def log_span(trace, name: str, input_data: dict, output_data: dict, metadata: dict = None):
    """Log a span (node execution) within a trace."""
    span = trace.span(
        name=name,
        input=input_data,
        output=output_data,
        metadata=metadata or {},
    )
    span.end()
    return span

def log_llm_call(trace, name: str, model: str, prompt: str, response: str, metadata: dict = None):
    """Log an LLM generation within a trace."""
    generation = trace.generation(
        name=name,
        model=model,
        input=prompt,
        output=response,
        metadata=metadata or {},
    )
    generation.end()
    return generation
