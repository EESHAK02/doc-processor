import os
from langfuse import Langfuse

def get_langfuse_client():
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

def get_or_create_trace(session_id: str, node_name: str):
    try:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

        if not public_key or not secret_key or public_key == "dummy":
            return None

        lf = get_langfuse_client()
        
        # Langfuse v3 uses trace() differently — create via client directly
        trace = lf.trace(
            name=node_name,
            user_id=session_id,
            session_id=session_id,
            metadata={"node": node_name}
        )
        return trace
    except Exception as e:
        print(f"[LANGFUSE] Error: {e}", flush=True)
        return None

def log_span(trace, name: str, input_data: dict, output_data: dict, metadata: dict = None):
    if not trace:
        return
    try:
        span = trace.span(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )
        span.end()
    except Exception as e:
        print(f"[LANGFUSE] Span error: {e}", flush=True)

def log_llm_call(trace, name: str, model: str, prompt: str, response: str, metadata: dict = None):
    if not trace:
        return
    try:
        generation = trace.generation(
            name=name,
            model=model,
            input=prompt,
            output=response,
            metadata=metadata or {},
        )
        generation.end()
    except Exception as e:
        print(f"[LANGFUSE] Generation error: {e}", flush=True)

def flush_langfuse():
    try:
        lf = get_langfuse_client()
        lf.flush()
    except Exception:
        pass