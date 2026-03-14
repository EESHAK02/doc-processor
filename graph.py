# Graph flow:load_documents, classify, extract, validate, respond

from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END

from nodes import (
    load_documents_node,
    classify_node,
    extract_node,
    validate_node,
    respond_node,
)


# state schema for the document processing agent

class DocAgentState(TypedDict, total=False):
    file_paths: List[str]          # input: local paths to uploaded PDFs
    documents: List[dict]          # populated by load, classify, extract, validate
    question: str                  # chat question from user
    answer: str                    # final answer from respond_node
    trace: Optional[Any]           # Langfuse trace object (not serialized)
    chat_history: List[dict] 


# building the graph

def build_graph():
    graph = StateGraph(DocAgentState)

    graph.add_node("load_documents", load_documents_node)
    graph.add_node("classify",       classify_node)
    graph.add_node("extract",        extract_node)
    graph.add_node("validate",       validate_node)
    graph.add_node("respond",        respond_node)

    graph.set_entry_point("load_documents")
    graph.add_edge("load_documents", "classify")
    graph.add_edge("classify",       "extract")
    graph.add_edge("extract",        "validate")
    graph.add_edge("validate",       "respond")
    graph.add_edge("respond",        END)

    return graph.compile()


# convenience runner

def run_processing_pipeline(file_paths: list, question: str, trace=None, chat_history: list = None) -> dict:
    
    # Run the full graph for document processing + answering a question, returns the final state dict
    
    app = build_graph()
    initial_state = {
        "file_paths": file_paths,
        "question": question,
        "trace": trace,
        "chat_history": chat_history or [], 
    }
    result = app.invoke(initial_state)
    return result
