# langraph nodes 

import json
import re
import os
import traceback
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
import tempfile
# import pytesseract
# import platform
# if platform.system() == "Windows":
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Users\eesha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_path
import fitz  # pymupdf
import base64
from groq import Groq

from prompts import (
    CLASSIFY_PROMPT,
    EXTRACT_PROMPT_INVOICE,
    EXTRACT_PROMPT_EXPENSE,
    EXTRACT_PROMPT_BOOKING,
    EXTRACT_PROMPT_GENERIC,
    VALIDATE_PROMPT,
    VALIDATE_PROMPT_GENERIC,
    CHAT_PROMPT,
)
from langfuse_config import log_span, log_llm_call

# helpers

def _get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        max_tokens=2048,
    )

def _parse_json(text: str) -> dict:
    # Strip markdown fences and parse JSON safely
    clean = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean)

def _truncate(text: str, max_chars: int = 6000) -> str:
    # Truncate document text to avoid token limits
    return text[:max_chars] + "\n...[truncated]" if len(text) > max_chars else text

# def _ocr_pdf(path: str) -> str:
#     """Convert PDF pages to images and extract text via Tesseract OCR."""
#     try:
#         images = convert_from_path(path, dpi=200)
#         pages_text = []
#         for img in images:
#             text = pytesseract.image_to_string(img)
#             pages_text.append(text)
#         return "\n".join(pages_text)
#     except Exception as e:
#         return ""
def _ocr_with_vision(path: str) -> str:
    """
    Render PDF pages as images and extract text using
    Groq's Llama vision model. No system dependencies needed.
    """
    try:
        doc = fitz.open(path)
        all_text = []

        for page_num in range(min(len(doc), 5)):  # cap at 5 pages
            page = doc[page_num]
            mat = fitz.Matrix(2, 2)  # 2x zoom ≈ 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this document image exactly as it appears. Return only the extracted text, no commentary."
                        }
                    ]
                }],
                max_tokens=2048,
            )
            all_text.append(response.choices[0].message.content)

        doc.close()
        return "\n".join(all_text)

    except Exception as e:
        print(f"[OCR] Vision OCR failed: {e}", flush=True)
        return ""

# Node 1: load_documents

def load_documents_node(state: dict) -> dict:
    # Parse uploaded PDF file paths into raw text.
    # Populates state["documents"] = [{"filename": str, "text": str}]
    
    trace = state.get("trace")
    file_paths = state.get("file_paths", [])
    documents = []

    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            full_text = "\n".join(p.page_content for p in pages)
            filename = os.path.basename(path)

            ocr_used = False
            if not full_text.strip():
                print(f"[OCR] No text found in {filename}, falling back to Tesseract...", flush=True)
                full_text = _ocr_with_vision(path)
                ocr_used = True

            documents.append({"filename": filename, "text": full_text, "path": path, "ocr_used": ocr_used})
        except Exception as e:
            filename = os.path.basename(path)
            documents.append({
                "filename": filename,
                "text": "",
                "path": path,
                "load_error": str(e),
                "ocr_used": False,
            })

    if trace:
        log_span(trace, "load_documents", 
                 {"file_count": len(file_paths)},
                 {"loaded": len(documents), 
                  "ocr_used": [d["filename"] for d in documents if d.get("ocr_used")]})

    return {**state, "documents": documents}


# Node 2: classify_node 

def classify_node(state: dict) -> dict:

    # Classify each document as invoice / expense_report / booking_confirmation / unknown and populate
    # doc["doc_type"] and doc["classification"] for each document.
    
    trace = state.get("trace")
    llm = _get_llm()
    documents = state.get("documents", [])

    for doc in documents:
        if doc.get("load_error") or not doc.get("text"):
            doc["doc_type"] = "unknown"
            doc["classification"] = {"doc_type": "unknown", "confidence": 0.0, "reasoning": "Could not load document"}
            continue
        try:
            prompt = CLASSIFY_PROMPT.format(text=_truncate(doc["text"], 3000))
            response = llm.invoke([HumanMessage(content=prompt)])
            result = _parse_json(response.content)
            doc["doc_type"] = result.get("doc_type", "unknown")
            doc["classification"] = result

            if trace:
                log_llm_call(trace, f"classify:{doc['filename']}", "llama-3.3-70b-versatile",
                             prompt, response.content,
                             {"filename": doc["filename"], "doc_type": doc["doc_type"]})
        except Exception as e:
            doc["doc_type"] = "unknown"
            doc["classification"] = {"doc_type": "unknown", "confidence": 0.0, "reasoning": f"Error: {e}"}

    return {**state, "documents": documents}


# Node 3: extract_node 

EXTRACT_PROMPT_MAP = {
    "invoice":              EXTRACT_PROMPT_INVOICE,
    "expense_report":       EXTRACT_PROMPT_EXPENSE,
    "booking_confirmation": EXTRACT_PROMPT_BOOKING,
}

def extract_node(state: dict) -> dict:
    trace = state.get("trace")
    llm = _get_llm()
    documents = state.get("documents", [])

    for doc in documents:
        doc_type = doc.get("doc_type", "unknown")

        if not doc.get("text"):
            doc["extracted"] = {}
            continue

        # Known type set to specific prompt. Unknown to generic fallback
        prompt_template = EXTRACT_PROMPT_MAP.get(doc_type, EXTRACT_PROMPT_GENERIC)

        try:
            prompt = prompt_template.format(text=_truncate(doc["text"]))
            response = llm.invoke([HumanMessage(content=prompt)])
            doc["extracted"] = _parse_json(response.content)
            doc["used_generic_extraction"] = doc_type == "unknown"  # flag for UI

            if trace:
                log_llm_call(trace, f"extract:{doc['filename']}", "llama-3.3-70b-versatile",
                             prompt, response.content,
                             {"filename": doc["filename"], "doc_type": doc_type,
                              "generic": doc_type == "unknown"})
        except Exception as e:
            doc["extracted"] = {"error": str(e)}

    return {**state, "documents": documents}


# Node 4: validate_node 

def validate_node(state: dict) -> dict:
    trace = state.get("trace")
    llm = _get_llm()
    documents = state.get("documents", [])

    for doc in documents:
        if not doc.get("extracted") or not doc.get("text"):
            doc["validation"] = {
                "anomalies": [], "overall_risk": "clean",
                "summary": "Document could not be processed."
            }
            continue

        try:
            # Known type to specific prompt. Unknown to generic fallback 
            doc_type = doc.get("doc_type", "unknown")
            if doc_type == "unknown":
                prompt = VALIDATE_PROMPT_GENERIC.format(
                    extracted_data=json.dumps(doc["extracted"], indent=2),
                    text=_truncate(doc["text"], 2000),
                )
            else:
                prompt = VALIDATE_PROMPT.format(
                    doc_type=doc_type,
                    extracted_data=json.dumps(doc["extracted"], indent=2),
                    text=_truncate(doc["text"], 2000),
                )

            response = llm.invoke([HumanMessage(content=prompt)])
            doc["validation"] = _parse_json(response.content)

            if trace:
                log_llm_call(trace, f"validate:{doc['filename']}", "llama-3.3-70b-versatile",
                             prompt, response.content,
                             {"filename": doc["filename"], "risk": doc["validation"].get("overall_risk")})
        except Exception as e:
            doc["validation"] = {
                "anomalies": [], "overall_risk": "unknown",
                "summary": f"Validation error: {e}"
            }

    return {**state, "documents": documents}

# Node 5: respond_node 

def respond_node(state: dict) -> dict:
    
    # Answer a chat question based on all processed documents using both per-doc and cross-doc context
    # Populates state["answer"]

    trace = state.get("trace")
    llm = _get_llm()
    question = state.get("question", "")
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", []) 

    # Build rich context string for each document
    context_parts = []
    for doc in documents:
        parts = [f"=== {doc['filename']} ==="]
        parts.append(f"Type: {doc.get('doc_type', 'unknown')}")

        classification = doc.get("classification", {})
        if classification.get("confidence"):
            parts.append(f"Classification confidence: {classification['confidence']:.0%}")

        extracted = doc.get("extracted", {})
        if extracted and not extracted.get("error"):
            parts.append(f"Extracted fields:\n{json.dumps(extracted, indent=2)}")

        validation = doc.get("validation", {})
        if validation:
            parts.append(f"Risk level: {validation.get('overall_risk', 'unknown')}")
            parts.append(f"Summary: {validation.get('summary', '')}")
            anomalies = validation.get("anomalies", [])
            if anomalies:
                parts.append(f"Anomalies ({len(anomalies)}):")
                for a in anomalies:
                    parts.append(f"  [{a.get('severity','?').upper()}] {a.get('issue','')} — {a.get('recommendation','')}")

        context_parts.append("\n".join(parts))

    documents_context = "\n\n".join(context_parts)

    # Build history string from last 10 messages (5 turns)
    history_text = ""
    for msg in chat_history[-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    try:
        prompt = CHAT_PROMPT.format(
            documents_context=documents_context,
            history=history_text,
            question=question,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

        if trace:
            log_llm_call(trace, "respond", "llama-3.3-70b-versatile",
                         prompt, answer, {"question": question})
    except Exception as e:
        answer = f"I encountered an error while answering: {e}"

    return {**state, "answer": answer}
