import os
import json
import tempfile
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from graph import run_processing_pipeline
from langfuse_config import get_or_create_trace

# FASTAPI app (for health checks and potential future API endpoints)

app = FastAPI(title="DocAgent API")

@app.get("/health")
def health():
    return {"status": "ok"}


# Sesion store to keep track of processed documents and their metadata across user interactions. 
# Keeps processed documents in memory per session (keyed by session_id)
# Each value: {"documents": [...], "file_paths": [...], "tmp_dir": str}

SESSION_STORE: dict = {}


# Core processing logic: runs the LangGraph pipeline for both document processing and chat interactions

# Runs load, classify, extract, validate nodes
def process_documents(files, session_state: dict):

    if not files:
        return "⚠️ Please upload at least one PDF.", session_state

    # Copy uploaded files to a temp dir tied to this session
    session_id = session_state.get("session_id", str(uuid.uuid4()))
    session_state["session_id"] = session_id

    tmp_dir = tempfile.mkdtemp(prefix=f"docagent_{session_id}_")
    file_paths = []
    for f in files:
        dest = os.path.join(tmp_dir, os.path.basename(f.name))
        shutil.copy(f.name, dest)
        file_paths.append(dest)

    # Run graph (no question needed for processing phase)
    result = run_processing_pipeline(
        file_paths=file_paths,
        question="Provide a brief overview of all uploaded documents.",
        #trace=trace,
    )

    documents = result.get("documents", [])
    session_state["documents"] = documents
    session_state["file_paths"] = file_paths
    session_state["tmp_dir"] = tmp_dir

    # Build display summary
    lines = [f"### ✅ Processed {len(documents)} document(s)\n"]
    for doc in documents:
        # doc_type = doc.get("doc_type", "unknown").replace("_", " ").title()
        raw_type = doc.get("doc_type", "unknown")
        if raw_type == "unknown":
            inferred = doc.get("extracted", {}).get("inferred_type", "Unknown")
            doc_type = f"Unknown (inferred: {inferred})"
            if doc.get("ocr_used"):
                lines.append("**OCR used** - scanned PDF detected")
        else:
            doc_type = raw_type.replace("_", " ").title()
            if doc.get("ocr_used"):
                lines.append("**OCR used** - scanned PDF detected")
        filename = doc.get("filename", "unknown")
        classification = doc.get("classification", {})
        confidence = classification.get("confidence", 0)
        validation = doc.get("validation", {})
        risk = validation.get("overall_risk", "unknown")
        summary = validation.get("summary", "")
        anomalies = validation.get("anomalies", [])

        risk_emoji = {"clean": "🟢", "low": "🟡", "medium": "🟠", "high": "🔴"}.get(risk, "⚪")

        lines.append(f"---\n**📄 {filename}**")
        lines.append(f"- **Type:** {doc_type} (confidence: {confidence:.0%})")
        lines.append(f"- **Risk:** {risk_emoji} {risk.title()}")
        lines.append(f"- **Summary:** {summary}")

        # Key extracted fields
        extracted = doc.get("extracted", {})
        if extracted and not extracted.get("error"):
            amount = extracted.get("total_amount")
            currency = extracted.get("currency", "")
            if amount:
                lines.append(f"- **Total Amount:** {currency} {amount:,.2f}" if isinstance(amount, (int, float)) else f"- **Total Amount:** {amount}")

            for field in ["vendor_name", "employee_name", "traveler_name", "booking_reference", "invoice_number"]:
                if extracted.get(field):
                    label = field.replace("_", " ").title()
                    lines.append(f"- **{label}:** {extracted[field]}")

        if anomalies:
            lines.append(f"\n⚠️ **{len(anomalies)} anomaly/anomalies flagged:**")
            for a in anomalies[:3]:  # show top 3
                sev = a.get("severity", "?").upper()
                sev_emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(sev, "⚪")
                lines.append(f"  {sev_emoji} [{sev}] {a.get('issue', '')}")

        lines.append("")

    lines.append("\n💬 **You can now ask questions about these documents in the chat below!**")
    return "\n".join(lines), session_state


def chat(message: str, history: list, session_state: dict):
    if not session_state.get("documents"):
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": "⚠️ Please upload and process documents first."}], session_state

    session_id = session_state.get("session_id", "anonymous")

    # get or initialize chat history
    chat_history = session_state.get("chat_history", [])

    result = run_processing_pipeline(
        file_paths=session_state.get("file_paths", []),
        question=message,
        #trace=trace,
        chat_history=chat_history,
    )

    answer = result.get("answer", "I could not generate an answer.")

    # history
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    session_state["chat_history"] = chat_history   # ← persist it

    return history + [{"role": "user", "content": message},
                      {"role": "assistant", "content": answer}], session_state

# Gradio UI with custom styling 

def build_ui():
    with gr.Blocks(
        title="DocAgent — Intelligent Document Processor",
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="zinc",
            neutral_hue="zinc",
            font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
            font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
        ),
        css="""
        /* global reset */
        body, .gradio-container {
            background: #0f1117 !important;
            color: #e2e8f0 !important;
        }

        /* header */
        #header {
            padding: 2rem 0 1.5rem 0;
            border-bottom: 1px solid #1e2433;
            margin-bottom: 1.5rem;
        }
        #header h1 {
            font-family: 'DM Mono', monospace;
            font-size: 1.6rem;
            font-weight: 600;
            color: #f8fafc;
            letter-spacing: -0.5px;
            margin: 0 0 0.3rem 0;
        }
        #header p {
            color: #64748b;
            font-size: 0.85rem;
            margin: 0;
        }

        /* panels */
        .panel {
            background: #161b27 !important;
            border: 1px solid #1e2d3d !important;
            border-radius: 10px !important;
            padding: 1.2rem !important;
        }

        /* zone */
        .upload-zone {
            border: 2px dashed #2d3f55 !important;
            background: #0d1520 !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease;
        }
        .upload-zone:hover { border-color: #4a90d9 !important; }

        /* process button */
        #process-btn {
            background: #1e3a5f !important;
            color: #93c5fd !important;
            border: 1px solid #2d5a8e !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.5px !important;
            border-radius: 6px !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.15s ease !important;
        }
        #process-btn:hover {
            background: #2d5a8e !important;
            border-color: #4a90d9 !important;
            color: #bfdbfe !important;
        }

        /* results panel */
        #results-panel {
            background: #0d1520 !important;
            border: 1px solid #1e2d3d !important;
            border-radius: 8px !important;
            font-size: 0.88rem !important;
            line-height: 1.7 !important;
            min-height: 200px;
        }

        /* chat */
        .chatbot {
            background: #0d1520 !important;
            border: 1px solid #1e2d3d !important;
            border-radius: 8px !important;
        }
        .chatbot .message.user { background: #1a2a3f !important; }
        .chatbot .message.bot  { background: #131c2b !important; }

        /* chat input */
        #chat-input textarea {
            background: #0d1520 !important;
            border: 1px solid #2d3f55 !important;
            color: #e2e8f0 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.88rem !important;
            border-radius: 6px !important;
        }

        /* labels / markdown */
        .label-text, label span {
            color: #64748b !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }
        .gr-markdown { color: #cbd5e1 !important; }

        /* divider */
        hr { border-color: #1e2433 !important; margin: 1rem 0 !important; }

        /* send button */
        #send-btn {
            background: #14532d !important;
            color: #86efac !important;
            border: 1px solid #166534 !important;
            border-radius: 6px !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.82rem !important;
        }
        #send-btn:hover {
            background: #166534 !important;
            color: #bbf7d0 !important;
        }

        /* footer */
        #footer {
            text-align: center;
            color: #334155;
            font-size: 0.75rem;
            padding: 1rem 0;
            border-top: 1px solid #1e2433;
            margin-top: 1.5rem;
        }
        """
    ) as demo:

        # session state to keep track of documents across interactions
        session_state = gr.State({})

        # header
        with gr.Row(elem_id="header"):
            gr.Markdown("""
# ⬡ DocAgent
**Intelligent multi-document processor** · Powered by LangGraph + Groq Llama 3.3 · Traced by Langfuse
""")

        # main layout with two columns: left for upload + results, right for chat
        with gr.Row():

            # LEFT - upload + results
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("#### 📂 Upload Documents")
                gr.Markdown(
                    "<span style='color:#64748b;font-size:0.82rem'>Accepts PDF invoices, expense reports, and booking confirmations</span>",
                    sanitize_html=False
                )
                file_upload = gr.File(
                    file_types=[".pdf"],
                    file_count="multiple",
                    label="",
                    elem_classes=["upload-zone"],
                )
                process_btn = gr.Button("⚡ Process Documents", elem_id="process-btn")
                gr.Markdown("---")
                gr.Markdown("#### 🔍 Analysis Results")
                results_display = gr.Markdown(
                    "*Upload PDFs and click Process to begin.*",
                    elem_id="results-panel",
                )

            # RIGHT - chat
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("#### 💬 Ask Questions")
                gr.Markdown(
                    "<span style='color:#64748b;font-size:0.82rem'>Ask per-document or cross-document questions after processing</span>",
                    sanitize_html=False
                )
                chatbot = gr.Chatbot(
                    label="",
                    height=420,
                    elem_classes=["chatbot"],
                    show_label=False,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="e.g. What is the total amount on the invoice?  /  Which doc has the highest risk?",
                        show_label=False,
                        elem_id="chat-input",
                        scale=5,
                    )
                    send_btn = gr.Button("Send →", elem_id="send-btn", scale=1)

                gr.Markdown("""
<div style='color:#334155;font-size:0.78rem;margin-top:0.5rem'>
💡 Try: <em>"Summarize all documents"</em> · <em>"Are there anomalies across all docs?"</em> · <em>"What is the total spend?"</em>
</div>
""", sanitize_html=False)

        # footer 
        gr.Markdown(
            "<div id='footer'>DocAgent · LangGraph multi-node pipeline · Groq Llama 3.3 70B · Langfuse observability</div>",
            sanitize_html=False
        )

        # event wiring 
        process_btn.click(
            fn=process_documents,
            inputs=[file_upload, session_state],
            outputs=[results_display, session_state],
        )

        send_btn.click(
            fn=chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state],
        ).then(lambda: "", outputs=chat_input)

        chat_input.submit(
            fn=chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state],
        ).then(lambda: "", outputs=chat_input)

    return demo


# Mount Gradio on FastAPI and launch

demo = build_ui()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting on 0.0.0.0:{port}", flush=True)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)