# contains all necessary prompts

CLASSIFY_PROMPT = """You are a document classification expert.

Analyze the following document text and classify it into exactly one of these categories:
- invoice: A bill for goods or services with amounts owed
- expense_report: A record of business expenses submitted for reimbursement
- booking_confirmation: A travel/hotel/flight booking confirmation with reservation details
- unknown: Cannot be determined

Respond ONLY with raw JSON, no markdown or explanation:
{{
  "doc_type": "<invoice|expense_report|booking_confirmation|unknown>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Document text:
{text}
"""

EXTRACT_PROMPT_INVOICE = """You are a financial document extraction expert.

Extract all key fields from this invoice. If a field is not present, use null.

Respond ONLY with raw JSON, no markdown:
{{
  "vendor_name": "<string or null>",
  "invoice_number": "<string or null>",
  "invoice_date": "<YYYY-MM-DD or null>",
  "due_date": "<YYYY-MM-DD or null>",
  "total_amount": <number or null>,
  "currency": "<string or null>",
  "line_items": [{{"description": "<string>", "amount": <number>}}],
  "tax_amount": <number or null>,
  "billing_party": "<string or null>"
}}

Invoice text:
{text}
"""

EXTRACT_PROMPT_EXPENSE = """You are a financial document extraction expert.

Extract all key fields from this expense report. If a field is not present, use null.

Respond ONLY with raw JSON, no markdown:
{{
  "employee_name": "<string or null>",
  "department": "<string or null>",
  "report_date": "<YYYY-MM-DD or null>",
  "total_amount": <number or null>,
  "currency": "<string or null>",
  "expenses": [{{"category": "<string>", "description": "<string>", "amount": <number>, "date": "<string>"}}],
  "purpose": "<string or null>",
  "approved_by": "<string or null>"
}}

Expense report text:
{text}
"""

EXTRACT_PROMPT_BOOKING = """You are a travel document extraction expert.

Extract all key fields from this booking confirmation. If a field is not present, use null.

Respond ONLY with raw JSON, no markdown:
{{
  "booking_reference": "<string or null>",
  "traveler_name": "<string or null>",
  "booking_date": "<YYYY-MM-DD or null>",
  "travel_date": "<YYYY-MM-DD or null>",
  "return_date": "<YYYY-MM-DD or null>",
  "destination": "<string or null>",
  "origin": "<string or null>",
  "total_amount": <number or null>,
  "currency": "<string or null>",
  "hotel_name": "<string or null>",
  "flight_number": "<string or null>",
  "provider": "<string or null>"
}}

Booking confirmation text:
{text}
"""

EXTRACT_PROMPT_GENERIC = """You are a document analysis expert.

Extract the most important information from this document. Identify what type of document it is and pull out all key fields that would be useful to someone asking questions about it.

Respond ONLY with raw JSON, no markdown:
{{
  "inferred_type": "<what kind of document this appears to be>",
  "title": "<document title or heading if present, else null>",
  "date": "<most relevant date found, YYYY-MM-DD or null>",
  "parties": ["<person or organization names mentioned>"],
  "key_fields": {{
    "<field_name>": "<value>"
  }},
  "summary": "<2-3 sentence plain English summary of what this document is about>"
}}

For key_fields, extract whatever is most important for THIS specific document type.
Examples: for a resume extract skills/experience/education, for a contract extract terms/parties/obligations, for a research paper extract findings/methodology/conclusions.

Document text:
{text}
"""

VALIDATE_PROMPT = """You are a document validation and anomaly detection expert.

Given this {doc_type} document with the following extracted data, identify any anomalies, risks, or issues.

Look for:
- Missing critical fields that should be present
- Suspicious or unusually high amounts
- Date inconsistencies (e.g. due date before invoice date)
- Incomplete or vague descriptions
- Round-number amounts that may indicate estimates rather than actuals
- Any other red flags for this document type

Extracted data:
{extracted_data}

Raw document text:
{text}

Respond ONLY with raw JSON, no markdown:
{{
  "anomalies": [
    {{
      "severity": "<high|medium|low>",
      "field": "<field name or 'general'>",
      "issue": "<description of the issue>",
      "recommendation": "<what to do about it>"
    }}
  ],
  "overall_risk": "<high|medium|low|clean>",
  "summary": "<2-3 sentence plain English summary of this document>"
}}
"""

VALIDATE_PROMPT_GENERIC = """You are a document review expert.

Given this document and its extracted data, provide a brief quality assessment.

Look for:
- Missing information that would typically be present in this document type
- Any inconsistencies or unusual content
- Overall completeness and clarity

Extracted data:
{extracted_data}

Raw document text:
{text}

Respond ONLY with raw JSON, no markdown:
{{
  "anomalies": [
    {{
      "severity": "<high|medium|low>",
      "field": "<field name or 'general'>",
      "issue": "<description>",
      "recommendation": "<what to do>"
    }}
  ],
  "overall_risk": "<high|medium|low|clean>",
  "summary": "<2-3 sentence plain English summary of this document>"
}}
"""

CHAT_PROMPT = """You are a helpful document analysis assistant. You have processed the following documents:

{documents_context}

Conversation so far:
{history}

Answer the user's question accurately based on the documents and conversation history above.
- For per-document questions, cite the filename.
- For cross-document questions, compare across all documents.
- If a question refers to something mentioned earlier in the conversation (e.g. 'tell me more about that', 'the first one'), use the conversation history to resolve it.
- If the answer cannot be found in the documents, say so clearly.

User question: {question}
"""
