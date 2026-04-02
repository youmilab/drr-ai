import io
import os
import json
import time
import datetime
from pathlib import Path

import anthropic
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text as pdf_extract_text

load_dotenv()

app = FastAPI(title="DRR-AI Audit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your domain in production
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
METADATA_FILE = Path(__file__).parent / "usage_log.jsonl"

# ── IES/NCES compliance rules ─────────────────────────────────────────────────

IES_RULES = """
You are an expert in IES/NCES restricted-use data Disclosure Risk Review (DRR).
Apply ALL of the following rules strictly:

1. UNWEIGHTED NS: Only applies when the manuscript contains unweighted sample sizes derived
   from restricted-use data. If no raw counts from restricted-use data are reported, do NOT
   flag this rule. When applicable: all unweighted sample sizes must be rounded to the nearest
   10 (nearest 50 for ECLS-B). Near-zero cells must show "<10", never "0".
2. ROUNDING STATEMENT: Only applies when the manuscript contains raw numbers (sample sizes,
   counts) derived from restricted-use data. If no such numbers are present, do NOT flag this
   rule. When applicable: manuscripts must include the statement "Numbers are unweighted and
   rounded to nearest 10" (or nearest 50 for ECLS-B).
3. PERCENTAGES: Percentages with disclosure risk must be rounded to tenths (e.g., 23.4%).
   Proportions must be rounded to hundredths (e.g., 0.08).
4. WEIGHTED/UNWEIGHTED LABELING: Only applies when the manuscript explicitly reports sample
   sizes (raw counts or Ns) derived from restricted-use data. When reported, the sample size
   must be clearly labeled as weighted or unweighted. Figures showing statistical curves,
   trends, or analyses do NOT need to specify weighted or unweighted status and must NOT be
   flagged for this rule.
5. SOURCE NOTES: Only applies to tables and figures that present point estimates or raw numbers
   derived from restricted-use datasets (e.g., tables with cell values, figures with data
   points, bar charts with frequencies or percentages). To determine whether a figure contains
   such values, read the figure description and surrounding text — just as a human reviewer
   would. Do NOT flag a figure for a missing SOURCE note if it contains no point estimates or
   raw numbers from restricted-use data, even if the figure is related to a restricted-use
   data study. Figures showing only curves, trends, theoretical frameworks, or causal diagrams
   must NOT be flagged. When applicable, the SOURCE note format is:
   "SOURCE: U.S. Department of Education, National Center for Education Statistics, [Survey Name]"
6. INTERNAL CONSISTENCY: Numbers in the text body must match numbers in tables and figures.
   Flag any discrepancies.
7. TABLE CVs: Only applies when standard errors are reported. If no standard errors are
   present, do NOT flag this rule. CV is defined as: CV = (Standard Error / Estimate) × 100.
   When applicable: CVs above 30% must be flagged with "!" and a table note; CVs above 50%
   must be suppressed with "‡" and a table note.
8. ROUNDING STANDARDS: Summary percentages max 1 decimal place; reference percentages
   max 2 decimal places; standard errors must show 1 more decimal place than their estimates.
"""

# ── ICPSR compliance rules ────────────────────────────────────────────────────

ICPSR_RULES = """
You are an expert in ICPSR restricted-use data Disclosure Risk Review (DRR).
Apply DUA-based rules appropriate for high-risk ICPSR datasets (those involving
children, mental health data, criminal history records, or other sensitive populations):

1. SMALL CELL SUPPRESSION: Any cell with fewer than 5 respondents must be suppressed or masked.
2. COMPLEMENTARY SUPPRESSION: When one cell is suppressed, complementary cells that would
   allow back-calculation must also be suppressed.
3. GEOGRAPHIC IDENTIFIERS: Sub-county geographic identifiers may not appear for small populations.
4. RARE CHARACTERISTICS: Flag any combination of variables that could uniquely identify
   individuals (e.g., rare disability status + age + race + location).
5. AGGREGATE MINIMUMS: No group-level statistics may be based on fewer than 10 observations.
6. DUA COMPLIANCE: Verify that the manuscript complies with stated DUA restrictions on
   publication scope and re-identification risk.
"""


def build_prompt(manuscript_text: str, framework: str) -> str:
    rules = IES_RULES if framework == "ies" else ICPSR_RULES
    # Cap text to avoid exceeding context window
    truncated = manuscript_text[:40000]
    return f"""{rules}

---

MANUSCRIPT TEXT TO AUDIT:
{truncated}

---

TASK:
Review the manuscript above and produce a structured compliance audit report.

CRITICAL INSTRUCTIONS:
- Do NOT quote specific restricted numbers, cell values, or direct text from the manuscript.
  Describe findings generically (e.g., "An unrounded sample size was found in Table 2"
  rather than repeating the actual value).
- Each finding must be actionable and cite the specific rule violated.
- Location should be as precise as possible (e.g., "Table 2, Row 3" or "Page 4, Paragraph 2").
- severity "High" = direct disclosure risk; "Review" = likely violation needing human check;
  "Low" = minor formatting or labeling issue.
- ONLY include a finding if there is an actual violation or genuine uncertainty requiring
  author action. Do NOT include findings where the conclusion is that no action is needed,
  no violation exists, or a figure/table is confirmed exempt from a rule. If you verify
  something is compliant or exempt, omit it from the findings entirely — do NOT include it
  at any severity level (not "Low", not "Review", not "High") for transparency, completeness,
  or any other reason. Exempt means absent from the findings, period.

OUTPUT FORMAT: Respond with ONLY valid JSON — no markdown fences, no preamble:
{{
  "status": "PASS",
  "item_count": 0,
  "findings": []
}}

OR if issues found:
{{
  "status": "REVIEW NEEDED",
  "item_count": 3,
  "findings": [
    {{
      "severity": "High",
      "location": "Table 2, Header Row",
      "flag": "An unrounded unweighted sample size was found",
      "rule": "IES requires unweighted sample sizes rounded to the nearest 10",
      "recommendation": "Round the N to the nearest 10"
    }}
  ]
}}

"status" rules:
- "PASS" if findings is empty
- "HIGH RISK" if any finding has severity "High"
- "REVIEW NEEDED" otherwise
"""


# ── Text extraction (in-memory only — no disk writes) ─────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    buf = io.BytesIO(file_bytes)
    text = pdf_extract_text(buf)
    return text or ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    buf = io.BytesIO(file_bytes)
    doc = Document(buf)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# ── Metadata logging (zero PII) ───────────────────────────────────────────────

def log_metadata(
    framework: str,
    tokens_in: int,
    tokens_out: int,
    processing_time: float,
    status: str,
) -> None:
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "framework_used": framework,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "tokens_processed": tokens_in + tokens_out,
        "processing_time_seconds": round(processing_time, 2),
        "overall_status": status,
    }
    with open(METADATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/audit")
async def audit_manuscript(
    file: UploadFile = File(...),
    framework: str = Form(default="ies"),
):
    start_time = time.monotonic()

    if framework not in ("ies", "icpsr"):
        raise HTTPException(status_code=400, detail="framework must be 'ies' or 'icpsr'")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured. Contact the site administrator."
        )

    # Read file into RAM — never written to disk
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit.")

    filename = (file.filename or "").lower()
    content_type = file.content_type or ""

    try:
        if filename.endswith(".pdf") or "pdf" in content_type:
            manuscript_text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx") or "wordprocessingml" in content_type:
            manuscript_text = extract_text_from_docx(file_bytes)
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Please upload a PDF or DOCX."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")
    finally:
        del file_bytes  # Free memory immediately

    if len(manuscript_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail=(
                "File appears empty or unreadable. "
                "If uploading a scanned PDF, please use a text-based PDF instead."
            )
        )

    prompt = build_prompt(manuscript_text, framework)
    del manuscript_text  # Free memory before API call

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI API error: {e}")

    raw = message.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Received an unexpected response from the AI. Please try again."
        )

    elapsed = time.monotonic() - start_time
    log_metadata(
        framework=framework,
        tokens_in=message.usage.input_tokens,
        tokens_out=message.usage.output_tokens,
        processing_time=elapsed,
        status=report.get("status", "UNKNOWN"),
    )

    return report
