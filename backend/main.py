import io
import os
import json
import time
import datetime
from pathlib import Path

import anthropic
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text as pdf_extract_text

load_dotenv()

app = FastAPI(title="DRR-AI Audit API")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
METADATA_FILE = Path(__file__).parent / "usage_log.jsonl"

# ── IES/NCES compliance rules ─────────────────────────────────────────────────

IES_RULES = """
You are an expert in IES/NCES restricted-use data Disclosure Risk Review (DRR).
Apply ALL of the following rules strictly:

SEVERITY LEVELS ARE FIXED — do not use your own judgment to assign severity. Each rule
specifies its required severity below. Always use exactly the severity stated.

1. UNWEIGHTED NS [severity: High for unrounded counts; severity: Review for ambiguous zeros]:
   Only applies when the manuscript contains unweighted sample sizes derived from
   restricted-use data. If no raw counts from restricted-use data are reported, do NOT
   flag this rule. When applicable: all unweighted sample sizes must be rounded to the nearest
   10 for all datasets EXCEPT ECLS-B, where the rounding increment is 50. "Nearest 50" refers
   ONLY to the rounding increment for ECLS-B — it is NOT a cell size threshold and must NOT
   be used to flag cells as small. A cell with N=43 in a non-ECLS-B dataset is not small; it
   simply needs to be rounded to 40. Small cells are defined solely as cells with 1–9
   observations, which must show "<10" instead of the exact count. Flag unrounded counts and
   near-zero cells shown as exact numbers as severity "High". A cell showing "0" must NOT be
   flagged if context indicates a true zero. Only flag "0" as severity "Review" if the
   surrounding text or table suggests the group may have some participants but the count
   appears to be a near-zero incorrectly reported as zero.
2. ROUNDING STATEMENT [severity: Review]: Only applies when the manuscript contains raw
   numbers (sample sizes, counts) derived from restricted-use data. If no such numbers are
   present, do NOT flag this rule. When applicable: manuscripts must include the statement
   "Numbers are unweighted and rounded to nearest 10" (or nearest 50 for ECLS-B).
3. PERCENTAGES AND ROUNDING [severity: Low]: Only applies to values that are explicitly
   percentages (e.g., 23.4%, values followed by "%" or described as "percent") or proportions
   (e.g., 0.08, described as a proportion or rate). Do NOT apply this rule to means, scores,
   test statistics, regression coefficients, standard errors, or any other type of estimate —
   even if they happen to be decimal numbers. When applicable: summary percentages must be
   rounded to tenths (max 1 decimal place); reference percentages max 2 decimal places;
   proportions must be rounded to hundredths (e.g., 0.08).
4. WEIGHTED/UNWEIGHTED LABELING [severity: Review]: Only applies when the manuscript
   explicitly reports a sample size value (a raw count or N) derived from restricted-use
   data. That N must be clearly labeled as weighted or unweighted. This rule applies ONLY
   to sample size counts (Ns) — it does NOT apply to estimates, means, percentages, standard
   errors, regression coefficients, or any other statistic. Figures, charts, or tables that
   show statistical results but no raw sample size counts must NOT be flagged for this rule.
5. SOURCE NOTES [severity: Review]: Only applies to tables and figures that present point
   estimates or raw numbers derived from restricted-use datasets (e.g., tables with cell
   values, figures with data points, bar charts with frequencies or percentages). Read the
   figure caption, title, and surrounding text to determine whether it contains such values.
   If the text describes a figure as a causal diagram, DAG, directed acyclic graph,
   theoretical framework, conceptual model, or any figure with no numeric data values, you
   must conclude it is exempt and omit it from findings entirely. Do NOT defer this
   determination to a human reviewer — make the call yourself. Do NOT flag figures or tables
   that do not use restricted-use data. When applicable, the SOURCE note format is:
   "SOURCE: U.S. Department of Education, National Center for Education Statistics, [Survey Name]"
6. INTERNAL CONSISTENCY [severity: Review]: Only applies when the manuscript text explicitly
   attributes a specific numeric value to a particular table or figure (e.g., "as shown in
   Table 1," "Table 2 shows that," "see Figure 3"). To flag this rule you must confirm ALL
   THREE: (a) the text explicitly cites a specific table or figure as the source of the value,
   (b) that exact value can be found in the cited table or figure, AND (c) the value in the
   text does not match the value in the table or figure. Do NOT flag a number that appears in
   the text without an explicit reference to a table or figure.
7. TABLE CVs [severity: High if CV > 50%; severity: Review if 30% < CV ≤ 50%]: Only applies
   when standard errors are reported in tables. If no standard errors are present, do NOT flag
   this rule. For every estimate/SE pair in a table, calculate CV = (SE / Estimate) × 100.
   STRICT THRESHOLD RULE: you may only produce a CV finding if the computed CV is strictly
   greater than 30%. If CV ≤ 30%, that row is fully compliant — omit it from findings with
   no exception. If you are uncertain whether CV exceeds 30%, compute it again; if still
   uncertain, do NOT flag it. Never flag a CV finding speculatively or as a precaution.
   CVs above 30% require "!" and a table note (severity "Review"); CVs above 50% require
   "‡" and suppression (severity "High"). When flagging, state the exact numeric CV computed
   and include CV = (SE / Estimate) × 100 in the recommendation. Never flag from text prose.
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
- CONSOLIDATE duplicate findings: if multiple rows or cells in the same table share the
  same violation and the same rule, report them as a single finding with the table as the
  location (e.g., "Table 1"). Do NOT create a separate finding for each row or cell.
- ONE ISSUE PER FINDING: each finding must address exactly one rule and one specific
  violation. Never combine two separate issues (e.g., a CV concern and a zero-cell concern)
  into a single finding. If a single row has two different violations, create two separate
  findings for that row.
- Location should be as precise as possible (e.g., "Table 2, Row 3" or "Page 4, Paragraph 2"),
  but use the table-level location when consolidating repeated violations across rows.
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
    blocks = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag
        if tag == "p":
            from docx.oxml.ns import qn
            text = "".join(node.text for node in block.iter(qn("w:t")) if node.text)
            if text.strip():
                blocks.append(text)
        elif tag == "tbl":
            for row in block.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                cells = []
                for cell in row.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                    from docx.oxml.ns import qn as _qn
                    cell_text = "".join(n.text for n in cell.iter(_qn("w:t")) if n.text)
                    if cell_text.strip():
                        cells.append(cell_text.strip())
                if cells:
                    blocks.append("\t".join(cells))
    return "\n".join(blocks)


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
            temperature=0,
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
