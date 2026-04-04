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
   WHAT THIS RULE COVERS: Only unweighted sample size COUNTS — values explicitly labeled
   as N, n, sample size, or count of respondents. This rule does NOT apply to any of the
   following, which must never be flagged under Rule 1: means, standard deviations, standard
   errors, test scores, IRT scores, scale scores, min/max values, percentages, proportions,
   regression coefficients, F-statistics, t-statistics, p-values, or any other statistical
   estimate. Point estimates of statistics (means, min, max, scores) do not identify
   individuals and back-calculation is not feasible — they are exempt from this rule entirely.
   Weighted sample sizes (population estimates inflated by survey weights) are also exempt.
   ROUNDING CHECK: A number ending in 0 is ALWAYS considered rounded — do not flag it,
   do not question it, do not treat it as suspicious. Only flag a count whose last digit
   is 1–9 (e.g., 43, 97, 182). If all counts end in 0, there is NO violation.
   SMALL CELL: A small cell is ONLY a count of 1–9. N=10 and above is NEVER a small cell.
   ECLS-B ONLY: rounding increment is 50 (not a size threshold).
   DEGREES OF FREEDOM: Check df2 in F(df1, df2) and df in t(df) — if df does not end in 0,
   flag as High. Check this in every run without exception.
   ZERO CELLS: "0" is correct for true zeros. Flag "0" as Review only if context suggests
   1–9 participants may be present but incorrectly reported as zero.
2. ROUNDING STATEMENT [severity: Review]: Only applies when the manuscript contains
   unweighted sample sizes derived from restricted-use data. If no such numbers are present,
   do NOT flag this rule. The manuscript must include a statement conveying that sample sizes
   are unweighted and rounded to the nearest 10 (or nearest 50 for ECLS-B). The exact wording
   does not need to match precisely — any paraphrase that clearly communicates both concepts
   (unweighted AND rounded to nearest 10) is acceptable and must NOT be flagged. A single
   global statement anywhere in the manuscript (e.g., in the methods section, a footnote, or
   a table note) is sufficient to cover all sample sizes throughout — authors are NOT required
   to repeat "unweighted" next to every individual N in the text. Search the entire manuscript;
   do not flag if the statement appears anywhere.
3. PERCENTAGES AND ROUNDING [severity: Low]: Only applies to values that are explicitly
   percentages (e.g., 23.4%, values followed by "%" or described as "percent") or proportions
   (e.g., 0.08, described as a proportion or rate). Do NOT apply this rule to means, scores,
   test statistics, regression coefficients, standard errors, or any other type of estimate —
   even if they happen to be decimal numbers. When applicable: summary percentages must be
   rounded to tenths (max 1 decimal place); reference percentages max 2 decimal places;
   proportions must be rounded to hundredths (e.g., 0.08).
4. WEIGHTED/UNWEIGHTED LABELING [severity: Review]: Only applies to TABLES that report
   sample size values (raw counts or Ns) derived from restricted-use data. Each such table
   must include a note or label indicating whether the Ns are weighted or unweighted.
   If a global rounding statement exists in the manuscript (see Rule 2), in-text sample sizes
   are already covered by that statement — do NOT flag individual in-text Ns for lacking a
   weighted/unweighted label. This rule applies ONLY to sample size counts (Ns) in tables —
   it does NOT apply to estimates, means, percentages, standard errors, regression
   coefficients, or any other statistic, nor to figures showing statistical results.
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
6. INTERNAL CONSISTENCY [severity: Review for exact mismatches; severity: Low for
   approximate or derived values]: Two sub-cases apply:
   (a) EXACT MISMATCH [severity: Review]: The text explicitly cites a specific table or
   figure (e.g., "as shown in Table 1") and restates a specific value from it, but the
   value in the text does not match the value in the table. All three conditions must be
   confirmed: the text names the source table/figure, the value appears in that table/figure,
   and the two values disagree. Do NOT flag potential ambiguity or uncertainty — only flag
   a confirmed, clear disagreement between two specific values.
   (b) APPROXIMATE OR DERIVED VALUE [severity: Low]: The text states an approximate or
   computed value (e.g., a gap, difference, or summary described as "approximately X") that
   is derived from table values but may not match exactly. Flag only if the approximation
   is materially inconsistent with what the table shows. If the values appear consistent or
   plausibly consistent, do NOT flag at any severity level — do not flag merely to encourage
   double-checking when no clear inconsistency is present.
   In both cases: do NOT quote the specific numeric value from the manuscript in the flag or
   recommendation — describe the issue generically (e.g., "an approximate gap described in
   the text" rather than the actual number).
7. TABLE CVs [severity: High if CV > 50%; severity: Review if 30% < CV ≤ 50%]: Only applies
   when standard errors are reported in tables. If no standard errors are present, do NOT flag
   this rule. For every estimate/SE pair in a table, calculate CV = (SE / Estimate) × 100.
   STRICT THRESHOLD RULE: you may only produce a CV finding if the computed CV is strictly
   greater than 30%. If CV ≤ 30%, that row is fully compliant — omit it from findings with
   no exception. If you are uncertain whether CV exceeds 30%, compute it again; if still
   uncertain, do NOT flag it. Never flag a CV finding speculatively or as a precaution.
   CVs above 30% require "!" and a table note (severity "Review"); CVs above 50% require
   "‡" and suppression (severity "High"). When flagging, state the exact numeric CV computed
   and include CV = (SE / Estimate) × 100 in the recommendation.
   LOCATION: CV findings must cite the specific table where the SE and estimate appear (e.g.,
   "Table 2"). Never cite a text section (e.g., "Section 3.2") as the location — if the
   values are in a table, name the table. Never produce a CV finding from text prose alone.
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
  violation. Never combine two separate issues into a single finding.
- Location should be as precise as possible (e.g., "Table 2, Row 3" or "Page 4, Paragraph 2"),
  but use the table-level location when consolidating repeated violations across rows.
- severity "High" = direct disclosure risk; "Review" = likely violation needing human check;
  "Low" = minor formatting or labeling issue.
- "item_count" must exactly equal the number of objects in the "findings" array. Count them
  carefully before writing the JSON — do not leave item_count as 0 if findings is non-empty.
- ONLY include a finding if there is an actual violation or genuine uncertainty requiring
  author action. Do NOT include findings where the conclusion is that no action is needed,
  no violation exists, or a figure/table is confirmed exempt from a rule. If you verify
  something is compliant or exempt, omit it from the findings entirely — do NOT include it
  at any severity level. Exempt means absent from the findings, period.
- NO-ACTION GATE: Do not include any finding whose recommendation would be "no action
  needed" or "already compliant." Every finding must require the author to take action.
- INTERNAL CHECKS ONLY: Before including a finding, silently verify: (a) for any N, its
  last digit must be 1–9 to be unrounded — if it ends in 0 it is compliant; (b) a small
  cell is only N=1 through N=9 — N≥10 is never a small cell; (c) CV must be computed as
  (SE/Estimate)×100 and must exceed 30% to flag. Do NOT output these checks — apply them
  silently and only output the final JSON.

OUTPUT FORMAT: Your entire response must be ONLY valid JSON with no text before or after it.
Do not write any explanation, reasoning, preamble, or commentary — start your response
with {{ and end with }}. No markdown fences:
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
            max_tokens=4096,
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
    except json.JSONDecodeError as e:
        import logging
        logging.error("JSON parse error: %s\nRaw response (first 500 chars): %s", e, raw[:500])
        raise HTTPException(
            status_code=502,
            detail=f"Received an unexpected response from the AI. Parse error: {e}. Response preview: {raw[:200]}"
        )

    # Always enforce item_count to match the actual findings array length
    findings = report.get("findings", [])
    report["item_count"] = len(findings)

    elapsed = time.monotonic() - start_time
    log_metadata(
        framework=framework,
        tokens_in=message.usage.input_tokens,
        tokens_out=message.usage.output_tokens,
        processing_time=elapsed,
        status=report.get("status", "UNKNOWN"),
    )

    return report
