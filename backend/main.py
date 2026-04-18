import io
import os
import re
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

INTERNAL TRIAGE (do this silently in your head — do NOT write it out):
Before applying any rule, mentally classify every table and figure in the manuscript as
either IN-SCOPE (IES/NCES restricted-use data) or OUT-OF-SCOPE (exempt). Only IN-SCOPE
items may be flagged. Never flag an OUT-OF-SCOPE item under any rule.
Do not output this classification. Do not explain your reasoning. Proceed directly to
the JSON output after completing this mental triage.

CLASSIFICATION RULES — apply in this exact order:

  RULE A (SOURCE note present → IN-SCOPE): A table or figure is IN-SCOPE if it has a
  SOURCE note that names "U.S. Department of Education, National Center for Education
  Statistics" or a known IES/NCES dataset (NAEP, ECLS-K, ECLS-B, ELS, HSLS, SSES,
  NSCG, or similar). This is the strongest signal — apply rules to this item.

  RULE B (SOURCE note absent → OUT-OF-SCOPE by default): A table or figure with NO
  SOURCE note is OUT-OF-SCOPE and EXEMPT from all rules, UNLESS the sentence or heading
  that directly introduces the table (not a distant earlier paragraph) explicitly names
  a known IES/NCES restricted-use dataset by its acronym (NAEP, ECLS, ELS, HSLS, etc.).
  If the introducing sentence cites a published paper (e.g., "Wong et al., 2007"),
  names a state program, or uses any data source other than a named IES/NCES dataset,
  the table is OUT-OF-SCOPE — do not flag it under any rule.

  RULE C (default): When the data source cannot be clearly determined, treat the item
  as OUT-OF-SCOPE. Never flag an item under uncertainty.

CERTAINTY REQUIREMENT: You must be 100% certain a table uses IES/NCES restricted-use
data before flagging it. If you have any doubt, classify it as OUT-OF-SCOPE.
CITATION SIGNAL: If you identify a published paper citation (e.g., "Author et al.,
YYYY" or "(YYYY)") as the data source for a table — even alongside words like
"restricted-use" or "program data" — that table is OUT-OF-SCOPE. Published papers
are not IES/NCES restricted-use files.
SELF-CHECK BEFORE FLAGGING: For every finding you are about to write, ask: "Can I
identify a SOURCE note or direct dataset name (NAEP, ECLS, etc.) that confirms this
table uses IES/NCES restricted-use data?" If the answer is no, discard the finding.

Apply ALL rules below ONLY to IN-SCOPE items. Ignore OUT-OF-SCOPE items entirely.

1. UNWEIGHTED NS [severity: High for unrounded counts; severity: Review for ambiguous zeros]:
   WHAT THIS RULE COVERS: Only unweighted counts of individual respondents or participants
   drawn directly from an IES/NCES restricted-use dataset — values labeled as N, n, sample
   size, number of students, number of respondents, or similar descriptions of people
   counted from restricted-use data.
   ANALYTICAL QUANTITIES THAT ARE NEVER SAMPLE SIZES: The following must NEVER be flagged
   under Rule 1 regardless of their numeric value, because they are methodological or
   analytical quantities, not counts of respondents from restricted-use data:
     - Number of strata (e.g., J, J_k, K, number of matched sets or matched pairs)
     - Number of clusters, schools, classrooms, or geographic units used as analytical units
     - Number of iterations, replications, bootstraps, or simulations
     - Number of covariates, variables, or model parameters
     - Index values, group labels, or factor identifiers
   These quantities describe the structure of the analysis, not the count of individuals
   in the restricted-use dataset, and are exempt from rounding requirements.
   This rule also does NOT apply to: means, standard deviations, standard errors, test
   scores, IRT scores, scale scores, min/max values, percentages, proportions, regression
   coefficients, F-statistics, t-statistics, p-values, or any other statistical estimate.
   Weighted sample sizes (population estimates inflated by survey weights) are also exempt.
   ROUNDING CHECK: To determine whether a number ends in 0, look only at the LAST digit of
   the integer portion of the number (ignoring commas, spaces, and any decimal portion).
   Examples of COMPLIANT counts (last digit = 0, do NOT flag):
     410 → last digit 0 ✓
     4130 → last digit 0 ✓
     1540 → last digit 0 ✓
     2,110 → last digit 0 ✓
     5,200 → last digit 0 ✓
     590 → last digit 0 ✓
     10,020 → last digit 0 ✓
     8,530 → last digit 0 ✓
   Examples of NON-COMPLIANT counts (last digit ≠ 0, flag these):
     43, 97, 182, 2,113, 5,847, 4,131.
   A number whose last digit is 0 is ALWAYS considered rounded — do not flag it, do not
   question it, do not treat it as suspicious. Only flag a count whose last digit is 1–9.
   If all counts end in 0, there is NO Rule 1 violation.
   MANDATORY PRE-FLAG CHECK: Before including any Rule 1 finding, identify the exact
   number being flagged and confirm its last digit is 1–9. If the last digit is 0, discard
   the finding entirely — do not include it at any severity level.
   SMALL CELL: A small cell is ONLY a count of 1–9. N=10 and above is NEVER a small cell —
   this includes N=10, N=20, N=30, N=40, and all larger values. Do not flag any N≥10 as a
   small cell regardless of how small it appears relative to other cells in the table.
   ECLS-B ONLY: rounding increment is 50 (not a size threshold).
   DEGREES OF FREEDOM: Check df2 in F(df1, df2) and df in t(df) — if df does not end in 0,
   flag as High. Check this in every run without exception.
   SMALL CELLS THAT ROUND TO ZERO (counts 1–4): When a count of 1–4 is rounded to the
   nearest 10 it becomes 0, which falsely implies the cell is empty. IES requires these
   cells to be presented as "<10" — never as "0", "#", "‡", "*", or any other symbol.
   MANDATORY CHECK — perform this for every restricted-use table:
     Step 1. Read every table note and footnote word by word.
     Step 2. If you find ANY of the following phrases, flag it immediately as a Rule 1
             violation (severity: High):
               - "#s are rounded to zero"
               - "rounded to zero"
               - "# = rounded to zero"
               - "# represents counts that round to zero"
               - any phrasing that equates a symbol (# ‡ * etc.) with rounding to 0
     Step 3. The flag must say: the table note uses [symbol] to denote counts that round
             to zero, but IES requires these cells to be presented as "<10" instead.
     Step 4. The recommendation must say: replace [symbol] with "<10" in all affected
             cells and update the table note to state counts that round to zero are
             presented as "<10".
   Do NOT skip this check. Do NOT treat "#s are rounded to zero" as acceptable notation.
   The required IES notation for counts that round to zero is exclusively "<10".
   ZERO CELLS: "0" is correct for true zeros. Flag "0" as Review only if context suggests
   1–9 participants may be present but incorrectly reported as zero.
2. ROUNDING STATEMENT [severity: Review]: Only applies when the manuscript contains
   unweighted sample sizes derived from restricted-use data. If no such numbers are present,
   do NOT flag this rule. The manuscript must include a statement conveying that sample sizes
   are unweighted and rounded to the nearest 10 (or nearest 50 for ECLS-B). The exact wording
   does not need to match precisely — any paraphrase that clearly communicates both concepts
   (unweighted AND rounded to nearest 10) is acceptable and must NOT be flagged. Explicitly
   acceptable phrasings include (but are not limited to): "rounded to nearest tens," "rounded
   to the nearest 10," "rounded to the nearest ten," "rounded to the nearest 10s," "rounded
   to the nearest tens," "unweighted sample sizes are rounded to the nearest 10," and any
   similar wording conveying the same meaning. Do NOT flag any of these as non-standard.
   A single global statement anywhere in the manuscript (e.g., in the methods section, a
   footnote, or a table note) is sufficient to cover all sample sizes throughout — authors
   are NOT required to repeat "unweighted" next to every individual N in the text. Search
   the entire manuscript; do not flag if the statement appears anywhere.
   CRITICAL: Rule 2 must NOT be triggered solely because a percentage appears alongside a
   rounded count. If the reported count already ends in 0 (i.e., it is compliant under
   Rule 1), then a co-located percentage does NOT make the rounding statement inconsistent.
   Only flag Rule 2 if the rounding statement is genuinely absent or contradicted by an
   actually unrounded count (last digit 1–9).
3. PERCENTAGES AND ROUNDING [severity: Low]: This rule covers ONLY the number of decimal
   places used in a percentage or proportion. It has nothing to do with back-calculation.
   Applies to values explicitly labeled as percentages (followed by "%" or described as
   "percent") or proportions (described as a proportion or rate). Do NOT apply to means,
   scores, test statistics, regression coefficients, standard errors, or any other estimate.
   Precision requirements: summary percentages → max 1 decimal place; reference percentages
   → max 2 decimal places; proportions → rounded to hundredths (e.g., 0.08).
   BACK-CALCULATION IS NOT PART OF THIS RULE. Do NOT flag any percentage under Rule 3 for
   back-calculation risk. Back-calculation is not a concern under IES rules unless a
   percentage or proportion could arithmetically reveal a small-cell count of 1–9 from an
   exact (unrounded) base. When the base count ends in 0 (rounded), dividing it by any
   percentage yields only an approximation — not the exact pre-rounding value — so
   back-calculation is impossible. Never flag a percentage as a back-calculation risk when
   the base count is rounded. In practice, IES back-calculation concerns arise only for
   percentages of very small subgroups where the base is exact and small (e.g., a group of
   exactly 12 where 8.3% would reveal exactly 1 person). They do not arise for large
   rounded base counts like 4130.
4. WEIGHTED/UNWEIGHTED LABELING [severity: Review]: Only applies to TABLES that report
   sample size values (raw counts or Ns) derived from restricted-use data. Each such table
   must include a note or label that explicitly uses the word "unweighted" (or "weighted")
   in connection with the sample size counts reported in that table.
   CRITICAL: A rounding statement alone does NOT satisfy this requirement. The word
   "unweighted" (or "weighted") must appear explicitly in the table note or label.
   FAILING example — flag this under Rule 4:
     "Numbers for |control| and |treated| are rounded to nearest tens."
     → Missing "unweighted": does not disclose whether Ns are weighted or unweighted.
   PASSING example — do NOT flag:
     "Numbers for |control| and |treated| are unweighted and rounded to nearest tens."
     → Explicitly states "unweighted": satisfies Rule 4.
   MANDATORY VERIFICATION: For every restricted-use table that reports Ns, read the exact
   text of its NOTE or footnote word by word. Physically check whether the word "unweighted"
   or "weighted" appears. A rounding statement alone ("rounded to nearest tens") is NOT
   sufficient — if "unweighted" (or "weighted") is absent from the note, flag the table
   under Rule 4. Do not assume compliance without reading the note text.
   If a global statement in the manuscript says both "unweighted" AND "rounded to nearest
   10", in-text sample sizes are covered — do NOT flag individual in-text Ns. This rule
   applies ONLY to sample size counts (Ns) in tables.
   The following must NEVER be flagged under Rule 4: regression coefficients, R-squared
   values, standard errors of regression estimates, F-statistics, t-statistics, p-values,
   means, percentages, or any statistical result reported in text. Regression results
   reported in text sections do not require a weighted/unweighted disclosure.
5. SOURCE NOTES [severity: Review]: Only applies to tables and figures that present point
   estimates or raw numbers derived from restricted-use datasets (e.g., tables with cell
   values, figures with data points, bar charts with frequencies or percentages). Read the
   figure caption, title, and surrounding text to determine whether it contains such values.
   EXEMPT FIGURES — omit entirely from findings, no exception:
   Any figure described as or visually identifiable as a causal diagram, DAG, directed
   acyclic graph, path diagram, theoretical framework, conceptual model, flow chart, or
   any figure that contains no numeric data values is FULLY EXEMPT from Rule 5. This
   exemption is unconditional — it applies even if the figure caption or surrounding text
   mentions a restricted-use dataset by name, even if the figure was motivated by or
   illustrates a concept from that dataset, and even if the figure appears in a paper that
   uses restricted-use data. The sole criterion is whether the figure itself displays numeric
   data values. If it does not, it is exempt. Do NOT flag it, do NOT defer to a human
   reviewer, do NOT add a conditional note. Omit it from the findings entirely.
   Do NOT flag figures or tables that do not use restricted-use data. When applicable, the
   SOURCE note format is:
   "SOURCE: U.S. Department of Education, National Center for Education Statistics, [Survey Name]"
   AGENCY NAME CHECK — MANDATORY FOR EVERY SOURCE NOTE [severity: Review]: For every
   SOURCE note you find, copy its exact text and compare it character by character against
   the only correct form:
   "National Center for Education Statistics"
   Any deviation — even a single wrong word — is a violation that must be flagged as Review:
     - "National Center on Education Statistics" → wrong preposition ("on" instead of "for")
     - "National Center on Educational Statistics" → wrong preposition AND wrong adjective
     - "National Center for Educational Statistics" → wrong adjective ("Educational" instead of "Education")
     - Any other deviation from the exact phrase above
   Do not read SOURCE notes casually — read every word. "Educational" and "Education" are
   different words; "on" and "for" are different prepositions. If the SOURCE note does not
   say exactly "National Center for Education Statistics", flag it as severity Review. Flag
   each table or figure with an incorrect agency name as a separate finding.
   NCES AFFILIATION LANGUAGE [severity: Review]: NCES is currently part of IES — it has
   not left or been separated from IES. Scan the manuscript body text for any phrase that
   describes NCES as "previously within", "formerly within", "formerly part of", or any
   other phrasing that implies NCES is no longer part of IES. Such language is factually
   incorrect and must be flagged. The correct phrasing is "within" or "in" (e.g., "NCES,
   within the Institute of Education Sciences" or "NCES, in IES"). Flag any use of
   "previously within the Institute of Education Sciences" or equivalent as severity Review,
   with a recommendation to replace "previously within" with "in".
6. INTERNAL CONSISTENCY [severity: Review for exact mismatches; severity: Low for
   approximate or derived values]: Two sub-cases apply:
   (a) EXACT MISMATCH [severity: Review]: The text explicitly cites a specific table or
   figure (e.g., "as shown in Table 1") and restates a specific value from it, but the
   value in the text does not match the value in the table. All three conditions must be
   confirmed: the text names the source table/figure, the value appears in that table/figure,
   and the two values disagree. Do NOT flag potential ambiguity or uncertainty — only flag
   a confirmed, clear disagreement between two specific values.
   SUBGROUP SUM CHECK: Before flagging that subgroup counts do not sum to a reported total,
   compute the actual arithmetic sum of all subgroup values and compare it to the stated
   total. Account for rounding (subgroup sums may differ from the total by a small amount
   due to rounding). Only flag if the computed sum is clearly inconsistent with the stated
   total after accounting for rounding. If the sum matches or is plausibly consistent, do
   NOT flag this as an inconsistency.
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
    truncated = manuscript_text[:120000]
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
- SAME SOURCE, SAME FINDING: if two or more violations trace back to the same specific
  value, expression, or statistic in the manuscript (e.g., both concerns stem from the
  same F-statistic, the same table cell, or the same sentence), consolidate them into one
  finding. Do not create two findings that point to the same source.
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
- NO-ACTION GATE: If you reason through a potential issue and conclude it is not a
  violation, do NOT include it in the findings array at all — not even with "N/A" fields
  or a note saying "discarding this finding." A finding that you discard must be completely
  absent from the JSON output. Every object in the findings array must represent a genuine
  violation requiring author action.
- INTERNAL CHECKS ONLY: Before including any finding, verify ALL of the following:
  (a) DATA SOURCE: confirm the table, figure, or passage uses IES/NCES restricted-use data.
  If the data source is a public dataset, state data, or any non-IES/NCES source, discard
  the finding entirely — it is exempt from all rules.
  (b) RESPONDENT COUNT vs. ANALYTICAL QUANTITY: for any number flagged under Rule 1,
  confirm it is a count of individual respondents from the restricted-use dataset, NOT a
  methodological quantity such as number of strata (J, J_k), number of clusters, number of
  matched sets, or any other structural quantity. If it is analytical/methodological,
  discard the finding.
  (c) LAST DIGIT: for any respondent count flagged under Rule 1, confirm its last digit is
  1–9. If the last digit is 0 the count is compliant — discard the finding.
  (d) Rule 2 must not be triggered by a compliant (last-digit-0) count alongside a
  percentage — only flag for a genuinely absent or contradicted rounding statement.
  (e) Rule 3 is about decimal precision only — never flag a percentage for back-calculation.
  (f) A small cell is only N=1–9 — N≥10 is never a small cell.
  (g) CV must be (SE/Estimate)×100 and must exceed 30% to flag.
  Omit any finding that fails any of these checks.

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
      "recommendation": "Round the N to the nearest 10",
      "rule1_last_digit": "7"
    }}
  ]
}}

RULE 1 VERIFICATION FIELD: Every finding with severity "High" that cites Rule 1
(unrounded sample size) MUST include a "rule1_last_digit" field containing ONLY the
last digit (0–9) of the exact integer count being flagged. Compute this digit carefully:
look at the integer, ignore any commas or formatting, and write only its final digit.
  Examples: 43 → "3", 120 → "0", 510 → "0", 2113 → "3", 80 → "0", 57 → "7"
If the last digit is "0", you must NOT include the finding — a count ending in 0 is
compliant. Remove the finding before writing the JSON.
For non-Rule-1 findings (Review, Low, or any finding not about unrounded counts),
omit the "rule1_last_digit" field entirely.

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


# ── Source-note detection ─────────────────────────────────────────────────────

def _find_tables_with_source_notes(text: str) -> set:
    """
    Return the set of table numbers (as strings, e.g. {'5'}) whose table caption
    is followed by a SOURCE: line within the next 5 000 characters of extracted text.
    Only tables confirmed to have a SOURCE note can be treated as IES restricted-use.
    """
    tables_with_source: set = set()
    # Match table captions: "Table 4:" / "Table 4." / "Table 4\n"
    for m in re.finditer(r'\bTable\s+(\d+)\s*[:\.\n]', text, re.IGNORECASE):
        table_num = m.group(1)
        window = text[m.start(): min(len(text), m.start() + 5000)]
        if re.search(r'\bSOURCE\s*:', window, re.IGNORECASE):
            tables_with_source.add(table_num)
    return tables_with_source


def _location_table_num(location: str):
    """Extract the table number string from a finding's location field, or None."""
    m = re.search(r'\bTable\s+(\d+)\b', location, re.IGNORECASE)
    return m.group(1) if m else None


def _find_non_ies_section_tables(text: str) -> set:
    """
    Return the set of table numbers (as strings) that appear inside sections whose
    text explicitly cites a published paper (Author et al., YYYY) as the primary data
    source without naming any IES/NCES restricted-use dataset.

    Example: a section that says "We use New Jersey's data from Wong et al. (2007)"
    but never mentions NAEP, ECLS, NCES, etc. — all Table references in that section
    are returned so they can be filtered from the audit output.
    """
    non_ies_tables: set = set()

    # Section header pattern: optional "Section" keyword, then a number like "7", "7.1", "7.1.2"
    # followed by whitespace and the beginning of a title word.
    section_header_re = re.compile(
        r'(?m)^[ \t]*(?:section\s+)?(\d+(?:\.\d+)*)[ \t]*[.\-–]?[ \t]+\S',
        re.IGNORECASE,
    )

    # Citation pattern: "et al. (YYYY)" or "et al., YYYY" or "et al. YYYY"
    citation_re = re.compile(r'\bet\s+al\.[\s,]+\(?\d{4}\)?', re.IGNORECASE)

    # IES / NCES restricted-use dataset indicators
    ies_dataset_re = re.compile(
        r'\b(?:naep|ecls|els|hsls|sses|nscg|nces|restricted[- ]use)\b'
        r'|national\s+assessment\s+of\s+educational\s+progress'
        r'|early\s+childhood\s+longitudinal'
        r'|education\s+longitudinal\s+study'
        r'|high\s+school\s+longitudinal'
        r'|institute\s+of\s+education\s+sciences',
        re.IGNORECASE,
    )

    # Table reference pattern
    table_ref_re = re.compile(r'\bTable\s+(\d+)\b', re.IGNORECASE)

    # Collect all section header positions
    headers = [(m.start(), m.group()) for m in section_header_re.finditer(text)]

    for i, (start, _) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        section_text = text[start:end]

        has_citation = bool(citation_re.search(section_text))
        has_ies = bool(ies_dataset_re.search(section_text))

        if has_citation and not has_ies:
            # Section uses data from a published (non-IES) source
            for m in table_ref_re.finditer(section_text):
                non_ies_tables.add(m.group(1))

    return non_ies_tables


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

    # Identify which tables have SOURCE notes before discarding the text.
    # A table confirmed to have a SOURCE note is treated as IES restricted-use data;
    # tables without SOURCE notes cannot be confirmed and are filtered out in post-processing.
    tables_with_source = _find_tables_with_source_notes(manuscript_text)
    non_ies_section_tables = _find_non_ies_section_tables(manuscript_text)

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
    # Fallback: if the model prepended reasoning text, find the first '{' and last '}'
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as e:
        import logging
        logging.error("JSON parse error: %s\nRaw response (first 500 chars): %s", e, raw[:500])
        raise HTTPException(
            status_code=502,
            detail=f"Received an unexpected response from the AI. Parse error: {e}. Response preview: {raw[:200]}"
        )

    # Remove Rule 1 High findings where the model's own last-digit check shows the count ends in 0
    def _rule1_last_digit_compliant(finding: dict) -> bool:
        """Returns True if the finding should be dropped because the flagged count ends in 0."""
        if finding.get("severity") != "High":
            return False
        last_digit = str(finding.get("rule1_last_digit", "")).strip()
        return last_digit == "0"

    # Remove findings that reveal the data source is non-IES/NCES:
    # (a) the model cites a published paper as the data source, or
    # (b) the recommendation itself expresses uncertainty about whether data is IES/NCES
    _CITATION_PATTERN = re.compile(r'\bet\s+al\..*?\d{4}|\(\s*\d{4}\s*\)', re.IGNORECASE)
    _NON_IES_SCOPE_PHRASES = (
        "if the data are from a non",
        "if the data are from an ies",
        "if this is from a non",
        "if this uses restricted",
        "if the pre-k data",
        "clarify the data source",
        "non-nces source",
        "non-ies source",
    )

    def _is_non_ies_data(finding: dict) -> bool:
        combined = " ".join([
            finding.get("flag", ""),
            finding.get("rule", ""),
            finding.get("recommendation", ""),
        ]).lower()
        # Drop if the recommendation hedges about whether data is IES/NCES
        if any(phrase in combined for phrase in _NON_IES_SCOPE_PHRASES):
            return True
        # Drop if the flag or recommendation cites a published paper (et al. + year)
        if _CITATION_PATTERN.search(combined):
            return True
        return False

    findings_raw = report.get("findings", [])
    findings_raw = [f for f in findings_raw if not _rule1_last_digit_compliant(f)]
    findings_raw = [f for f in findings_raw if not _is_non_ies_data(f)]
    report["findings"] = findings_raw

    # Remove findings that indicate no action is needed, checking all text fields
    NO_ACTION_PHRASES = (
        "no action needed", "no additional action", "no further action",
        "no changes needed", "no changes required", "no correction needed",
        "already compliant", "no issue", "discarding this finding",
        "not a rule", "not a violation",
    )
    def _has_no_action(finding: dict) -> bool:
        combined = " ".join([
            finding.get("recommendation", ""),
            finding.get("flag", ""),
            finding.get("rule", ""),
        ]).lower()
        return (
            any(phrase in combined for phrase in NO_ACTION_PHRASES)
            or finding.get("recommendation", "").strip().upper() == "N/A"
            or finding.get("rule", "").strip().upper() == "N/A"
        )
    findings = [f for f in findings_raw if not _has_no_action(f)]

    # SOURCE-NOTE-BASED SCOPE FILTER
    # Tables WITH a SOURCE note: confirmed IES restricted-use — keep all findings.
    # Tables WITHOUT a SOURCE note: data source unconfirmed — apply selective filtering:
    #   - Remove Rules 1, 2, 4 findings (cannot confirm IES data).
    #   - Keep Rule 5 "missing SOURCE note" findings ONLY if the model's own flag field
    #     explicitly names a known IES/NCES dataset (NAEP, ECLS, etc.), signalling that
    #     the model correctly identified restricted-use data that just lacks the note.
    #   - This allows detection of genuinely missing SOURCE notes while blocking false
    #     positives for non-IES tables like those from published third-party sources.

    _IES_DATASET_TOKENS = (
        'naep', 'ecls', 'els:', 'hsls', 'sses', 'nscg',
        'national assessment', 'early childhood longitudinal',
        'high school longitudinal', 'education longitudinal',
    )
    _MISSING_SOURCE_PHRASES = (
        'missing a source', 'no source note', 'lacks a source',
        'source note is missing', 'add a source note',
        'does not have a source', 'source note identifying',
        'include a source note', 'missing source note',
    )

    def _flag_names_ies_dataset(finding: dict) -> bool:
        """True if the FLAG field explicitly names a known IES/NCES dataset."""
        flag = finding.get("flag", "").lower()
        return any(tok in flag for tok in _IES_DATASET_TOKENS)

    def _is_missing_source_finding(finding: dict) -> bool:
        """True if this finding is specifically about a missing SOURCE note."""
        combined = (finding.get("flag", "") + " " + finding.get("recommendation", "")).lower()
        return any(phrase in combined for phrase in _MISSING_SOURCE_PHRASES)

    def _unconfirmed_table(finding: dict) -> bool:
        """True if this finding should be removed due to unconfirmed IES data source."""
        tnum = _location_table_num(finding.get("location", ""))
        if tnum is None:
            return False  # Not a table-specific finding — leave it
        # SOURCE note is the strongest signal — a confirmed IES table is NEVER removed,
        # even if its section also cites a published paper in passing.
        if tnum in tables_with_source:
            return False  # SOURCE note present — confirmed IES data, keep everything
        # No SOURCE note. If the section explicitly declares a non-IES data source, remove.
        if tnum in non_ies_section_tables:
            return True
        # No SOURCE note and no section signal. Keep only legitimate missing-SOURCE findings.
        if _is_missing_source_finding(finding) and _flag_names_ies_dataset(finding):
            return False  # Legitimate: restricted-use table missing its SOURCE note
        return True  # Remove: cannot confirm this table uses IES restricted-use data

    findings = [f for f in findings if not _unconfirmed_table(f)]
    report["findings"] = findings
    report["item_count"] = len(findings)

    # Recompute status based on filtered findings
    if not findings:
        report["status"] = "PASS"
    elif any(f.get("severity") == "High" for f in findings):
        report["status"] = "HIGH RISK"
    else:
        report["status"] = "REVIEW NEEDED"

    elapsed = time.monotonic() - start_time
    log_metadata(
        framework=framework,
        tokens_in=message.usage.input_tokens,
        tokens_out=message.usage.output_tokens,
        processing_time=elapsed,
        status=report.get("status", "UNKNOWN"),
    )

    return report
