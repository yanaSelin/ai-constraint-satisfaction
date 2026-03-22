"""Prompt templates for ReAct multi-hop constraint satisfaction evaluation.

All prompt templates are stored as named constants — no framework dependencies.
"""

REACT_SYSTEM = """\
You are a compliance analyst. Given a set of business rules and a data table, your task is to
identify ALL rows that violate any rule.

Use a structured ReAct (Reason + Act) approach. Perform exactly 5 steps in order:

1. **PARSE_RULES** — For each rule, extract: the column names involved, the logical operator
   (AND / OR / NOT), threshold values, and the required flag or status. Restate each rule as
   a precise logical expression. Identify which rules reference outputs of other rules (multi-hop).
2. **MAP_COLUMNS** — List the columns in the data. Confirm which columns each rule references.
   Note any boundary conditions (e.g., strictly greater than vs. greater-than-or-equal).
3. **EVALUATE** — For each row, check each rule systematically:
   - For AND rules: ALL sub-conditions must be true for a violation to occur.
   - For OR rules: ANY sub-condition being true is enough.
   - For multi-hop rules: first resolve the defined term (e.g., "qualified applicant"),
     then apply the main condition to that resolved set.
   - Do NOT skip rows. Do NOT skip conditions within a rule.
4. **FLAG** — List every row that violates at least one rule. For each: row ID, rule name(s),
   and the specific column values that triggered the violation.
5. **VERIFY** — Re-examine your flag list:
   - Did you check every rule against every row?
   - For OR conditions: did you check BOTH branches independently?
   - For negation: are you reading NOT correctly?
   - For boundary values: is "exactly at the threshold" a violation?
   Correct any errors before finalising.

Rules:
- Treat ">" as strictly greater than; ">=" as greater-than-or-equal.
- A row that satisfies all conditions of an AND rule is flagged. If ANY condition is false, it is NOT flagged.
- Multi-hop: resolve the embedded definition first, then apply the outer rule.
- Do NOT invent column values — use only the data provided.\
"""

REACT_USER = """\
Business rules:
{rules_text}

Data table (CSV):
{data_csv}

Identify all violations using the 5-step ReAct process.\
"""

BASELINE_SYSTEM = """\
You are a compliance analyst. Given the following business rules and a data table,
identify all rows that violate any rule.
For each violation, provide: the row ID, the rule(s) violated, and a brief reason.\
"""

BASELINE_USER = """\
Business rules:
{rules_text}

Data table (CSV):
{data_csv}

List all violations.\
"""


def format_react_messages(rules_text: str, data_csv: str) -> list[dict]:
    """Build message list for a ReAct constraint evaluation call."""
    return [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": REACT_USER.format(rules_text=rules_text, data_csv=data_csv)},
    ]


def format_baseline_messages(rules_text: str, data_csv: str) -> list[dict]:
    """Build message list for a baseline constraint evaluation call."""
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": BASELINE_USER.format(rules_text=rules_text, data_csv=data_csv)},
    ]


def format_steps_for_display(steps: list) -> str:
    """Format ReAct steps for human-readable console output.

    Args:
        steps: List of ReActStep objects.

    Returns:
        Formatted string with the reasoning chain.
    """
    blocks = []
    for i, step in enumerate(steps, 1):
        block = f"[{i}/{len(steps)}] {step.action}\n"
        block += f"  Thought:     {step.thought}\n"
        block += f"  Observation: {step.observation}"
        blocks.append(block)
    return "\n\n".join(blocks)
