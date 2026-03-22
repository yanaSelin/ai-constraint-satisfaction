# ReAct Multi-hop Constraint Satisfaction

A compliance checking tool that detects business rule violations in tabular data using ReAct (Reason + Act) prompt engineering. The model works through a 5-step structured reasoning chain — parsing rules into logical expressions, mapping columns, evaluating each row against each rule, flagging violations, and verifying edge cases — before committing to a final violation list.

Structured reasoning eliminates the most common failure modes of naive LLM-based checkers: skipping conditions in compound AND rules, misreading OR as AND, and failing multi-hop rules where one rule defines a term used in another.

## Architecture

The system has two modes:

- **Compliance checker** (`src/main.py`) — the main interface. Paste business rules and a CSV table, receive a step-by-step ReAct analysis with all detected violations.
- **Quality pipeline** (`eval/run.py`) — continuous quality assurance. Runs the checker against a labeled dataset, measures exact-match F1 against known violations, and produces a versioned results archive.

Key design decisions:
1. **Checker/QA separation** — the checker and quality pipeline share models and client code but have independent entry points; product logic is never coupled to test infrastructure
2. **5-step ReAct chain** — PARSE_RULES → MAP_COLUMNS → EVALUATE → FLAG → VERIFY; each step has a defined scope and forces exhaustive per-row, per-rule checking
3. **Exact-match F1 scoring** — violation answers are deterministic row IDs; no LLM judge needed, scoring is noise-free and reproducible
4. **Python ground truth** — gold labels are computed from Python logic that exactly mirrors the natural language rules, eliminating label uncertainty

See [docs/architecture.md](docs/architecture.md) for full decision rationale and [docs/asr.md](docs/asr.md) for requirements.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

## Usage

### Compliance Checker

```bash
python -m src.main

# Enable debug logging (writes to checker.log)
LOG_LEVEL=DEBUG python -m src.main
```

Enter rules (one per line), then the CSV data:

```
Enter business rules:
> Rule-A: Orders from unverified customers with order_value > 500 must be flagged.
> Rule-B: Express shipping is prohibited when has_fragile=true AND is_international=true.
>

Enter data table (CSV):
> order_id,customer_verified,order_value,shipping_type,has_fragile,is_international
> ORD-001,true,320.00,standard,false,false
> ORD-002,false,650.00,standard,false,false
> ORD-003,true,800.00,express,true,true
>

[Thought 1] Parsing each rule into a precise logical expression.
[Action: PARSE_RULES] Rule-A: customer_verified=false AND order_value > 500 → manual_review=true...
...

Violations found: 2
  ORD-002  [Rule-A]  — customer_verified=false, order_value=650.00 > 500
  ORD-003  [Rule-B]  — shipping_type=express, has_fragile=true, is_international=true
```

### Quality Pipeline

```bash
# Quick run (5 samples, ~2 min)
EVAL_SAMPLE_SIZE=5 python -m eval.run

# Full run (50 samples, 10 per category, ~18 min)
python -m eval.run
```

The pipeline:
1. Loads stratified samples from the labeled dataset
2. Runs zero-shot baseline and ReAct checker on each sample
3. Scores both against gold violation row IDs using exact-match F1
4. Prints F1 comparison by category
5. Archives results to `results/vNNN.json`

### Regenerate Labeled Dataset

```bash
python scripts/generate_data.py
```

## Quality Benchmark

**Labeled dataset**: 50 synthetic compliance tables across 5 categories (10 per category): `order_compliance`, `hr_policy`, `inventory_rules`, `loan_approval`, `access_control`. Each table has 15 rows (5 true violations + 4 near-miss rows + 6 compliant decoys) and 4 business rules at three difficulty levels:
- **Easy**: single boolean condition (`customer_verified=false AND order_value > 500`)
- **Medium**: compound AND with near-miss rows (satisfy N-1 of N conditions — not violations, but naive checkers flag them as false positives)
- **Hard**: multi-hop (one rule defines a term used in another rule's condition) or OR branches (naively misread as AND)

**Metric**: Mean F1 across 50 samples (exact match of predicted vs gold violation row IDs). Deterministic — no LLM judge.

**Results** (50 samples, 5 categories × 10):

| Metric | Zero-shot baseline | ReAct |
|--------|--------------------|-------|
| Mean F1 | 0.938 | **1.000** |
| Mean Precision | 0.890 | **1.000** |
| Mean Recall | 1.000 | 1.000 |
| Exact Match Rate | 48% | **100%** |

**Per-category breakdown**:

| Category | Baseline F1 | ReAct F1 | Delta |
|----------|-------------|----------|-------|
| access_control | 0.902 | 1.000 | +0.098 |
| hr_policy | 0.991 | 1.000 | +0.009 |
| inventory_rules | 0.929 | 1.000 | +0.071 |
| loan_approval | 0.867 | 1.000 | +0.133 |
| order_compliance | 1.000 | 1.000 | +0.000 |

ReAct achieves perfect precision on all 50 samples. The zero-shot baseline produces false positives on near-miss rows in 3 of 5 categories — flagging rows that satisfy N-1 of N AND conditions. The operationally meaningful measure is exact match rate: the baseline delivered the exact correct violation list on only 24 of 50 samples (48%); ReAct on all 50 (100%).

## Project Structure

```
src/
├── main.py        # Compliance checker (entry point)
├── prompts.py     # ReAct and baseline prompt templates
├── models.py      # Pydantic schemas for structured output
└── client.py      # AzureOpenAI client with retry/backoff

eval/
├── run.py         # Quality pipeline orchestrator (entry point)
├── baseline.py    # Zero-shot baseline
├── scorer.py      # Exact-match F1 scorer (no LLM judge)
├── data.py        # Dataset loading, stratified sampling
└── metrics.py     # F1 aggregation, report, versioned archive

data/
└── constraints_golden.json  # 50 labeled tables (5 categories × 10)

scripts/
└── generate_data.py         # Labeled dataset generator with planted violations

docs/
├── asr.md            # Architecturally Significant Requirements
└── architecture.md   # Design decisions with tradeoffs
```
