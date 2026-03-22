# Architecture

## System Overview

Two-mode application: a compliance checker (the main interface) and a quality assurance pipeline.

```
┌─────────────────────────────────────────────────┐
│                  src/ (Checker)                 │
│                                                 │
│  main.py ──> prompts.py ──> client.py ──> Azure │
│    REPL        ReAct          retry             │
│              template        backoff            │
│                                                 │
│  models.py ── Pydantic schemas (shared) ────────┤
│                                                 │
│                   eval/ (QA Pipeline)           │
│                                                 │
│  run.py ──> baseline.py ──> client.py ──> Azure │
│    │         zero-shot                          │
│    │                                            │
│    ├──> scorer.py  (exact-match F1, no LLM)     │
│    │                                            │
│    └──> metrics.py                              │
│          mean F1, report                        │
│                                                 │
│  data.py ── JSON loading, stratified sampling   │
└─────────────────────────────────────────────────┘
```

## Design Decisions

### DD-1: Checker vs QA Separation

**Context**: The compliance checker and the quality pipeline serve different purposes and have different change rates.

**Decision**: Separate `src/` (compliance checker) from `eval/` (quality pipeline). Shared code lives in `src/models.py`, `src/client.py`, and `src/prompts.py`.

**Rationale**: The checker is the operational interface — it changes when rules or prompts change. The quality pipeline is test infrastructure — it changes when measurement methodology changes. Coupling them conflates operational logic with quality assurance concerns.

**Tradeoff**: Slight duplication in how ReAct analysis is invoked. Accepted because the invocation is one line.

---

### DD-2: Exact-Match F1 Scorer (No LLM Judge)

**Context**: Evaluation scoring can use an LLM-as-judge or a deterministic scorer. LLM judges introduce variance that can mask real signal, especially when the delta between systems is small.

**Decision**: Use exact-match F1 on row IDs: `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = harmonic mean`. Gold labels are computed from Python logic that exactly mirrors the natural language rules.

**Rationale**: Violation detection has deterministic ground truth — a row either violates a rule or it doesn't. Python logic generates gold labels, eliminating label uncertainty. F1 captures both false positives (over-flagging) and false negatives (missed violations), which matters for compliance tasks where both error types have operational cost.

**Tradeoff**: Requires careful dataset design to ensure Python logic matches natural language exactly. Synthetic data may not capture all real compliance noise — the benchmark measures structured reasoning capability on representative rule patterns, not coverage of a specific domain.

**ASRs**: ASR-04 (quality pipeline budget), ASR-07 (scoring determinism)

---

### DD-3: 5-Step ReAct Chain with Explicit VERIFY

**Context**: Constraint satisfaction requires different reasoning operations than open-ended analysis — each step must have a defined scope and produce verifiable intermediate output.

**Decision**: PARSE_RULES → MAP_COLUMNS → EVALUATE → FLAG → VERIFY. The VERIFY step explicitly re-examines: (1) OR branch coverage, (2) negation correctness, and (3) boundary conditions (e.g., `≤` vs `<`).

**Rationale**: The most common baseline errors are: (a) checking only the first condition of an AND rule, (b) misreading OR as AND, and (c) being sloppy about boundary values. VERIFY forces the model to reconsider each of these after the initial evaluation pass.

**Tradeoff**: 5 steps is more verbose than necessary for easy violations. Accepted because the VERIFY step costs one extra reasoning step but catches systematic errors on hard violations.

**ASRs**: ASR-06 (checker accuracy), ASR-05 (token budget)

---

### DD-4: Structured Output via Pydantic Parse

**Context**: LLM responses need to be parsed into typed structures.

**Decision**: Use `client.beta.chat.completions.parse()` with Pydantic models as `response_format`.

**Rationale**: Guaranteed valid JSON matching the schema. Zero parsing failures. Type safety flows from Pydantic through the entire codebase.

**Tradeoff**: Locked to OpenAI-compatible API. Accepted since Azure OpenAI is the only provider.

**ASRs**: ASR-01 (analysis response time)

---

### DD-5: Planted Violation Dataset with Python Ground Truth

**Context**: A labeled evaluation dataset is needed where gold labels are accurate.

**Decision**: Generate 50 synthetic datasets (5 categories × 10) via `scripts/generate_data.py`. Each dataset: 15 rows (5 planted violations + 4 near-miss rows + 6 explicit compliant decoys), 4 rules, gold violations computed by the same Python functions used to generate the data.

**Rationale**: Gold labels computed from Python logic cannot drift from the natural language rules — the same functions produce both. Explicit compliant decoys prevent accidental violations and ensure baseline cannot score well by over-flagging all rows.

**Tradeoff**: Synthetic data may not capture real compliance noise. The evaluation measures structured reasoning capability on representative rule patterns.

**ASRs**: ASR-06 (checker accuracy), ASR-07 (scoring determinism)

---

### DD-6: Prompt Templates as Named Constants

**Context**: Prompts can be stored in files, template engines, or code constants.

**Decision**: All prompts are Python string constants in `src/prompts.py` with `.format()` interpolation.

**Rationale**: No framework dependency. Prompts are version-controlled, diffable, and reviewable.

**Tradeoff**: No dynamic prompt composition. Sufficient for this scope.

**ASRs**: ASR-09 (prompt isolation)
