# Architecturally Significant Requirements

## ASR Catalog

| ID | Category | Requirement | Threshold | Priority | Architectural Impact |
|----|----------|-------------|-----------|----------|---------------------|
| ASR-01 | Performance | Analysis response time | < 30s per dataset | High | Single API call with structured output; no multi-turn chains |
| ASR-02 | Performance | Eval pipeline runtime | < 20 min for 50 samples | Medium | Sequential processing with incremental save; resume support |
| ASR-03 | Reliability | Retry on rate limits | Exponential backoff, max 3 retries | High | Centralized retry logic in `client.py`; all callers inherit |
| ASR-04 | Cost | Quality pipeline budget | 2 API calls per dataset in quality pipeline (baseline + ReAct) — no judge call | High | Exact-match F1 scorer replaces LLM judge; deterministic scoring |
| ASR-05 | Cost | Token budget per analysis | < 3000 tokens per ReAct call | Medium | 5-step structure with brevity guidance in prompt |
| ASR-06 | Quality | Checker accuracy | F1 = 1.000 on quality benchmark (50 datasets, 5 categories) | High | 5-step chain with explicit EVALUATE and VERIFY steps |
| ASR-07 | Quality | Scoring determinism | Exact-match F1, no judge variance | High | Python scorer with deterministic row ID matching |
| ASR-08 | Security | API key handling | Never logged, fail-fast if missing | High | `os.environ["KEY"]` (not `.get()`); no key in logs or git |
| ASR-09 | Maintainability | Prompt isolation | All templates in `prompts.py` | Medium | Single file to version, review, and iterate on prompts |
| ASR-10 | Observability | Request-level logging | Each check logs: rules count, rows evaluated, violations found | Medium | `logging` module throughout; structured output logged at INFO |

