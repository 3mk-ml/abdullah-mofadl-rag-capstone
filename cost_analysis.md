# Cost Analysis — Baseline

**Pricing check date:** 2026-09-28. Re-check provider pricing before final submission.

## Cost architecture
- Ingestion embeddings: local `intfloat/multilingual-e5-small` -> API cost **$0**
- Query embeddings: local -> **$0**
- BM25 + RRF: local -> **$0**
- Multilingual reranker: local -> **$0**
- Answer generation: OpenAI API -> token-based variable cost
- Hosting: Hugging Face CPU Basic hardware may be free depending on the current account/Space plan; verify the actual deployment tier before submission.

## Initial model assumption
Baseline generator: `gpt-5.6-luna`.

Before final submission, verify current API pricing from the official provider pricing page and update the rates below. Do not treat these planning numbers as final measured costs.

Assumed average query after retrieval:
- 4,500 input tokens
- 450 output tokens

## Required scenarios
The final report must show cost at:
- 1,000 users
- 10,000 users
- 100,000 users

Because LLM billing is query-driven, explicitly state an activity assumption such as **10 questions per active user per month**, then compute queries/month and model cost from the verified current token rates.

## What must be replaced before submission
Run the final app, measure actual average input/output tokens, then regenerate this analysis with:

```bash
python scripts/cost_analysis.py --input-tokens ACTUAL_INPUT --output-tokens ACTUAL_OUTPUT --input-price CURRENT_INPUT_RATE --output-price CURRENT_OUTPUT_RATE --queries-per-user 10
```

Also document any paid Hugging Face plan or upgraded hardware actually used. Do not present the baseline as a measured final cost.
