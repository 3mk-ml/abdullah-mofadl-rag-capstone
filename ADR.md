# ADR-001 — AgriRAG Final Architecture

**Status:** Accepted  
**Decision date:** 2026-10-01

## Context
The Track B Week 3 capstone requires a public RAG system over 20–50 quality documents, justified ingestion choices, hybrid retrieval plus reranking, a 30-question Recall@5 evaluation, a 20-question RAGAS report, a bilingual interface with authentication, deployment, a one-page ADR, and cost analysis. AgriRAG also has to remain practical on student-scale infrastructure and support Arabic questions over primarily English agricultural references.

## Decision
AgriRAG uses 20 successfully collected authoritative agriculture documents from a 24-source curated manifest, focused on FAO/FAO-partner, ICARDA/CGIAR, and World Bank material. PyMuPDF extracts documents. Text is split with recursive structure-aware chunks targeting about 350 tokens with 60-token overlap.

Embeddings use local `intfloat/multilingual-e5-small` vectors stored in persistent Chroma. Retrieval combines vector top-20 and BM25 top-20 results with Reciprocal Rank Fusion (RRF, k=60). The fused set is reranked locally with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, and the best five passages are supplied to the generator.

The final generation provider is Cohere using `command-r7b-12-2024`. The system prompt requires answers to use only retrieved evidence, preserve units, expose insufficient evidence, and cite source passages. Streamlit provides English/Arabic UI, RTL rendering, Amiri typography, source excerpts, latency display, and a shared-password gate. Docker is the deployment boundary; Railway is the primary operational target, with Hugging Face Spaces remaining compatible.

## Evidence for the decision
The final corpus produced **1,828 extracted pages/units and 5,540 chunks**. The manually verified 30-question retrieval test achieved **Recall@5 = 90.00% (27/30)**, exceeding the required 80% threshold without relabelling the three misses (Q03, Q12, Q24).

The successful 20-question RAGAS run (#21) used Cohere Command R7B consistently for generation and judging and produced:

- Faithfulness: **0.8329**
- Answer relevancy: **0.9819**
- Context precision: **0.9840**
- Context recall: **1.0000**
- Mean end-to-end answer latency recorded by the evaluation: **8.16 s**

These measurements support keeping the hybrid + reranker design. The faithfulness result also shows that generation is not perfect; Q08, Q10, and Q20 require manual review rather than presenting the benchmark as flawless.

## Alternatives considered
Pure vector retrieval was rejected because technical agriculture questions contain exact names, units, acronyms, and terminology that BM25 can recover well. Cloud vector databases were unnecessary at this corpus scale and add credentials/cost. Paid embedding and reranking APIs were rejected because local multilingual models provide repeatable evaluation with zero per-query API charge. Earlier OpenAI, Gemini, and Groq evaluation routes were not retained because free-credit/access/rate-limit constraints made the final capstone run unreliable. Cohere Command R7B was accepted after the native structured-output path completed all 20 RAGAS questions successfully.

## Consequences
The design is inexpensive and reproducible, but CPU deployments have model cold-start and memory costs. The local index must be rebuilt when corpus/chunking/embedding settings change. A shared password is sufficient for the capstone but not for multi-tenant production. Production cost estimates must use measured Cohere `billed_units` from live traffic. Three real-user tests remain a human deliverable and must not be fabricated.
