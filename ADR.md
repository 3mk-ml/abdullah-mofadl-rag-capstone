# ADR-001 — AgriRAG Final Architecture

**Status:** Accepted  
**Decision date:** 2026-10-01

## Context
The Track B Week 3 capstone requires a RAG system over 20–50 quality documents, justified ingestion choices, hybrid retrieval with reranking, 30-question Recall@5 evaluation, 20-question RAGAS evaluation, authenticated bilingual UI, deployment, cost analysis, and a one-page architecture decision. AgriRAG must also support Arabic questions over mostly English agricultural references while remaining practical on student-scale infrastructure.

## Decision
AgriRAG uses 20 successfully collected authoritative agriculture documents from FAO/FAO-partners, ICARDA/CGIAR, and World Bank sources.

Documents are extracted with PyMuPDF and split using recursive structure-aware chunking at about 350 tokens with 60-token overlap. Local `intfloat/multilingual-e5-small` embeddings are stored in persistent Chroma. Retrieval combines vector top-20 and BM25 top-20 results with Reciprocal Rank Fusion (`RRF_K=60`). A local `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranks the fused candidates and returns the best five evidence passages.

The final generator is Cohere `command-r7b-12-2024`. The prompt requires answers to use retrieved evidence, preserve units, expose insufficient evidence instead of inventing thresholds or doses, and cite sources. Streamlit provides a polished English-only UI, a validated question picker, source excerpts, latency display, and shared-password authentication. Docker is the deployment boundary and Railway is the primary hosting target.

## Evidence
The final corpus produced **1,828 extracted pages/units and 5,540 chunks**.

The 30-question strict retrieval evaluation achieved **Recall@5 = 90.00% (27/30)**, exceeding the required 80% threshold. The three misses (Q03, Q12, Q24) were retained as measured.

The successful 20-question RAGAS run (#21) produced:

- Faithfulness: **0.8329**
- Answer relevancy: **0.9819**
- Context precision: **0.9840**
- Context recall: **1.0000**
- Mean latency: **8.16 s**

These results support retaining hybrid retrieval and reranking. Faithfulness is not perfect, so generated answers still require source-visible behavior and manual review for higher-risk agronomic decisions.

## Alternatives
Pure vector retrieval was rejected because agriculture questions frequently contain exact crop names, units, acronyms, and technical terminology where BM25 adds value. Managed vector databases were unnecessary at 5,540 chunks. Paid embedding and reranking APIs were rejected because the local multilingual models provide reproducible evaluation with no per-query API cost. Earlier OpenAI, Gemini, and Groq evaluation routes were not retained because available free-access constraints made the final benchmark unreliable.

## Consequences
The system is inexpensive and reproducible but CPU hosting has model cold-start and memory costs. The index must be rebuilt when the corpus, chunking, or embedding model changes. Shared-password authentication is appropriate for the capstone but not for multi-tenant production. Three real-user tests remain a human deliverable and must not be fabricated.
