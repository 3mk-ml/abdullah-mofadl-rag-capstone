# ADR-001 — AgriRAG Capstone Architecture

**Status:** Proposed for implementation and validation  
**Date:** 2026-09-28

## Context
The capstone requires a publicly deployed RAG system over 20–50 high-quality documents, hybrid search, reranking, Recall@5 evaluation, RAGAS, authentication, and cost analysis. The system must be feasible on a student laptop and a low-cost public deployment while preserving enough retrieval quality to exceed 80% Recall@5 on a manually reviewed 30-question set.

## Decision
AgriRAG will use authoritative agriculture documents from FAO, ICARDA/CGIAR, and the World Bank. Documents are extracted with PyMuPDF and split with recursive structure-aware chunking. The baseline uses `intfloat/multilingual-e5-small` embeddings and persistent Chroma. Retrieval combines dense vector search with BM25 using Reciprocal Rank Fusion. The fused candidate set is reranked with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, and the top five passages are sent to an OpenAI model via the Responses API. Streamlit provides a bilingual English/Arabic interface, Arabic RTL/Amiri styling, a simple password gate, and source display. Deployment target is Hugging Face Spaces.

## Rationale
This architecture separates expensive generation from low-cost local retrieval. Multilingual local embedding and reranking support Arabic questions without spending the limited API balance on indexing or rerank calls. Chroma is appropriate for a small capstone corpus and avoids a separate server. Hybrid retrieval is selected because technical agriculture questions mix semantic paraphrases with exact terms, crop names, units, and acronyms. RRF avoids incompatible raw score scales. A cross-encoder reranker improves precision after broad retrieval while remaining small enough for CPU use. Streamlit and Hugging Face Spaces minimize deployment work before the deadline.

## Alternatives considered
Paid embeddings/reranking were rejected as the default because repeated experiments would spend the limited course balance. Pinecone/Qdrant cloud and pgvector were rejected because the corpus does not justify extra infrastructure. Pure vector retrieval was rejected because it can miss lexical identifiers and the capstone explicitly requires hybrid search. Semantic chunking was deferred because its extra complexity should be justified by measured Recall@5 rather than assumed.

## Consequences
The first deployment may have a cold-start delay while local transformer models load. The index must be rebuilt whenever chunking or embedding settings change. Quality claims are not assumed: the final architecture values are accepted only after the 30-question Recall@5 run and the 20-question RAGAS evaluation. If the baseline misses the target, the tuning order is chunking -> candidate depth/BM25 -> reranker -> embedding model.
