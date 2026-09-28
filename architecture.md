# AgriRAG Architecture

## 1. System overview

### Offline ingestion loop
`PDF/TXT/MD -> text extraction -> recursive chunking -> multilingual embeddings -> Chroma vector index + BM25 lexical index`

### Online query loop
`Question -> multilingual query embedding -> vector top-k + BM25 top-k -> Reciprocal Rank Fusion -> multilingual cross-encoder reranker -> top 5 evidence chunks -> LLM answer with citations`

## 2. Chunking decision
**Choice:** recursive structure-aware chunking, initial target about 700 tokens with about 100-token overlap, then tune against Recall@5.

**Why it fits this corpus:** FAO/ICARDA manuals contain long technical paragraphs, numbered sections, equations, and page-level concepts. Fixed-size splitting can cut an irrigation rule away from its explanation. Recursive splitting tries paragraph and sentence boundaries first while keeping a controlled maximum size.

**Why not semantic chunking initially:** semantic chunking would add embedding calls and tuning complexity before we know it is necessary. The capstone has a short deadline and requires measured improvement. We begin with a strong deterministic baseline, then change it only if the 30-question retrieval evaluation shows a need.

**Why overlap:** modest overlap protects context that crosses paragraph/chunk boundaries without excessively duplicating the corpus.

## 3. Embedding model decision
**Choice:** `intfloat/multilingual-e5-small` (384 dimensions).

**Why:**
- Multilingual retrieval supports Arabic user questions over mostly English technical manuals.
- Small enough for CPU deployment and local development.
- No per-document embedding API bill, important because the course API balance is limited.
- 384-dimensional vectors keep the local Chroma index compact.

**Alternative rejected for the baseline:** paid embedding APIs can be excellent, but would consume the course balance during repeated indexing experiments. If Recall@5 remains below target after chunk/retrieval tuning, an API embedding model can be tested as an ablation.

## 4. Vector database decision
**Choice:** persistent Chroma.

**Why it fits this project:**
- Corpus is only 20–50 documents, not millions of vectors.
- Zero infrastructure cost.
- Persists locally and can ship with the deployment artifact.
- Fast enough for a portfolio-scale capstone.

**Why not Pinecone/Qdrant cloud:** those are strong production choices, but add deployment credentials, network failure modes, and monthly cost without solving a scale problem this corpus actually has.

**Why not pgvector:** excellent when PostgreSQL is already part of the product. AgriRAG has no relational workload that justifies introducing a database server solely for vectors.

## 5. Hybrid retrieval decision
**Vector search:** captures semantic similarity and paraphrases.

**BM25:** captures exact agronomic terms, crop names, model names, codes, units, and rare phrases that dense retrieval can miss.

**Fusion:** Reciprocal Rank Fusion (RRF) combines the rank positions rather than raw scores, avoiding difficult score normalization between BM25 and cosine similarity.

**Candidate depth:** start with vector top-20 + BM25 top-20; fuse; rerank; return top-5.

## 6. Reranker decision
**Choice:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

**Why:**
- Cross-encoder reranking scores the query and passage jointly, improving precision after broad first-stage retrieval.
- The model supports Arabic and English, matching the bilingual query requirement.
- It is smaller than very large multilingual rerankers, making CPU deployment more practical.
- No per-query reranking API cost.

## 7. Generator decision
**Choice:** OpenAI Responses API with model name controlled by `OPENAI_MODEL`; baseline default is `gpt-5.6-luna` for a cost-sensitive portfolio workload.

The prompt explicitly requires answers only from retrieved evidence, says to report insufficient evidence, and requires [n] source citations. The model is deliberately configurable so the student can use the model allowed by the course key without changing application code.

## 8. Arabic handling
- Arabic/English multilingual embeddings and reranking
- Conservative Unicode/whitespace normalization
- Arabic normalization only in BM25 tokenization (not destructive rewriting of source passages)
- Streamlit RTL mode and Amiri font

## 9. Evaluation
### Retrieval
30 golden questions. Each question includes a manually verified gold passage/source. Metric: Recall@5 = fraction of questions for which the gold evidence appears in the top five final retrieved chunks.

Target: **>= 80%**.

### End-to-end
20 golden questions with ground-truth answers. RAGAS report includes at minimum faithfulness, answer relevancy, context precision, and context recall.

### Tuning order
1. Inspect misses manually.
2. Tune chunk size/overlap.
3. Tune vector_k/BM25_k.
4. Verify BM25 tokenization for Arabic and technical terms.
5. Compare with/without reranker.
6. Only then test a different embedding model.

This order minimizes API spend and preserves a defensible experimental trail.

## 10. Interface and authentication
Streamlit is used because one framework satisfies the assignment and keeps deployment simple. Authentication is a course-appropriate shared password stored in deployment secrets, not in Git.

## 11. Deployment
Primary target: **Hugging Face Spaces** with Streamlit. The index is built before deployment and committed via Git LFS if necessary. Secrets hold `OPENAI_API_KEY`, `OPENAI_MODEL`, and `APP_PASSWORD`.

## 12. Failure modes and mitigations
| Failure | Detection | Mitigation |
|---|---|---|
| Scanned PDF extracts empty text | corpus check shows near-zero characters | replace with OCR/text version before indexing |
| Exact crop/chemical term missed | BM25 result better than vector result | retain hybrid retrieval + inspect tokenizer |
| Correct chunk retrieved but ranked low | gold appears in candidates but not top 5 | reranker/tune candidate depth |
| Unsupported answer | citation does not support claim | stronger prompt, RAGAS faithfulness review, manual spot checks |
| Slow Space cold start | first query much slower | prebuild index; cache embedding and reranker models |
| API balance unexpectedly drops | token/cost log rises | keep local embedding/reranking; cap context; run RAGAS once after retrieval is stable |
