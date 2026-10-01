# AgriRAG Architecture

## 1. System overview

### Offline ingestion
`PDF/TXT/MD -> PyMuPDF extraction -> recursive structure-aware chunking -> multilingual E5 embeddings -> Chroma + BM25`

### Online query
`Question -> E5 query embedding -> vector top-20 + BM25 top-20 -> RRF -> multilingual cross-encoder reranker -> top-5 evidence chunks -> Cohere Command R7B -> grounded answer + citations`

## 2. Final corpus and index
- Curated manifest: 24 official/authoritative candidates
- Successfully collected final corpus: **20 documents**
- Extracted pages/units: **1,828**
- Final chunks: **5,540**
- Primary source families: FAO/FAO-partners, ICARDA/CGIAR, and World Bank

The final 20-document corpus satisfies the course requirement of 20–50 high-quality documents.

## 3. Chunking decision
**Choice:** recursive structure-aware chunking, target about **350 tokens** with about **60-token overlap**.

FAO and related technical manuals contain long paragraphs, numbered sections, equations, and page-level concepts. Recursive splitting preserves paragraph/sentence boundaries better than blind fixed windows. Overlap protects concepts crossing boundaries without duplicating excessive text.

Semantic chunking was not adopted because the deterministic recursive baseline already cleared the required retrieval threshold, so additional complexity was not justified by the measured result.

## 4. Embedding model
**Choice:** `intfloat/multilingual-e5-small` (384 dimensions), run locally.

Reasons:
- Arabic questions can retrieve mostly English source passages.
- Small enough for CPU inference.
- No embedding API charge during ingestion or querying.
- E5 query/passage prefixes are used as intended.

## 5. Vector store
**Choice:** persistent Chroma.

At 5,540 chunks, a managed vector service would add infrastructure without solving a real scale problem. Chroma persists locally and is sufficient for the capstone workload.

## 6. Hybrid retrieval
- Vector candidates: top 20
- BM25 candidates: top 20
- Fusion: Reciprocal Rank Fusion, `RRF_K=60`
- Final reranked evidence: top 5

Vector search covers semantic paraphrase; BM25 covers exact technical terms, crop names, acronyms, and units. RRF avoids comparing incompatible raw BM25 and cosine-score scales.

## 7. Reranker
**Choice:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, local CPU inference.

The reranker jointly scores question + passage and improves final evidence precision after broad hybrid candidate retrieval. Keeping it local removes per-query rerank API cost.

## 8. Generator
**Final choice:** Cohere `command-r7b-12-2024`.

The model is used through Cohere Chat V2 for the final app and was also used consistently in the successful RAGAS evaluation. The application prompt:
- answers only from supplied evidence;
- reports insufficient evidence rather than inventing thresholds/doses;
- preserves units;
- cites retrieved passages using `[n]`;
- answers in Arabic when the question is Arabic and English otherwise.

The provider remains environment-controlled so another provider can be substituted without changing retrieval.

## 9. Arabic handling
- Multilingual E5 retrieval
- Multilingual cross-encoder reranking
- Conservative Unicode/whitespace normalization
- Arabic-specific normalization in BM25 tokenization
- Streamlit RTL mode
- Amiri font for Arabic UI

## 10. Evaluation

### Retrieval
30 manually verified golden questions using strict gold chunk IDs.

**Final Recall@5: 90.00% (27/30).**  
Misses retained as measured: **Q03, Q12, Q24**.

### End-to-end RAGAS
Successful GitHub Actions run: **#21**  
Questions: **20**

| Metric | Final mean |
|---|---:|
| Faithfulness | **0.8329** |
| Answer relevancy | **0.9819** |
| Context precision | **0.9840** |
| Context recall | **1.0000** |
| Mean latency | **8.16 s** |
| Median latency | **7.31 s** |

The RAGAS report is committed at `data/eval/ragas_report.csv`; the machine-readable mean summary is at `data/eval/ragas_summary.json`.

## 11. Interface and authentication
Streamlit provides the capstone UI. A shared password is loaded only from `APP_PASSWORD`; secrets are not committed. The app displays retrieved source snippets and page metadata so users can inspect evidence.

## 12. Deployment
The app is containerized. Railway is the primary deployment target; the same Docker image remains compatible with a Docker-based Hugging Face Space.

Deployment requirements:
- `LLM_PROVIDER=cohere`
- `COHERE_API_KEY`
- `COHERE_MODEL=command-r7b-12-2024`
- `APP_PASSWORD`
- enough RAM for the local embedding and reranker models

The deployment build prepares the corpus/index if they are not already present in the image.

## 13. Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| Source download fails | fewer than 20 valid docs | fail corpus preparation; repair manifest/source |
| PDF extracts little/no text | corpus validation | replace source or OCR before indexing |
| Exact term missed | BM25 beats dense retrieval | retain hybrid retrieval |
| Gold chunk ranked below top 5 | Recall@5 miss report | inspect chunking/candidate depth/reranker |
| Unsupported answer | low faithfulness/manual audit | stricter evidence prompt + source review |
| Trial API 429 | Cohere 429 response | paced requests / production key |
| Cold start | first request latency | prebuild index/models where hosting permits |
| Missing secret | startup/API failure | deployment secret validation |
