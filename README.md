# 🌿 AgriRAG — Evidence-Grounded Agriculture Assistant

**Done by:** Abdullah Mofadl

AgriRAG is a production-style Retrieval-Augmented Generation capstone for irrigation, soil fertility, plant nutrition, crop water stress, water harvesting, and climate-smart agriculture. It was built for the Yemen4LLM Track B Week 3 RAG capstone.

**Repository:** https://github.com/3mk-ml/abdullah-mofadl-rag-capstone  
**Live demo:** https://agrirag-production.up.railway.app

## Final verified results

| Requirement / metric | Final result |
|---|---:|
| Final curated corpus | **20 documents** |
| Extracted pages/units | **1,828** |
| Indexed chunks | **5,540** |
| Golden retrieval questions | **30** |
| Recall@5 | **90.00% (27/30)** |
| RAGAS questions | **20/20 completed** |
| RAGAS faithfulness | **0.8329** |
| RAGAS answer relevancy | **0.9819** |
| RAGAS context precision | **0.9840** |
| RAGAS context recall | **1.0000** |
| Mean evaluation latency | **8.16 s** |
| Median evaluation latency | **7.31 s** |

Final RAGAS workflow: [GitHub Actions run #21](https://github.com/3mk-ml/abdullah-mofadl-rag-capstone/actions/runs/36805862750)

Machine-readable reports:
- `data/eval/recall_at_5_report.json`
- `data/eval/ragas_report.csv`
- `data/eval/ragas_summary.json`

The strict Recall@5 misses were Q03, Q12, and Q24 and were retained rather than relabelled after seeing retrieval results. RAGAS also exposed faithfulness weaknesses on some questions; the score is reported as measured rather than presented as perfect.

## Architecture

```text
OFFLINE
Official agriculture documents
  -> PyMuPDF
  -> recursive structure-aware chunks (~350 tokens, ~60 overlap)
  -> intfloat/multilingual-e5-small
  -> Chroma vector index + BM25 index

ONLINE
English agriculture question
  -> E5 query embedding
  -> vector top-20 + BM25 top-20
  -> Reciprocal Rank Fusion
  -> multilingual cross-encoder reranker
  -> top-5 evidence passages
  -> Cohere command-r7b-12-2024
  -> evidence-grounded answer + [n] citations
```

See `architecture.md` for the technical rationale and `ADR.md` for the final one-page architecture decision.

## Final technology stack
- Python 3.11
- Streamlit
- PyMuPDF
- `intfloat/multilingual-e5-small`
- Chroma
- `rank-bm25`
- Reciprocal Rank Fusion
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for the validated local/CI reranker
- Cohere `rerank-v3.5` for the 1 GB Railway live-demo reranker
- Cohere `command-r7b-12-2024` for generation
- RAGAS 0.4.3
- Docker
- Railway-ready deployment configuration

## Repository structure

```text
.
├── app.py
├── domain.md
├── architecture.md
├── ADR.md
├── cost_analysis.md
├── user_testing.md
├── requirements.txt
├── Dockerfile
├── railway.toml
├── data/
│   ├── sources_manifest.csv
│   ├── raw/
│   ├── index/
│   └── eval/
│       ├── golden_questions.json
│       ├── recall_at_5_report.json
│       ├── ragas_report.csv
│       └── ragas_summary.json
├── scripts/
│   ├── download_corpus.py
│   ├── check_corpus.py
│   ├── ingest.py
│   ├── evaluate_recall.py
│   ├── evaluate_ragas.py
│   └── cost_analysis.py
└── src/
    ├── auth.py
    ├── chunking.py
    ├── embeddings.py
    ├── bm25_index.py
    ├── retrieval.py
    ├── reranker.py
    ├── llm.py
    └── rag_pipeline.py
```

## Run locally

### Windows PowerShell

```powershell
git clone https://github.com/3mk-ml/abdullah-mofadl-rag-capstone.git
cd abdullah-mofadl-rag-capstone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Linux/macOS

```bash
git clone https://github.com/3mk-ml/abdullah-mofadl-rag-capstone.git
cd abdullah-mofadl-rag-capstone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure at minimum:

```text
LLM_PROVIDER=cohere
COHERE_API_KEY=...
COHERE_MODEL=command-r7b-12-2024
APP_PASSWORD=...
```

Never commit the real `.env` or API keys.

## Prepare corpus and index

```bash
python scripts/download_corpus.py
python scripts/check_corpus.py
python scripts/ingest.py
```

The downloader requires at least 20 successful source documents. The final verified corpus has 20 documents.

## Retrieval evaluation

```bash
python scripts/evaluate_recall.py \
  --golden data/eval/golden_questions.json \
  --k 5 \
  --fail-below 0.80
```

Verified result: **Recall@5 = 90.00% (27/30)**.

The metric uses manually verified exact gold chunk IDs, not post-hoc source relabelling.

## RAGAS evaluation

The project evaluates:
- faithfulness
- answer relevancy
- context precision
- context recall

Run locally:

```bash
python scripts/evaluate_ragas.py \
  --golden data/eval/golden_questions.json \
  --out data/eval/ragas_report.csv \
  --checkpoint data/eval/ragas_checkpoint.json \
  --evaluator-model command-r7b-12-2024
```

For GitHub Actions, configure the repository secret:

```text
COHERE_API_KEY
```

The successful final evaluation was **run #21**, using Cohere Command R7B for both answer generation and evaluation with local E5 embeddings for answer-relevancy scoring.

## Run the UI

```bash
streamlit run app.py
```

Features:
- polished English-only interface
- dropdown question picker backed by the validated 30-question golden set
- custom-question input
- shared-password authentication
- vector + BM25 + RRF hybrid retrieval
- multilingual reranking
- a fixed **top 5** evidence passages, matching the validated evaluation setting
- visible evidence snippets and source/page metadata
- answer latency display

## Cost

See `cost_analysis.md` for the final cost model.

Current Command R7B production pricing used in the report:
- input: **$0.0375 / 1M tokens**
- output: **$0.15 / 1M tokens**

With the report's conservative 2,500-input / 250-output token planning envelope, estimated generation cost is **$0.00013125/query**, or about:
- **$0.13 / 1K queries**
- **$1.31 / 10K queries**
- **$13.13 / 100K queries**

Embedding and reranking API cost is $0 because both are local.

## Live deployment

Production URL: **https://agrirag-production.up.railway.app**

Railway's deployment completed successfully and the Streamlit health endpoint `/_stcore/health` returned HTTP 200 after deployment.

## Docker / Railway deployment

The Docker image runs Streamlit on the platform-provided `PORT` (falling back to 7860). `railway.toml` contains the deployment health check.

Required deployment variables:

```text
LLM_PROVIDER=cohere
COHERE_API_KEY=<secret>
COHERE_MODEL=command-r7b-12-2024
APP_PASSWORD=<secret>
```

The deployment process must have enough memory for its active retrieval models.

### Railway memory profile

The capstone's strict Recall@5 evaluation was produced with the local multilingual cross-encoder. Railway's current 1 GB service limit cannot keep both the local E5 embedder and the local cross-encoder resident simultaneously: observed memory reached about 0.986 GB and the process was killed while the second model loaded. The live Railway service therefore uses `RERANKER_PROVIDER=cohere` with `rerank-v3.5`, while keeping the same vector + BM25 + RRF candidate generation and fixed top-5 evidence output. Local/CI evaluation remains `RERANKER_PROVIDER=local` for exact reproduction of the published Recall@5 result.

## User testing

The capstone requires testing with **three real users**. `user_testing.md` intentionally remains a real-user log and must not be fabricated. Each tester should sign in, ask straightforward and multi-concept questions, inspect evidence, and report what was confusing before the final submission.

## Safety and scope

AgriRAG is an educational decision-support assistant, not a substitute for local agricultural expertise. It must not invent crop thresholds, fertilizer/pesticide doses, irrigation quantities, or source citations. If the retrieved corpus does not contain enough evidence, the correct behavior is to say that the evidence is insufficient.

## Final submission checklist

- [x] Public GitHub repository
- [x] 20–50 high-quality documents (**20 final documents**)
- [x] Justified chunking / embedding / vector database decisions
- [x] Hybrid search + reranking
- [x] 30 manually verified golden questions
- [x] Recall@5 >= 80% (**90.00%**)
- [x] Streamlit UI
- [x] Authentication implementation
- [x] Polished English-only UI (RTL/Amiri not applicable because Arabic mode was removed)
- [x] 20-question RAGAS report
- [x] Final RAGAS metrics in README
- [x] One-page accepted ADR
- [x] Cost analysis with 1K / 10K / 100K scenarios
- [x] Docker deployment configuration
- [x] Live public demo URL
- [ ] Three real-user tests recorded

The only remaining unchecked project item is testing with three real users; that feedback must come from actual people and is deliberately not fabricated.
