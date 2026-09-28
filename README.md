# 🌿 AgriRAG — Evidence-Grounded Agriculture Assistant

A production-style Retrieval-Augmented Generation capstone for irrigation, soil fertility, plant nutrition, crop water stress, dryland agriculture, and climate-smart agriculture.

The project is designed for the Yemen4LLM Track B Week 3 capstone and includes the complete submission path: 20–50 curated documents, ingestion, hybrid retrieval, multilingual reranking, Recall@5 evaluation, Streamlit authentication, Arabic RTL/Amiri support, RAGAS, cost analysis, and public deployment.

## Why this project

Agricultural guidance is spread across long FAO, ICARDA/CGIAR, and World Bank technical reports. General LLMs can sound confident while inventing agronomic thresholds or doses. AgriRAG retrieves evidence first, reranks it, and instructs the generator to answer only from the supplied sources with citations.

## Architecture

```text
OFFLINE INGESTION
PDF/TXT/MD
   -> PyMuPDF extraction
   -> recursive structure-aware chunks
   -> multilingual-e5-small embeddings
   -> persistent Chroma
   -> BM25 lexical index

ONLINE QUERY
User question (Arabic or English)
   -> query embedding
   -> vector top-20 + BM25 top-20
   -> Reciprocal Rank Fusion (RRF)
   -> multilingual cross-encoder reranker
   -> top-5 evidence chunks
   -> OpenAI Responses API
   -> grounded answer + source/page citations
```

See `architecture.md` for the written technical justifications and `ADR.md` for the one-page decision record.

## Tech stack

- Python 3.11
- PyMuPDF for extraction
- `intfloat/multilingual-e5-small` for local multilingual embeddings
- Chroma for persistent vector storage
- `rank-bm25` for lexical retrieval
- Reciprocal Rank Fusion for hybrid ranking
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for multilingual reranking
- OpenAI Responses API for final answer generation
- Streamlit for UI and authentication
- RAGAS for end-to-end evaluation
- Hugging Face Spaces (Docker) for deployment

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
├── data/
│   ├── sources_manifest.csv
│   ├── raw/                  # put the 20–50 final documents here
│   ├── index/                # generated Chroma/BM25/chunk files
│   └── eval/
│       └── golden_questions.json
├── scripts/
│   ├── check_corpus.py
│   ├── ingest.py
│   ├── evaluate_recall.py
│   ├── evaluate_ragas.py
│   └── cost_analysis.py
├── src/
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── bm25_index.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── llm.py
│   └── rag_pipeline.py
└── tests/
    └── test_smoke.py
```

## 1. Clone and create environment

Windows PowerShell:

```powershell
git clone https://github.com/3mk-ml/abdullah-mofadl-rag-capstone.git
cd abdullah-mofadl-rag-capstone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS:

```bash
git clone https://github.com/3mk-ml/abdullah-mofadl-rag-capstone.git
cd abdullah-mofadl-rag-capstone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set at minimum:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-luna
APP_PASSWORD=...
```

Do not commit `.env`.

## 2. Collect the corpus

`data/sources_manifest.csv` contains 25 high-quality source candidates from FAO, ICARDA/CGIAR, and the World Bank. The curated manifest now contains 24 official FAO/FAO-partner publications. Download them reproducibly with:\n\n```bash\npython scripts/download_corpus.py\n```\n\nThe downloader saves the corpus into `data/raw/`, writes SHA-256 inventory metadata, and fails the run if fewer than 20 documents are collected.

Important rules:
- Prefer complete PDF publications, not random blogs.
- Keep the original descriptive filename.
- Avoid duplicate editions of the same document unless there is a clear reason.
- If a PDF is scanned and extracts no text, replace it with an OCR/text version before indexing.

Check the corpus:

```bash
python scripts/check_corpus.py
```

## 3. Build the index

```bash
python scripts/ingest.py
```

This creates:
- `data/index/chroma/`
- `data/index/bm25.pkl`
- `data/index/chunks.jsonl`

Do not run ingestion repeatedly unless the corpus, chunking, or embedding configuration changed. This project deliberately uses local embedding to avoid unnecessary API spend.

## 4. Build the 30-question golden set

Open `data/eval/golden_questions.json`. The file already contains 30 realistic domain questions, but the gold fields are intentionally blank because they must be verified against the **actual final corpus**.

For every question:
1. Find the passage that really answers it.
2. Fill `gold_source` with the exact filename.
3. Add a distinctive `must_contain` phrase or, preferably, the final `gold_chunk_ids` after inspecting `chunks.jsonl`.
4. Write a concise human `ground_truth` answer for at least 20 questions used in RAGAS.

Do not invent gold answers before checking the source.

## 5. Measure Recall@5

```bash
python scripts/evaluate_recall.py --golden data/eval/golden_questions.json --k 5
```

Submission target: **Recall@5 >= 80%**.

If it misses the target, tune in this order:
1. inspect misses manually;
2. change chunk size/overlap;
3. change vector/BM25 candidate depth;
4. inspect BM25 tokenization and technical terms;
5. compare reranker on/off;
6. only then test a different embedding model.

Record every experiment in a small table in your final README. This makes the improvement defensible rather than anecdotal.

## 6. Run locally

```bash
streamlit run app.py
```

The UI supports English and Arabic. Arabic mode is RTL and loads the Amiri font. The password comes from `APP_PASSWORD`.

## 7. Test with three real users

Use `user_testing.md`. Test the live app with three actual people and record what they tried and what you changed. Do not fabricate this deliverable.

## 8. RAGAS evaluation

### GitHub Actions RAGAS workflow

A manual workflow is included at `.github/workflows/ragas.yml`.

Before running it, add a repository Actions secret named `OPENAI_API_KEY` under:

`Settings -> Secrets and variables -> Actions -> New repository secret`

Then open **Actions -> Run 20-question RAGAS evaluation -> Run workflow**. The workflow rebuilds the verified corpus/index, confirms Recall@5 is still >=80%, runs RAGAS on 20 reviewed questions, and uploads `ragas_report.csv` plus `ragas_summary.json`.



After retrieval is stable and Recall@5 has cleared the target, run RAGAS **once** on 20 reviewed questions to protect the API balance:

```bash
python scripts/evaluate_ragas.py \
  --golden data/eval/golden_questions.json \
  --out data/eval/ragas_report.csv
```

The script evaluates:
- faithfulness
- answer relevancy
- context precision
- context recall

Save the CSV and summarize the mean scores in this README before submission.

## 9. Cost analysis

The baseline cost model is in `cost_analysis.md`. Before submission, replace assumed token counts with measured averages from the final app.

```bash
python scripts/cost_analysis.py \
  --input-tokens ACTUAL_INPUT \
  --output-tokens ACTUAL_OUTPUT \
  --input-price CURRENT_INPUT_RATE \
  --output-price CURRENT_OUTPUT_RATE \
  --queries-per-user 10
```

Report both the query-volume view and the required 1K / 10K / 100K user scenarios, with the activity assumption explicitly stated.

## 10. Deploy to Hugging Face Spaces

Create a **Docker Space**, then push this repository to it. The included `Dockerfile` runs Streamlit on port 7860.

Add Space secrets:
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `APP_PASSWORD`

Build the vector/BM25 index before deployment so the public app does not re-index the corpus on every restart. If index files become large, use Git LFS.

## Evaluation results

Fill this only after real runs:

| Metric | Result |
|---|---:|
| Recall@5, 30 questions | **90.00% (27/30)** |
| RAGAS faithfulness | TBD |
| RAGAS answer relevancy | TBD |
| RAGAS context precision | TBD |
| RAGAS context recall | TBD |
| Median latency | TBD |

## Retrieval ablation

Fill this after experiments:

| Variant | Recall@5 | Notes |
|---|---:|---|
| Vector only | TBD | baseline |
| Vector + BM25 + RRF | TBD | hybrid |
| Hybrid + reranker | TBD | final |

## Final submission checklist

- [ ] Public GitHub repository link
- [ ] Live demo URL opens for another person
- [ ] `ADR.md` is one page when rendered/printed
- [ ] `data/eval/ragas_report.csv` generated from 20 reviewed questions
- [ ] `cost_analysis.md` updated with measured tokens and all 3 scenarios
- [ ] 20–50 high-quality documents in the final corpus
- [x] 30 manually verified golden questions
- [x] Recall@5 >= 80% (**90.00%**)
- [ ] Authentication enabled
- [ ] Arabic RTL/Amiri works if Arabic is used
- [ ] Three real-user tests documented
- [ ] README contains exact reproducible run/deploy steps

## Safety and scope

AgriRAG is an educational and decision-support assistant. It must not fabricate agronomic thresholds, chemical doses, or farm-specific prescriptions. When evidence is missing or conflicting, the correct behavior is to say that the provided corpus is insufficient and show the available sources.


## Verified retrieval evaluation

GitHub Actions run [#7](https://github.com/3mk-ml/abdullah-mofadl-rag-capstone/actions/runs/36487406502) evaluated the final hybrid retrieval pipeline against 30 manually verified questions using exact gold chunk IDs.

- **Recall@5: 90.00% (27/30)**
- Required threshold: **>= 80%**
- Result: **passed**
- Indexed corpus: **20 documents, 1,828 extracted units, 5,540 chunks**
- Evaluation report: `data/eval/recall_at_5_report.json`

The three strict-ID misses were Q03, Q12, and Q24. The score is retained as measured rather than relabelled after observing retrieved results.
