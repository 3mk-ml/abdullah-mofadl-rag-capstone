# AgriRAG — Northflank Docker deployment (original Railway pipeline)

This branch uses the **original Railway-tested Dockerfile, app.py, model code and runtime requirements** from `main`. The *only* application configuration change is removing the Railway-specific public hostname from `.streamlit/config.toml`. Do **not** use the `deploy/streamlit-free` branch here: its first-query index architecture differs from this original Docker build.

## Deploy the existing project

1. Sign into [Northflank](https://app.northflank.com/) and connect GitHub.
2. Create a project and a **Combined Service** (or equivalent build-and-deploy service) from repository `3mk-ml/abdullah-mofadl-rag-capstone`, branch `deploy/northflank-docker`.
3. Choose **Dockerfile** as the build method, Dockerfile path `/Dockerfile`, and repository root `/` as build context. No separate build or start command: Dockerfile controls both.
4. Create one **public HTTP endpoint** routed to **internal container port 7860**.
5. Start with at least **1 GiB RAM** and 0.5 vCPU, if the selected free plan supports that size. The Railway 1 GiB deployment needed remote Cohere reranking for memory reasons.
6. Set runtime variables *privately* in the Northflank service UI (not in Git):
   - `LLM_PROVIDER=cohere`
   - `COHERE_API_KEY=<your private Cohere key>`
   - `COHERE_MODEL=command-r7b-12-2024`
   - `RERANKER_PROVIDER=cohere`
   - `COHERE_RERANK_MODEL=rerank-v3.5`
   - `APP_PASSWORD=<choose a strong password>`
   - `PORT=7860` (optional: the Dockerfile defaults to 7860)
7. Allow a sufficiently long **build**: the original Dockerfile downloads the ~20-document corpus, generates the Chroma+BM25 index and caches a reranker model during the image build, rather than making a user's first question wait.
8. Check that the service successfully deploys, visit `/_stcore/health` on its public URL, sign into the app, and test a sample question for a cited answer.

## Important caveats

- Hosting is not deployed merely by creating this branch; you need to connect a Northflank workspace and configure its secrets. Free plan availability/credits can change, and 1 GiB RAM is a practical minimum **estimate**, not a verified sizing guarantee.
- The existing Dockerfile has a PyTorch CPU wheel requirement; if its build fails, inspect the full build log rather than changing the RAG architecture.
- The Dockerfile currently executes `from src.reranker import model; model()` at build time even when `RERANKER_PROVIDER=cohere`, potentially increasing build size and time. This reproduces the Railway build exactly. Only change it if the host hits a build resource limit.
- Cohere API requests are not guaranteed free, regardless of free hosting. Use your own API key and stay within your account's quotas.
- The original app has Streamlit password authentication, and the host must support WebSockets for interactive Streamlit sessions.
- Deploying a Git branch to a new host does not change the original `main` branch or modify the previous Railway deployment.
