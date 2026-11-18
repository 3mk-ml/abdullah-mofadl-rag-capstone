# Free Streamlit Community Cloud deployment (AgriRAG)

This branch is a **separate hosting experiment**, not a replacement for the existing Railway Docker production build. It has not been verified live on Streamlit Community Cloud.

1. Visit https://share.streamlit.io and sign in using GitHub.
2. Create app from existing repository: `3mk-ml/abdullah-mofadl-rag-capstone`.
3. Choose branch `deploy/streamlit-free`, entrypoint `app.py` and Python **3.11** in Advanced settings.
4. In Advanced settings > Secrets, paste this TOML, replacing the two placeholders with your *private* values:

   ```toml
   LLM_PROVIDER = "cohere"
   COHERE_API_KEY = "YOUR_PRIVATE_COHERE_API_KEY"
   COHERE_MODEL = "command-r7b-12-2024"
   RERANKER_PROVIDER = "cohere"
   COHERE_RERANK_MODEL = "rerank-v3.5"
   APP_PASSWORD = "YOUR_STRONG_APP_PASSWORD"
   ```

5. Click Deploy, sign in to the app with `APP_PASSWORD`, then try one sample question.

## Important constraints

- The Git repository deliberately **does not include** the PDFs or Chroma/BM25 indices. On the first question, the app downloads the source corpus and embeds/indexes it on the server. This can be slow, and may fail due to free-tier RAM/CPU/disk or source-site download restrictions. No deployment is guaranteed until checked on the live platform.
- Streamlit Cloud's local filesystem is not persistent: after resets the index may be rebuilt. For a reliable app, precompute the index offline, store it as a downloadable release artifact and fetch it at startup (with integrity checking), or use persistent storage.
- The Cohere model and reranker are **third-party API services**: hosting may be free but API requests are only free within any applicable trial/free quota and could cost money afterward. This configuration requires your own key, securely stored in the service's secrets panel, never in Git.
- The previous Railway deployment needed Cohere reranking because loading both local embedding and local cross-encoder was too memory-intensive. This branch uses the same remote reranking approach.
- If Streamlit Cloud cannot complete indexing, the proper next step is an offline-generated retrieval artifact plus a small downloader, not repeated redeploys of the entire raw corpus.
