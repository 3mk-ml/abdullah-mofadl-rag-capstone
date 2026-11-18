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

5. The retrieval index must first be published as the GitHub Release asset `agrirag-index-v1.zip` under tag `agrirag-streamlit-index-v1`. The workflow `Publish prebuilt AgriRAG search index` on this branch is configured to build/publish it when the workflow itself changes. Check GitHub Actions and Releases for success before testing the application.\n6. Click Deploy, sign in to the app with `APP_PASSWORD`, then try one sample question.

## Important constraints

- The Git repository deliberately **does not include** the PDFs or Chroma/BM25 indices. Instead, GitHub Actions downloads the corpus and builds the Chroma/BM25 search index **once**. Streamlit fetches the prebuilt, SHA256-verified ZIP from a public GitHub Release on the first question. If this release does not exist yet, requests will fail with a clear error rather than hang while rebuilding.\n- To check publication: open `https://github.com/3mk-ml/abdullah-mofadl-rag-capstone/actions` and `https://github.com/3mk-ml/abdullah-mofadl-rag-capstone/releases`. The action's build must succeed and both `.zip` and `.sha256` assets must be visible.
- Streamlit Cloud's local filesystem is not persistent: after resets the index may be rebuilt. For a reliable app, precompute the index offline, store it as a downloadable release artifact and fetch it at startup (with integrity checking), or use persistent storage.
- The Cohere model and reranker are **third-party API services**: hosting may be free but API requests are only free within any applicable trial/free quota and could cost money afterward. This configuration requires your own key, securely stored in the service's secrets panel, never in Git.
- The previous Railway deployment needed Cohere reranking because loading both local embedding and local cross-encoder was too memory-intensive. This branch uses the same remote reranking approach.
- On free Streamlit hosting, initial **download and model loading** can still take time, but it no longer downloads and embeds the entire document collection on a chat request. If the index remains unavailable, inspect the GitHub Actions build logs and the Streamlit runtime logs.
