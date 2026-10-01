FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements-runtime.txt .

# CPU-only PyTorch keeps the production image far smaller than the default
# CUDA-enabled wheel set pulled on generic Linux runners.
RUN python -m pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-runtime.txt

COPY . .

# Build the authoritative corpus and retrieval index into the immutable image.
# This makes each Railway replica self-contained and avoids rebuilding on every
# container restart.
RUN python scripts/prepare_deploy.py

# Cache the reranker during build so the first real query does not need a model
# download.
RUN python - <<'PY'
from src.reranker import model
model()
print("Reranker cached")
PY

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860} --server.headless=true --server.fileWatcherType=none"]
