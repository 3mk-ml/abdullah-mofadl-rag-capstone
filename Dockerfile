FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .

# Railway is CPU-only for this deployment. Installing the CPU wheel first
# prevents pip from pulling multi-gigabyte CUDA runtime packages that are not
# used by the local embedding/reranking models.
RUN python -m pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# Build the exact corpus/index during the image build so the live service
# starts ready to answer queries and does not depend on ephemeral disk state.
RUN python scripts/prepare_deploy.py

# Pre-cache the multilingual reranker as part of the immutable image.
RUN python - <<'PY'
from src.reranker import model
model()
print("Reranker cached")
PY

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860} --server.headless=true"]
