# Job Decision Engine — Docker image for HuggingFace Spaces.
#
# Reproducibility lock (Step 5 rule #2):
# - Pinned Python base (python:3.12-slim-bookworm).
# - Pinned deps via requirements.txt (exact == pins).
# - Non-root user matching HF Space's uid 1000 requirement.
# - Model pre-download at build time so first request isn't slow on cold start.
# - Same output anywhere — local `docker run` and HF Space behave identically.
#
# HF Space convention:
# - SDK: docker
# - Port: 7860 (EXPOSE + Streamlit CLI flag)
# - Config + secrets: env vars (OPENAI_API_KEY, MONGODB_URI) injected at runtime
#
# Read-only runtime (Step 5 rule #3):
# - No runtime pip installs.
# - No mutation of /app at container start.
# - Provider selection is surfaced in the UI; no hidden switching.

FROM python:3.14-slim-bookworm

# HF Spaces require uid 1000 to own /home/user/app. Create the user first.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /home/user/app

# Install deps first to keep this layer cacheable across source edits.
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-warm the sentence-transformers model at build time. First real
# request becomes instant; subsequent image rebuilds reuse the layer.
# The download path ~/.cache/huggingface/hub is owned by `user`.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the source tree. .dockerignore keeps MEMORY/, docs/, tests/, .git/,
# EXECUTION_RULES.md, __pycache__, etc. out of the image.
COPY --chown=user:user src ./src
COPY --chown=user:user streamlit_app ./streamlit_app
COPY --chown=user:user scripts ./scripts
COPY --chown=user:user README.md ./

# HF Space port convention.
EXPOSE 7860

# Streamlit-specific env: disable usage stats collection (slow + noisy on
# locked-down networks) and disable CORS/XSRF (HF Space proxies in front).
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

CMD ["python", "-m", "streamlit", "run", "streamlit_app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true"]
