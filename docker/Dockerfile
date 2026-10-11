# syntax=docker/dockerfile:1.7
# legal-rag API image.
#   docker build -f docker/Dockerfile -t <user>/legal-rag-api:0.1.0 .

# ---------- builder: install the locked dependencies ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Optional HTTPS-inspection root CA, passed as a BuildKit secret. Builder stage only (never
# shipped); a no-op without the secret.
RUN --mount=type=secret,id=local_ca,required=false \
    if [ -s /run/secrets/local_ca ]; then cat /run/secrets/local_ca >> /etc/ssl/certs/ca-certificates.crt; fi

# dependencies first: this layer stays cached until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group monitoring --no-install-project

# then the project itself, installed non-editable into the venv
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group monitoring --no-editable

# ---------- runtime: the venv only (no uv, no build cache, no source tree) ----------
FROM python:3.12-slim AS runtime
# Volume mount points must exist and be owned by app, otherwise Docker creates them as root.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/feedback /app/events /home/app/.cache/huggingface \
 && chown -R app:app /app/feedback /app/events /home/app/.cache
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
USER app
EXPOSE 8000
# /health is a readiness check: 200 only when the index is reachable and non-empty.
# start-period covers loading bge-m3 (and downloading it on first start).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "legalrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
