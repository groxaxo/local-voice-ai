# syntax=docker/dockerfile:1.5
ARG SOPRANO_BASE=python:3.11-slim
FROM ${SOPRANO_BASE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install soprano-tts uvicorn

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "soprano.server:app", "--host", "0.0.0.0", "--port", "8000"]
