# syntax=docker/dockerfile:1.5
ARG PARAKEET_BASE=python:3.11-slim
FROM ${PARAKEET_BASE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
    nemo_toolkit[asr] \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

WORKDIR /app
COPY server.py .

ENV PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2
ENV DATA_DIR=/data

EXPOSE 8015

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8015"]
