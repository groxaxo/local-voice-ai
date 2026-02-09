"""Minimal OpenAI-compatible transcription server backed by NVIDIA Parakeet (NeMo)."""

import io
import logging
import os
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("parakeet-server")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv(
    "PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"
)

app = FastAPI(title="Parakeet STT API")

_asr_model = None


def _get_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr

        logger.info("Loading Parakeet model: %s", MODEL_NAME)
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        logger.info("Parakeet model loaded successfully")
    return _asr_model


@app.on_event("startup")
async def startup():
    _get_model()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=None),
    language: str = Form(default="en"),
    response_format: str = Form(default="json"),
):
    """OpenAI-compatible transcription endpoint."""
    audio_bytes = await file.read()

    # Write to a temporary file so NeMo can read it
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        asr_model = _get_model()
        results = asr_model.transcribe([tmp_path])

        # NeMo returns a list of Hypothesis objects or plain strings
        if results and hasattr(results[0], "text"):
            text = results[0].text
        elif results and isinstance(results[0], str):
            text = results[0]
        else:
            text = str(results[0]) if results else ""

    finally:
        os.unlink(tmp_path)

    return JSONResponse(content={"text": text})
