<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice AI</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A full-stack, Dockerized AI voice assistant with speech, text, and voice synthesis delivered via WebRTC powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
</div>

## Overview

This repo contains everything needed to run a real-time AI voice assistant locally using:

- **LiveKit** for WebRTC realtime audio + rooms.
- **LiveKit Agents (Python)** to orchestrate the STT → LLM → TTS pipeline.
- **Whisper (via VoxBox)** for speech-to-text.
- **vLLM** for fast GPU-accelerated LLM inference (OpenAI-compatible API) serving **Gemma 3 27B** (GPTQ 4-bit).
- **Kokoro** for text-to-speech voice synthesis.
- **Next.js + Tailwind** frontend UI.
- Fully containerized via Docker Compose.

## Getting Started

Windows uses the PowerShell command; Linux and OSX use the bash command. Both will prompt you to choose CPU or Nvidia GPU.

Windows:
```bash
./compose-up.ps1
```

Mac / Linux:
```bash
chmod +x filename.sh
./compose-up.sh
```

Once it's up, visit [http://localhost:3000](http://localhost:3000) in your browser to start chatting.

### Notes on models and resources

- The LLM runs via **vLLM** and auto-downloads the model from Hugging Face on first boot (no manual model download needed).
- The default model is `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` (change `VLLM_MODEL` in `.env` to use a different model).
- The API exposes the model under an alias (default `gemma-3-27b` via `VLLM_MODEL_ALIAS`); the agent uses that alias.
- **GPU required**: vLLM with GPTQ models requires an NVIDIA GPU. Use `./compose-up.sh gpu` (or `./compose-up.ps1 gpu`) to start with GPU support.
- If you want to use a different STT model, change `VOXBOX_HF_REPO_ID`.
- You can swap out the LLM/STT/TTS URLs to use cloud models if you want (see `livekit_agent/src/agent.py`).
- The first run downloads a lot of data (often tens of GB) for models and supporting libraries. GPU-enabled images are bigger and take longer.
- Ongoing VRAM/RAM usage depends heavily on the model, context size, and `VLLM_GPU_MEMORY_UTILIZATION` setting.

### Startup readiness

`vllm` returns errors while the model is downloading/loading/warming up. The Compose stack includes a healthcheck for `vllm`, and `livekit_agent` waits until `vllm` is healthy (i.e. `/v1/models` responds) before starting.

## Architecture

Each service is containerized and communicates over a shared Docker network:

- `livekit`: WebRTC signaling server
- `livekit_agent`: Python agent (LiveKit Agents SDK)
- `whisper`: Speech-to-text (VoxBox + Whisper)
- `vllm`: GPU-accelerated LLM provider (vLLM, OpenAI-compatible API)
- `kokoro`: TTS engine
- `frontend`: Next.js client UI

## Agent

The agent entrypoint is `livekit_agent/src/agent.py`. It uses the LiveKit Agents OpenAI-compatible plugins to talk to local inference services:

- `openai.STT` → the VoxBox Whisper container
- `openai.LLM` → `vllm` (vLLM OpenAI-compatible server)
- `openai.TTS` → the Kokoro container
- `silero.VAD` for voice activity detection

## Environment variables

Example env files:

- `.env` (used by Docker Compose)
- `frontend/.env.example`
- `livekit_agent/.env.example`

For local (non-Docker) development, use `.env.local` files:

- `frontend/.env.local`
- `livekit_agent/.env.local`

### LiveKit URLs (important)

The LiveKit URL is used in two different contexts:

- `LIVEKIT_URL` is the internal, server-to-server address (e.g. `ws://livekit:7880`) used by containers like the agent.
- `NEXT_PUBLIC_LIVEKIT_URL` is the browser-reachable LiveKit address returned by the frontend API (e.g. `ws://localhost:7880`).

The frontend only signs tokens; it does not connect to LiveKit directly. The browser connects using the `serverUrl` returned by `/api/connection-details`, so make sure `NEXT_PUBLIC_LIVEKIT_URL` points to a reachable LiveKit endpoint.

### LLM (vLLM) settings

The Compose stack runs vLLM with a HuggingFace model so models are fetched automatically and cached in a Docker volume:

- `VLLM_MODEL`: HuggingFace model ID (e.g. `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g`)
- `VLLM_MODEL_ALIAS`: Name exposed via the API and returned from `/v1/models` (default `gemma-3-27b`)
- `VLLM_BASE_URL`: LLM base URL for the agent (default `http://vllm:8000/v1`)
- `VLLM_HOST_PORT`: Host port mapping for vLLM (default `8000`)
- `VLLM_MAX_MODEL_LEN`: Maximum context length (default `4096`)
- `VLLM_GPU_MEMORY_UTILIZATION`: Fraction of GPU memory to use (default `0.90`)
- `HF_TOKEN`: HuggingFace token (needed if the model is gated)

Model weights are cached in the `vllm-data` Docker volume.

## Development

Use `.env.local` files in both `frontend` and `livekit_agent` dirs to set the dev environment variables for the project. This way, you can run either of those with `pnpm dev` or `uv run python src/agent.py dev` and test them without needing to build the Docker projects.

## Rebuild / redeploy

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

## Project structure

```
.
├─ frontend/        # Next.js UI client
├─ inference/       # Local inference services (llama/whisper/kokoro)
├─ livekit/         # LiveKit server config
├─ livekit_agent/   # Python voice agent (LiveKit Agents)
├─ docker-compose.yml
└─ docker-compose.gpu.yml
```

## Requirements

- Docker + Docker Compose
- **NVIDIA GPU required** for vLLM (GPTQ model inference)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Recommended VRAM: 24GB+ for the default Gemma 3 27B GPTQ model

## Credits

- Built with LiveKit: https://livekit.io/
- Uses LiveKit Agents: https://docs.livekit.io/agents/
- STT via VoxBox + Whisper: https://pypi.org/project/vox-box/
- LLM inference via vLLM: https://github.com/vllm-project/vllm
- Default model: Gemma 3 27B GPTQ: https://huggingface.co/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g
- TTS via Kokoro: https://github.com/remsky/kokoro
