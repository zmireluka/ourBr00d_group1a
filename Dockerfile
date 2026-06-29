# Base dockerfile for stt and other things, the dockerfile.tts is only for tts
# Base image: NVIDIA CUDA + Python — gives us GPU access
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Version pins (WHY) ──────────────────────────────────────────────────────
# Without "==version", pip pulls the latest version of each package on EVERY
# rebuild. That means a fresh clone (e.g. the professor) or a rebuild in a few
# months gets different packages — and newer releases frequently change their
# API and break our code (transformers already did this to us once). The versions
# below are the TESTED state that was live on the server on 18.06.2026
# (read out with `docker exec ourbr00d-whisper-1 pip freeze`). This way everyone
# reproduces our working stack exactly instead of playing a lottery.
# Note: only directly installed packages are pinned; their sub-dependencies
# (e.g. torch 2.8.0, transformers 4.57.6) are pulled in transitively by whisperx.

# WhisperX (STT) — brings faster-whisper + torch transitively as dependencies
RUN pip install whisperx==3.8.5

# Environment variables
RUN pip install python-dotenv==1.2.2

# WebSocket (live audio transport)
RUN pip install websockets==16.0

# VAD: silero-vad is loaded at runtime via torch.hub — no pip package needed
# Ollama client (LLM integration)
RUN pip install ollama==0.6.2

# Async HTTP client (TTS streaming)
RUN pip install httpx==0.28.1

# Speaker embedding (ECAPA-TDNN, more robust on short segments)
RUN pip install speechbrain==1.1.0

# In-session RAG (ChromaDB in-memory, brings all-MiniLM embedding)
RUN pip install chromadb==1.5.9

# sentence-transformers: runs all-MiniLM on GPU instead of ChromaDB default (ONNX/CPU)
# → query embedding ~ms instead of ~0.5s. Used by server.py + build_knowledge_base.py.
RUN pip install sentence-transformers==5.5.1

# Working directory inside the container
WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
