# ═══════════════════════════════════════════════════════════════
# Jetson-Dream: Desktop/Server Dockerfile (CUDA)
# ═══════════════════════════════════════════════════════════════
# For Jetson Nano, use Dockerfile.jetson instead.
#
# Build:
#   docker build -t jetson-dream .
#
# Run:
#   docker run --gpus all --device /dev/video0 \
#     -v $(pwd)/models:/app/models \
#     -p 8080:8080 \
#     jetson-dream --stream
# ═══════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libv4l-dev v4l-utils \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY scripts/ scripts/
COPY demo_image.py .

# Download default style models (cached layer)
RUN python3 -m scripts.download_styles -o models --subset core

# Copy any pre-existing models
COPY models/ models/

# Ports: MJPEG stream
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import scripts.dream_engine; print('ok')" || exit 1

ENTRYPOINT ["python3", "-m", "scripts.main"]
CMD ["--stream", "--process-res", "medium"]
