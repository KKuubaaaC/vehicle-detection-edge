# Multi-stage Dockerfile for Vehicle Detection on Raspberry Pi
# Optimized for ARM64 architecture

FROM python:3.9-slim-bullseye as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deploy_rpi.py .
COPY inference.py .
COPY export_onnx.py .

# Create directories
RUN mkdir -p models results data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ONNXRUNTIME_FORCE_CPU=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import onnxruntime; import cv2; import numpy" || exit 1

ENTRYPOINT ["python", "deploy_rpi.py"]
CMD ["--help"]