# ─────────────────────────────────────────────────────────────────────────────
# SOMA Avatar System – Dockerfile
#
# Build strategy: 4 cache layers, ordered by change frequency (slowest first)
#
#   Layer 1 (base-ml):     CUDA + PyTorch + heavy scientific packages
#                          → rebuilds only when torch version changes (~monthly)
#
#   Layer 2 (fitting):     SMPL-X, trimesh, git-installed fitting tools
#                          → rebuilds only when fitting deps change (~weekly)
#
#   Layer 3 (app-deps):    Django, Channels, Redis client, lightweight deps
#                          → rebuilds when requirements.txt changes
#
#   Layer 4 (app):         Application code
#                          → rebuilds on every code change (fast, no pip)
#
# Usage:
#   docker build -t soma-avatar .
#   docker compose up
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base-ml ─────────────────────────────────────────────────────────
# Use CUDA base if GPU available, else CPU-only torch
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base-ml

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3.11 -m pip install --upgrade pip setuptools wheel

# ── Heavy ML packages (own cache layer) ──────────────────────────────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN python3.11 -m pip install \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && python3.11 -m pip install -r /tmp/requirements-ml.txt


# ── Stage 2: fitting ──────────────────────────────────────────────────────────
FROM base-ml AS fitting

# SMPL-X and geometry tools
COPY requirements-fitting.txt /tmp/requirements-fitting.txt
RUN python3.11 -m pip install -r /tmp/requirements-fitting.txt

# pytorch3d (build from source for CUDA compatibility)
RUN python3.11 -m pip install \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || echo "pytorch3d build failed – falling back to CPU render"

# RAFT-Stereo (optical flow)
RUN git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo.git /opt/raft-stereo \
    && (python3.11 -m pip install -r /opt/raft-stereo/requirements.txt 2>/dev/null || echo "RAFT-Stereo requirements not found – skipping")

# DECA (face fitting)
RUN git clone --depth 1 https://github.com/yfeng95/DECA.git /opt/deca \
    && python3.11 -m pip install -r /opt/deca/requirements.txt 2>/dev/null || true

# Real-ESRGAN (texture super-resolution)
RUN python3.11 -m pip install basicsr facexlib gfpgan \
    && python3.11 -m pip install realesrgan

# SCHP (clothing segmentation)
RUN git clone --depth 1 https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git /opt/schp \
    && python3.11 -m pip install -r /opt/schp/requirements.txt 2>/dev/null || true

# Set PYTHONPATH so app can import these tools
ENV PYTHONPATH="/opt/raft-stereo:/opt/deca:/opt/schp:${PYTHONPATH}"


# ── Stage 3: app-deps ─────────────────────────────────────────────────────────
FROM fitting AS app-deps

COPY requirements.txt /tmp/requirements.txt
RUN python3.11 -m pip install -r /tmp/requirements.txt


# ── Stage 4: app (final) ──────────────────────────────────────────────────────
FROM app-deps AS app

# Create non-root user
RUN groupadd -r soma && useradd -r -g soma -d /app -s /sbin/nologin soma

WORKDIR /app

# Copy application code (this layer rebuilds on every code change – fast)
COPY --chown=soma:soma . .

# Create runtime directories
RUN mkdir -p \
    /app/avatar_data \
    /app/video_data \
    /app/media \
    /app/staticfiles \
    /data/db \
    && chown -R soma:soma /app /data

USER soma

# Collect static files
RUN python manage.py collectstatic --noinput 2>/dev/null || true

EXPOSE 8000

# Use daphne for ASGI (Django Channels)
CMD ["daphne", \
     "-b", "0.0.0.0", \
     "-p", "8000", \
     "--websocket_timeout", "3600", \
     "avatar_system.asgi:application"]
