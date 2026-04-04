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
# nvidia/cuda: well-known amd64 image; Python 3.10 from Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base-ml

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    gosu \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap a fresh pip directly – Ubuntu's pip3 wrapper ignores --index-url correctly
# only after the pip *module* is up-to-date; get-pip.py fixes this in one step.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install --upgrade setuptools wheel

# ── Heavy ML packages (own cache layer) ──────────────────────────────────────
# CPU torch from PyPI – the web container does SMPL-X fitting, not GPU inference.
# All GPU-heavy work (RTMPose, ViTPose) runs in the pose-worker container.
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip3 install torch==2.3.0 torchvision==0.18.0 \
    && pip3 install -r /tmp/requirements-ml.txt


# ── Stage 2: fitting ──────────────────────────────────────────────────────────
FROM base-ml AS fitting

# SMPL-X and geometry tools
COPY requirements-fitting.txt /tmp/requirements-fitting.txt
RUN pip3 install -r /tmp/requirements-fitting.txt \
    && pip3 install hatchling \
    && pip3 install --no-build-isolation --ignore-requires-python \
       git+https://github.com/nghorbani/human_body_prior

# simple_romp: setup.py imports Cython directly without declaring it as build dep,
# so --no-build-isolation is required (uses already-installed Cython from above).
# This step invalidates automatically when requirements-fitting.txt changes.
RUN pip3 install simple_romp --no-build-isolation

# RAFT-Stereo (optical flow)
RUN git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo.git /opt/raft-stereo \
    && (pip3 install -r /opt/raft-stereo/requirements.txt 2>/dev/null || echo "RAFT-Stereo requirements not found – skipping")

# DECA (face fitting)
RUN git clone --depth 1 https://github.com/yfeng95/DECA.git /opt/deca \
    && pip3 install -r /opt/deca/requirements.txt 2>/dev/null || true

# Real-ESRGAN (texture super-resolution)
RUN pip3 install basicsr facexlib gfpgan \
    && pip3 install realesrgan

# SCHP (clothing segmentation)
RUN git clone --depth 1 https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git /opt/schp \
    && pip3 install -r /opt/schp/requirements.txt 2>/dev/null || true

# Set PYTHONPATH so app can import these tools
ENV PYTHONPATH="/opt/raft-stereo:/opt/deca:/opt/schp:${PYTHONPATH}"


# ── Stage 3: app-deps ─────────────────────────────────────────────────────────
FROM fitting AS app-deps

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt


# ── Stage 4: app (final) ──────────────────────────────────────────────────────
FROM app-deps AS app

# Create non-root user
RUN groupadd -r soma && useradd -r -g soma -d /app -s /sbin/nologin soma

WORKDIR /app

# Copy application code (this layer rebuilds on every code change – fast)
COPY --chown=soma:soma . .

# Create runtime directories and set ownership
RUN mkdir -p \
    /app/avatar_data \
    /app/video_data \
    /app/media \
    /app/staticfiles \
    /data/db \
    /data/media \
    /data/media/thumbnails \
    /data/avatars \
    && chown -R soma:soma /app /data

USER soma

# Collect static files
RUN python3 manage.py collectstatic --noinput 2>/dev/null || true

# Switch back to root so the entrypoint can fix volume permissions at runtime
USER root
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["daphne", \
     "-b", "0.0.0.0", \
     "-p", "8000", \
     "--websocket_timeout", "3600", \
     "avatar_system.asgi:application"]
