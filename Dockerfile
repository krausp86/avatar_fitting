# ─────────────────────────────────────────────────────────────────────────────
# SOMA Avatar System – Dockerfile
#
# Der Web-Container macht kein ML mehr — alles GPU/Pose läuft im pose-worker.
# Zwei einfache Layers:
#   Layer 1 (base):  Ubuntu 22.04 + Python + System-Pakete + pip-Deps
#   Layer 2 (app):   Application code
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04 AS base

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
    curl \
    gosu \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install --upgrade setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt


# ── App (final) ───────────────────────────────────────────────────────────────
FROM base AS app

RUN groupadd -r soma && useradd -r -g soma -d /app -s /sbin/nologin soma

WORKDIR /app

COPY --chown=soma:soma . .

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

RUN python3 manage.py collectstatic --noinput 2>/dev/null || true

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
