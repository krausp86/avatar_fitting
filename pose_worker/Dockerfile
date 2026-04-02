# ─────────────────────────────────────────────────────────────────────────────
# Pose Worker – all ML backends (MediaPipe, RTMPose, ViTPose)
# CPU-only build (falls back to CPU automatically if no GPU)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEDIAPIPE_DISABLE_GPU=1 \
    HOME=/tmp \
    PYOPENGL_PLATFORM=egl \
    DISPLAY=

# gcc required to compile xtcocotools from source (numpy ABI fix)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgles2 libegl1 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# ── Base Python packages ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── PyTorch CPU (2.1.0 – matches mmcv 2.1.0 wheel availability) ──────────────
RUN pip install \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── MMPose stack ──────────────────────────────────────────────────────────────
# torch 2.1.0 is used here so that openmim can find pre-built mmcv 2.1.0 wheels.
# mmdet requires mmcv>=2.0.0rc4,<2.2.0 – mmcv 2.2.0 breaks the assertion.
# chumpy: broken setup.py calls `import pip` → --no-build-isolation
# xtcocotools: PyPI wheel compiled against numpy 2.x, we use 1.26.4 → build from source
#   reinstalled AFTER mmpose because mmpose overwrites it with the incompatible PyPI wheel
# setuptools reinstalled last to ensure pkg_resources is importable at runtime
RUN pip install --no-build-isolation chumpy \
    && pip install openmim \
    && python -m mim install "mmcv==2.1.0" \
    && python -m mim install mmdet \
    && python -m mim install mmpose \
    && python -m mim install mmpretrain \
    && pip install cython \
    && pip install --no-binary xtcocotools --no-build-isolation --force-reinstall xtcocotools \
    && pip install "setuptools>=65,<71" \
    && pip install "numpy==1.26.4"

# ── SMPL-X package (für SMPLer-X Backend) ─────────────────────────────────────
RUN pip install "smplx[all]>=0.1.28"

# ── 4D-Humans / HMR2 (expressive body init) ──────────────────────────────────
# Gibt SMPL-Parameter zurück; body_pose wird auf SMPL-X (21 Gelenke) gemappt.
# Weights werden beim ersten Aufruf automatisch von HuggingFace geladen.
RUN pip install --no-build-isolation git+https://github.com/shubham-goel/4D-Humans \
    && pip install scipy \
    && pip install "numpy==1.26.4"

WORKDIR /app
COPY . .

COPY download_weights.py /app/download_weights.py

EXPOSE 8001
CMD ["sh", "-c", "python /app/download_weights.py && uvicorn main:app --host 0.0.0.0 --port 8001"]
