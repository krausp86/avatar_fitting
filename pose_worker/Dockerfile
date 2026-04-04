# ─────────────────────────────────────────────────────────────────────────────
# Pose Worker – all ML backends (MediaPipe, RTMPose, ViTPose)
# GPU build – requires NVIDIA runtime (nvidia-container-toolkit)
# Base: nvidia/cuda 13.0 – matches torch 2.11.0+cu130 (Blackwell sm_100)
# mmcv is built from source so CUDA ops (NMS etc.) are compiled correctly.
# human_body_prior: installed with --ignore-requires-python (metadata says >=3.11
#   but code is compatible with Python 3.10)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    PYOPENGL_PLATFORM=egl \
    DISPLAY= \
    # Build mmcv CUDA ops for all archs torch supports + PTX for forward compat
    # GB10 is compute 12.1 (sm_121); sm_120+PTX covers it via JIT compilation
    TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 11.0 12.0+PTX" \
    MMCV_WITH_OPS=1 \
    # FORCE_CUDA=1 bypasses torch.cuda.is_available() check during docker build
    # (no GPU during build, but nvcc is available to compile CUDA extensions)
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools \
    gcc g++ git wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgles2 libegl1 libgbm1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip via get-pip.py so `import pip` works as a Python module.
# (Ubuntu's apt pip3 wrapper is a shell script, not importable – breaks chumpy's setup.py.)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install "setuptools>=60,<67" wheel ninja

# ── Base Python packages ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# PyTorch with CUDA 13.0 – cu130 index has aarch64 CUDA wheels
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ── MMPose stack ──────────────────────────────────────────────────────────────
# xtcocotools has no ARM64 wheel and its source dist lacks pre-generated _mask.c.
# Fix: install pycocotools (which does ship _mask.c), create a fake xtcocotools
# package with dist-info so pip considers it satisfied, THEN install mmdet/mmpose
# (which would otherwise pull in xtcocotools as a dependency and fail).
#
# mmcv is built from source (MMCV_WITH_OPS=1) so CUDA NMS and other ops are
# compiled for the exact torch/CUDA version in this image.
RUN pip3 install "cython<3" "numpy==1.26.4" \
    && pip3 install --no-build-isolation pycocotools \
    && python3 -c "import site,os; sp=site.getsitepackages()[0]; xt=os.path.join(sp,'xtcocotools'); os.makedirs(xt,exist_ok=True); [open(os.path.join(xt,f),'w').write(b) for f,b in [('__init__.py','from pycocotools import *\n'),('coco.py','from pycocotools.coco import *\n'),('cocoeval.py','from pycocotools.cocoeval import *\n'),('mask.py','from pycocotools.mask import *\n')]]; di=os.path.join(sp,'xtcocotools-1.2.dist-info'); os.makedirs(di,exist_ok=True); open(os.path.join(di,'METADATA'),'w').write('Metadata-Version: 2.1\nName: xtcocotools\nVersion: 1.2\n'); open(os.path.join(di,'INSTALLER'),'w').write('pip\n'); open(os.path.join(di,'RECORD'),'w').write('')" \
    && pip3 install --no-build-isolation chumpy \
    && pip3 install --no-build-isolation "mmcv>=2.1.0" --no-binary mmcv \
    && pip3 install --no-build-isolation mmdet \
    && pip3 install --no-build-isolation mmpose \
    && pip3 install --no-build-isolation mmpretrain \
    && python3 -c "import re,site,os; f=os.path.join(site.getsitepackages()[0],'mmdet/__init__.py'); s=open(f).read(); open(f,'w').write(s.replace(\"mmcv_maximum_version = '2.2.0'\",\"mmcv_maximum_version = '2.3.0'\")); print('mmdet patched')" \
    && python3 -c "import site,os; f=os.path.join(site.getsitepackages()[0],'mmengine/runner/checkpoint.py'); s=open(f).read(); s=s.replace('torch.load(filename, map_location=map_location)','torch.load(filename, map_location=map_location, weights_only=False)'); open(f,'w').write(s); print('mmengine checkpoint patched')"

# ── SMPL-X + VPoser (für Fitting-Endpoints) ───────────────────────────────────
# human_body_prior metadata says >=3.11 but code is Py3.10-compatible
RUN pip3 install "smplx[all]>=0.1.28" \
    && pip3 install hatchling \
    && pip3 install --no-build-isolation --ignore-requires-python \
       git+https://github.com/nghorbani/human_body_prior \
    && pip3 install "numpy==1.26.4"

# ── 4D-Humans / HMR2 (expressive body init) ──────────────────────────────────
RUN pip3 install --no-build-isolation git+https://github.com/shubham-goel/4D-Humans \
    && pip3 install scipy \
    && pip3 install "numpy==1.26.4"

WORKDIR /app
COPY . .

EXPOSE 8001
CMD ["sh", "-c", "python3 /app/download_weights.py && uvicorn main:app --host 0.0.0.0 --port 8001"]
