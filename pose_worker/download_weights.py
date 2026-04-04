"""
Download / pre-warm all model weights at container startup.
Runs synchronously before uvicorn starts so the first request never waits.

Backends covered:
  - HMR2        → hmr2.models.download_models
  - RTMPose-L   → mim download (if not already in /data/models)
  - ViTPose-H   → mim download (if not already in /data/models)
  - MediaPipe   → urllib download (small .task file)
"""
import glob
import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODELS_DIR = '/data/models'

# ── HMR2 ──────────────────────────────────────────────────────────────────────

def _hmr2_ready():
    return bool(glob.glob(f'{MODELS_DIR}/hmr2/**/*.ckpt', recursive=True))

if _hmr2_ready():
    log.info("HMR2 weights already present – skipping download")
else:
    log.info("HMR2 weights not found – downloading to %s …", MODELS_DIR)
    try:
        from hmr2.models import download_models
        download_models(MODELS_DIR)
        log.info("HMR2 weights ready") if _hmr2_ready() else log.error("HMR2 download finished but no .ckpt found")
    except Exception as e:
        log.error("HMR2 download failed: %s – HMR2 will be unavailable", e)


# ── MMPose weights (RTMPose + ViTPose) ────────────────────────────────────────

MMPOSE_WEIGHTS = [
    {
        'name':   'RTMPose-L Wholebody',
        'file':   'rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth',
        'config': 'rtmpose-l_8xb32-270e_coco-wholebody-384x288',
    },
    {
        'name':   'ViTPose-H',
        'file':   'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
        'config': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192',
    },
]

for w in MMPOSE_WEIGHTS:
    dest = os.path.join(MODELS_DIR, w['file'])
    if os.path.exists(dest):
        log.info("%s weights already present at %s", w['name'], dest)
        continue
    log.info("%s weights not found – downloading via mim …", w['name'])
    try:
        from mim import download as mim_download
        mim_download('mmpose', configs=[w['config']], dest_root=MODELS_DIR)
        log.info("%s weights ready", w['name']) if os.path.exists(dest) else log.warning(
            "%s: mim download done but expected file missing at %s", w['name'], dest)
    except Exception as e:
        log.error("%s mim download failed: %s – will download on first request", w['name'], e)


# ── MediaPipe pose model ───────────────────────────────────────────────────────

MP_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MP_PATH = os.path.join(MODELS_DIR, 'pose_landmarker_heavy.task')

if os.path.exists(MP_PATH):
    log.info("MediaPipe model already present at %s", MP_PATH)
else:
    log.info("MediaPipe model not found – downloading …")
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(MP_URL, MP_PATH)
        log.info("MediaPipe model ready at %s", MP_PATH)
    except Exception as e:
        log.error("MediaPipe download failed: %s", e)

log.info("download_weights.py complete – starting uvicorn")
