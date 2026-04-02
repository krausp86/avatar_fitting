"""
Download HMR2 weights to /data/models/hmr2 at container startup if not present.
Runs synchronously before uvicorn starts so the first request never waits.
"""
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

CACHE_DIR = '/data/models/hmr2'


def checkpoint_exists():
    return bool(glob.glob(f'{CACHE_DIR}/**/*.ckpt', recursive=True))


if checkpoint_exists():
    log.info("HMR2 weights already present in %s – skipping download", CACHE_DIR)
else:
    log.info("HMR2 weights not found – downloading to %s (this may take a while) …", CACHE_DIR)
    try:
        from hmr2.models import download_models
        from pathlib import Path
        download_models(CACHE_DIR)
        if checkpoint_exists():
            log.info("HMR2 weights downloaded successfully")
        else:
            log.error("HMR2 download completed but no .ckpt found – starting anyway")
    except Exception as e:
        log.error("HMR2 download failed: %s – starting anyway (HMR2 will be unavailable)", e)
