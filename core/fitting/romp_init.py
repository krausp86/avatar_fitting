"""
ROMP initialiser for Stage 1.

Runs ROMP over a set of frames and returns median β (shape) and per-frame θ
(pose) as numpy arrays.  Everything is wrapped in a try/except so that missing
weights or a missing `romp` package degrade gracefully to None.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def romp_init_frames(
    frames: list,                    # list of np.ndarray (H, W, 3) BGR uint8
    model_path: Optional[str] = None,
    smpl_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Run ROMP over *frames* to get a shape/pose initialisation.

    Returns
    -------
    dict with keys:
        'beta'   : np.ndarray (10,)   — median β over all frames
        'thetas' : np.ndarray (N, 72) — per-frame [global_orient(3) | body_pose(69)]
    or None if ROMP is unavailable / model_path missing / inference fails.
    """
    if not frames:
        return None

    if not model_path or not os.path.isfile(model_path):
        log.info("ROMP init skipped: model not found at %s", model_path)
        return None

    try:
        import romp
    except ImportError:
        log.info("ROMP init skipped: `romp` package not installed")
        return None

    try:
        import cv2

        # romp_settings(input_args=sys.argv[1:]) captures sys.argv at import time,
        # so patching sys.argv later has no effect.  Pass input_args=[] explicitly
        # to get clean defaults, then override our paths directly on the namespace.
        settings = romp.romp_settings(input_args=[])
        settings.model_path = model_path
        if smpl_path and os.path.isfile(smpl_path):
            settings.smpl_path = smpl_path
        settings.mode = "video"
        settings.show = False
        settings.render_mesh = False
        settings.show_largest = False
        settings.save_path = None
        settings.GPU = 0 if _cuda_available() else -1

        model = romp.ROMP(settings)

        betas_list:  list[np.ndarray] = []
        thetas_list: list[np.ndarray] = []

        for frame in frames:
            # ROMP V2 expects BGR (same convention as OpenCV)
            result = model(frame)
            if result is None:
                continue
            # Keys: 'smpl_thetas' (M, 72), 'smpl_betas' (M, 10), 'cam' (M, 3)
            smpl_thetas = result.get("smpl_thetas")   # (M, 72)
            betas       = result.get("smpl_betas")    # (M, 10)
            if smpl_thetas is None or betas is None or len(smpl_thetas) == 0:
                continue
            # Take the first detected person (index 0 = highest confidence in ROMP)
            thetas_list.append(np.array(smpl_thetas[0], dtype=np.float32))
            betas_list.append(np.array(betas[0],        dtype=np.float32))

        if not thetas_list:
            log.warning("ROMP init: no detections in any frame")
            return None

        beta_median  = np.median(np.stack(betas_list,  axis=0), axis=0)   # (10,)
        thetas_array = np.stack(thetas_list, axis=0)                       # (N_det, 72)

        # Pad or trim to match len(frames) so callers can index by frame index.
        # If ROMP missed some frames, repeat the last valid theta for those gaps.
        if len(thetas_array) < len(frames):
            pad = np.tile(thetas_array[-1:], (len(frames) - len(thetas_array), 1))
            thetas_array = np.concatenate([thetas_array, pad], axis=0)
        else:
            thetas_array = thetas_array[:len(frames)]

        log.info(
            "ROMP init: %d/%d frames detected, beta[0]=%.3f",
            len(thetas_list), len(frames), float(beta_median[0]),
        )
        return {"beta": beta_median, "thetas": thetas_array}

    except Exception as exc:
        log.warning("ROMP init failed (%s) — falling back to zero init", exc)
        return None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
