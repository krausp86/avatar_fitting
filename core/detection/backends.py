"""
Pose estimation backends – HTTP client to pose-worker service.

The actual ML runs in the pose-worker container.  This module provides
the same interface that views.py expects, but delegates all computation
via HTTP.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

POSE_WORKER_URL = os.environ.get("POSE_WORKER_URL", "http://pose-worker:8001")

# Cached backend availability from pose-worker (refreshed on each /backends request)
_BACKENDS_CACHE: Optional[List[Dict]] = None


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


def _decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _fetch_backends() -> List[Dict]:
    """Fetch backend list from pose-worker. Returns [] on error."""
    try:
        import httpx
        resp = httpx.get(f"{POSE_WORKER_URL}/backends", timeout=5.0)
        resp.raise_for_status()
        return resp.json()["backends"]
    except Exception as e:
        log.warning("Could not reach pose-worker at %s: %s", POSE_WORKER_URL, e)
        return []


class PoseBackend:
    backend_id:   str = ''
    display_name: str = ''
    available:    bool = False

    @classmethod
    def analyze(cls, frame_bgr: np.ndarray,
                include_segmentation: bool = False) -> Dict:
        """
        Send frame to pose-worker and return combined result dict:
          landmarks       – list of landmark dicts
          detection       – bool
          landmark_count  – int
          render          – np.ndarray (annotated frame)
          segmentation    – np.ndarray or None (only for mediapipe)
        """
        import httpx
        payload = {
            "backend": cls.backend_id,
            "frame_b64": _encode_frame(frame_bgr),
            "include_render": True,
            "include_segmentation": include_segmentation,
        }
        resp = httpx.post(
            f"{POSE_WORKER_URL}/analyze", json=payload, timeout=3600.0
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "landmarks":      data.get("landmarks", []),
            "detection":      data.get("detection", False),
            "landmark_count": data.get("landmark_count", 0),
            "render":         _decode_image(data["render_b64"]) if data.get("render_b64") else frame_bgr,
            "segmentation":   _decode_image(data["segmentation_b64"]) if data.get("segmentation_b64") else None,
        }

    @classmethod
    def connections(cls) -> List[Tuple[int, int]]:
        return []


class MediaPipeBackend(PoseBackend):
    backend_id   = 'mediapipe'
    display_name = 'MediaPipe Heavy'


class RTMPoseBackend(PoseBackend):
    backend_id   = 'rtmpose'
    display_name = 'RTMPose-L Wholebody'


class ViTPoseBackend(PoseBackend):
    backend_id   = 'vitpose'
    display_name = 'ViTPose-H'


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_BACKENDS: List[type] = [MediaPipeBackend, RTMPoseBackend, ViTPoseBackend]
BACKEND_BY_ID: Dict[str, type] = {b.backend_id: b for b in ALL_BACKENDS}


def refresh_availability():
    """Re-fetch backend availability from pose-worker and update .available flags."""
    global _BACKENDS_CACHE
    _BACKENDS_CACHE = _fetch_backends()
    info = {b['id']: b['available'] for b in _BACKENDS_CACHE}
    for backend_cls in ALL_BACKENDS:
        backend_cls.available = info.get(backend_cls.backend_id, False)


def combined_analyze(frame_bgr) -> dict:
    """Call /analyze/combined on pose-worker. Returns combined result dict."""
    import httpx
    payload = {"frame_b64": _encode_frame(frame_bgr)}
    resp = httpx.post(f"{POSE_WORKER_URL}/analyze/combined", json=payload, timeout=3600.0)
    resp.raise_for_status()
    data = resp.json()
    return {
        "detection":       data.get("detection", False),
        "body_count":      data.get("body_count", 0),
        "face_count":      data.get("face_count", 0),
        "lhand_count":     data.get("lhand_count", 0),
        "rhand_count":     data.get("rhand_count", 0),
        "body_landmarks":  data.get("body_landmarks", []),
        "face_landmarks":  data.get("face_landmarks", []),
        "lhand_landmarks": data.get("lhand_landmarks", []),
        "rhand_landmarks": data.get("rhand_landmarks", []),
        "render":          _decode_image(data["render_b64"]) if data.get("render_b64") else frame_bgr,
    }


# Populate availability at import time (best-effort; pose-worker may not be up yet)
try:
    refresh_availability()
except Exception as _e:
    log.warning("pose-worker not reachable at startup: %s", _e)
