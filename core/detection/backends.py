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


def _decode_mask(b64: str) -> np.ndarray:
    """Decode a lossless PNG binary mask (uint8 0/255) to float32 [0,1]."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.float32)


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
    async def async_analyze(cls, frame_bgr: np.ndarray,
                            include_segmentation: bool = False) -> Dict:
        import httpx
        payload = {
            "backend": cls.backend_id,
            "frame_b64": _encode_frame(frame_bgr),
            "include_render": True,
            "include_segmentation": include_segmentation,
        }
        log.debug("%s.async_analyze: POST %s/analyze", cls.backend_id, POSE_WORKER_URL)
        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                resp = await client.post(f"{POSE_WORKER_URL}/analyze", json=payload)
            log.debug("%s.async_analyze: status=%d", cls.backend_id, resp.status_code)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            log.error("%s.async_analyze: cannot connect to pose-worker – %s", cls.backend_id, e)
            raise
        except httpx.HTTPStatusError as e:
            log.error("%s.async_analyze: HTTP %d – %s", cls.backend_id, e.response.status_code, e.response.text[:300])
            raise
        except Exception as e:
            log.error("%s.async_analyze: %s: %s", cls.backend_id, type(e).__name__, e)
            raise
        data = resp.json()
        log.debug("%s.async_analyze: detection=%s landmark_count=%d",
                  cls.backend_id, data.get("detection"), data.get("landmark_count", 0))
        return {
            "landmarks":      data.get("landmarks", []),
            "detection":      data.get("detection", False),
            "landmark_count": data.get("landmark_count", 0),
            "render":         _decode_image(data["render_b64"]) if data.get("render_b64") else frame_bgr,
            "segmentation":   _decode_image(data["segmentation_b64"]) if data.get("segmentation_b64") else None,
            "mask":           _decode_mask(data["mask_b64"]) if data.get("mask_b64") else None,
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
    log.info("Backend availability: %s",
             {b.backend_id: b.available for b in ALL_BACKENDS})


async def async_refresh_availability():
    """Async version of refresh_availability."""
    global _BACKENDS_CACHE
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{POSE_WORKER_URL}/backends")
        resp.raise_for_status()
        _BACKENDS_CACHE = resp.json()["backends"]
    except Exception as e:
        log.warning("Could not reach pose-worker at %s: %s", POSE_WORKER_URL, e)
        _BACKENDS_CACHE = []
    info = {b['id']: b['available'] for b in _BACKENDS_CACHE}
    for backend_cls in ALL_BACKENDS:
        backend_cls.available = info.get(backend_cls.backend_id, False)
    log.info("Backend availability (async refresh): %s",
             {b.backend_id: b.available for b in ALL_BACKENDS})


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


# Circuit-breaker: set to True after first permanent failure so we stop retrying.
_smplx_permanently_unavailable = False


def smplx_regress(frame_bgr: np.ndarray) -> Optional[dict]:
    """
    Ruft SMPLer-X Regression auf dem pose-worker auf.
    Gibt SMPL-X Parameter zurück oder None wenn keine Person erkannt / nicht verfügbar.

    Returns dict mit Keys:
        beta (10), body_pose (63), global_orient (3), transl (3),
        expression (10), jaw_pose (3),
        left_hand_pose (45), right_hand_pose (45)
    """
    global _smplx_permanently_unavailable
    if _smplx_permanently_unavailable:
        return None

    import httpx
    payload = {"frame_b64": _encode_frame(frame_bgr)}
    try:
        resp = httpx.post(f"{POSE_WORKER_URL}/hmr2/regress", json=payload, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (500, 503):
            log.warning("smplx_regress: HMR2 permanently unavailable (%d) – disabling for this run. %s",
                        e.response.status_code, e.response.text[:200])
            _smplx_permanently_unavailable = True
            return None
        log.error("smplx_regress: HTTP %d – %s", e.response.status_code, e.response.text[:200])
        return None
    except Exception as e:
        log.warning("smplx_regress: request failed – %s", e)
        return None
    data = resp.json()
    if not data.get("detection"):
        return None
    return data.get("params")


async def async_smplx_regress(frame_bgr: np.ndarray) -> Optional[dict]:
    """Async version von smplx_regress."""
    if _smplx_permanently_unavailable:
        return None
    import httpx
    payload = {"frame_b64": _encode_frame(frame_bgr)}
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(f"{POSE_WORKER_URL}/smplx/regress", json=payload)
        if resp.status_code in (500, 503):
            log.warning("async_smplx_regress: SMPLer-X unavailable (%d)", resp.status_code)
            return None
        resp.raise_for_status()
    except Exception as e:
        log.warning("async_smplx_regress: %s: %s", type(e).__name__, e)
        return None
    data = resp.json()
    if not data.get("detection"):
        return None
    return data.get("params")


async def async_combined_analyze(frame_bgr) -> dict:
    """Async version – doesn't block the event loop while waiting for pose-worker."""
    import httpx
    payload = {"frame_b64": _encode_frame(frame_bgr)}
    log.debug("async_combined_analyze: POST %s/analyze/combined (frame %dx%d)",
              POSE_WORKER_URL, frame_bgr.shape[1], frame_bgr.shape[0])
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            resp = await client.post(f"{POSE_WORKER_URL}/analyze/combined", json=payload)
        log.debug("async_combined_analyze: status=%d", resp.status_code)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        log.error("async_combined_analyze: cannot connect to pose-worker at %s – %s", POSE_WORKER_URL, e)
        raise
    except httpx.HTTPStatusError as e:
        log.error("async_combined_analyze: HTTP %d from pose-worker: %s", e.response.status_code, e.response.text[:300])
        raise
    except Exception as e:
        log.error("async_combined_analyze: unexpected error – %s: %s", type(e).__name__, e)
        raise
    data = resp.json()
    log.debug("async_combined_analyze: detection=%s body_count=%d",
              data.get("detection"), data.get("body_count", 0))
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
