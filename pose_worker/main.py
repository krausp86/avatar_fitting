"""Pose estimation microservice (FastAPI).

Endpoints:
  GET  /backends          – list available backends
  POST /analyze           – run pose estimation on a single frame
"""
import base64
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import ALL_BACKENDS, BACKEND_BY_ID, MediaPipeBackend, RTMPoseBackend, ViTPoseBackend, SMPLerXBackend, HMR2Backend, draw_combined

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(name)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="Pose Worker")


class AnalyzeRequest(BaseModel):
    backend: str
    frame_b64: str                   # base64-encoded JPEG/PNG (BGR)
    include_render: bool = True
    include_segmentation: bool = False


class CombinedRequest(BaseModel):
    frame_b64: str                   # base64-encoded JPEG/PNG (BGR)


def _decode_frame(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame image")
    return frame


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _render_segmentation(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    seg = frame.copy()
    mask_bin = (mask > 0.5).astype(np.uint8)
    overlay = seg.copy()
    overlay[mask_bin == 1] = [0, 160, 80]
    seg = cv2.addWeighted(seg, 0.55, overlay, 0.45, 0)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 100), 2)
    return seg


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/backends")
def list_backends():
    return {"backends": [
        {"id": b.backend_id, "name": b.display_name, "available": b.available}
        for b in ALL_BACKENDS
    ]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    backend_cls = BACKEND_BY_ID.get(req.backend)
    if backend_cls is None:
        raise HTTPException(400, f"Unknown backend: {req.backend}")
    if not backend_cls.available:
        raise HTTPException(503, f"Backend not available: {req.backend}")

    try:
        frame = _decode_frame(req.frame_b64)
    except Exception as e:
        raise HTTPException(400, f"Frame decode error: {e}")

    try:
        landmarks = backend_cls.detect(frame)
        result = {
            "landmarks": landmarks,
            "detection": bool(landmarks),
            "landmark_count": len(landmarks),
        }

        if req.include_render:
            rendered = backend_cls.render(frame, landmarks)
            result["render_b64"] = _encode_frame(rendered)

        if req.include_segmentation and req.backend == "mediapipe":
            mask = MediaPipeBackend.segmentation_mask(frame)
            if mask is not None:
                result["segmentation_b64"] = _encode_frame(
                    _render_segmentation(frame, mask)
                )
                # Raw binary mask as lossless PNG (JPEG would corrupt the binary data)
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255
                _, buf = cv2.imencode('.png', mask_u8)
                result["mask_b64"] = base64.b64encode(buf).decode()

        return result

    except Exception as e:
        log.exception("analyze failed: backend=%s", req.backend)
        raise HTTPException(500, str(e))


class SMPLXRegressRequest(BaseModel):
    frame_b64: str    # base64-encoded JPEG/PNG (BGR)


@app.post("/smplx/regress")
def smplx_regress(req: SMPLXRegressRequest):
    """
    Führt SMPLer-X Regression durch und gibt SMPL-X Parameter zurück.
    Benötigt SMPLer-X Weights in /data/models/smpler_x/.
    """
    if not SMPLerXBackend.available:
        raise HTTPException(503, "SMPLer-X not available (missing mmpose or smplx package)")

    try:
        frame = _decode_frame(req.frame_b64)
    except Exception as e:
        raise HTTPException(400, f"Frame decode error: {e}")

    try:
        params = SMPLerXBackend.regress(frame)
    except Exception as e:
        log.exception("smplx_regress failed")
        # After a load failure _get_model sets available=False — return 503 so
        # callers treat this as "permanently unavailable" and stop retrying.
        if not SMPLerXBackend.available:
            raise HTTPException(503, f"SMPLer-X not available: {e}")
        raise HTTPException(500, str(e))

    if params is None:
        return {"detection": False, "params": None}

    return {"detection": True, "params": params}


@app.post("/hmr2/regress")
def hmr2_regress(req: SMPLXRegressRequest):
    """HMR2 (4D-Humans) body regression → SMPL-X compatible params."""
    if not HMR2Backend.available:
        raise HTTPException(503, "HMR2 not available (pip install 4D-Humans)")

    try:
        frame = _decode_frame(req.frame_b64)
    except Exception as e:
        raise HTTPException(400, f"Frame decode error: {e}")

    try:
        params = HMR2Backend.regress(frame)
    except Exception as e:
        log.exception("hmr2_regress failed")
        if not HMR2Backend.available:
            raise HTTPException(503, f"HMR2 not available: {e}")
        raise HTTPException(500, str(e))

    if params is None:
        return {"detection": False, "params": None}
    return {"detection": True, "params": params}


@app.post("/analyze/combined")
def analyze_combined(req: CombinedRequest):
    """Run ViTPose + RTMPose and return a combined body+face+hands visualization."""
    if not ViTPoseBackend.available or not RTMPoseBackend.available:
        raise HTTPException(503, "ViTPose and/or RTMPose not available")

    try:
        frame = _decode_frame(req.frame_b64)
    except Exception as e:
        raise HTTPException(400, f"Frame decode error: {e}")

    try:
        log.info("analyze_combined: frame %dx%d", frame.shape[1], frame.shape[0])
        t0 = time.perf_counter()
        vit_lm = ViTPoseBackend.detect(frame)
        rtm_lm = RTMPoseBackend.detect(frame)
        log.info("analyze_combined: done in %.0f ms  (vit=%d rtm=%d landmarks)",
                 (time.perf_counter() - t0) * 1000, len(vit_lm), len(rtm_lm))

        body_lm  = [lm for lm in vit_lm if lm['idx'] <= 16]
        face_lm  = [lm for lm in rtm_lm if 23 <= lm['idx'] <= 90]
        lhand_lm = [lm for lm in rtm_lm if 91 <= lm['idx'] <= 111]
        rhand_lm = [lm for lm in rtm_lm if 112 <= lm['idx'] <= 132]

        rendered = draw_combined(frame, vit_lm, rtm_lm)

        return {
            "detection": bool(body_lm),
            "body_count":  len(body_lm),
            "face_count":  len(face_lm),
            "lhand_count": len(lhand_lm),
            "rhand_count": len(rhand_lm),
            "body_landmarks":  body_lm,
            "face_landmarks":  face_lm,
            "lhand_landmarks": lhand_lm,
            "rhand_landmarks": rhand_lm,
            "render_b64": _encode_frame(rendered),
        }
    except Exception as e:
        log.exception("analyze_combined failed")
        raise HTTPException(500, str(e))
