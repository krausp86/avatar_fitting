"""Pose estimation microservice (FastAPI).

Endpoints:
  GET  /backends          – list available backends
  POST /analyze           – run pose estimation on a single frame
"""
import base64
import logging

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import ALL_BACKENDS, BACKEND_BY_ID, MediaPipeBackend, RTMPoseBackend, ViTPoseBackend, draw_combined

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

        return result

    except Exception as e:
        log.exception("analyze failed: backend=%s", req.backend)
        raise HTTPException(500, str(e))


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
        vit_lm = ViTPoseBackend.detect(frame)
        rtm_lm = RTMPoseBackend.detect(frame)

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
