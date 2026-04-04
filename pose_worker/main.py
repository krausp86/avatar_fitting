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
from fit_router import router as fit_router

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(name)s %(message)s')
log = logging.getLogger(__name__)

app = FastAPI(title="Pose Worker")
app.include_router(fit_router, prefix="/fit", tags=["fitting"])


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


class DetectVideoRequest(BaseModel):
    video_path:       str
    sample_every:     int   = 3
    min_track_frames: int   = 10
    iou_threshold:    float = 0.3
    max_gap_frames:   int   = 30


def _lm_to_bbox(landmarks: list, margin: float = 0.1):
    xs = [lm['x'] for lm in landmarks]
    ys = [lm['y'] for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return (max(0., x1 - dx), max(0., y1 - dy),
            min(1., x2 + dx), min(1., y2 + dy))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    if inter == 0.:
        return 0.
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _match_track(bbox, active_tracks: list, threshold: float):
    best, best_score = None, threshold
    for t in active_tracks:
        if t['bboxes']:
            score = _iou(bbox, t['bboxes'][-1])
            if score > best_score:
                best, best_score = t, score
    return best


@app.post("/detect/video")
def detect_video(req: DetectVideoRequest):
    """
    GPU-basierte Personen-Erkennung auf einer Video-Datei.
    Priorität: ViTPose (GPU) → RTMPose (GPU) → MediaPipe (CPU-Fallback).
    Gibt eine Liste von Tracks zurück.
    """
    if ViTPoseBackend.available:
        active_backend = ViTPoseBackend
    elif RTMPoseBackend.available:
        log.warning("detect_video: ViTPose nicht verfügbar, weiche auf RTMPose aus")
        active_backend = RTMPoseBackend
    elif MediaPipeBackend.available:
        log.warning("detect_video: MMPose nicht verfügbar, weiche auf MediaPipe (CPU) aus")
        active_backend = MediaPipeBackend
    else:
        raise HTTPException(503, "Kein Pose-Backend verfügbar")

    import os
    if not os.path.exists(req.video_path):
        raise HTTPException(400, f"Video nicht gefunden: {req.video_path}")

    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise HTTPException(400, f"Video kann nicht geöffnet werden: {req.video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    active_tracks: list = []
    closed_tracks: list = []
    last_seen: dict     = {}
    next_id             = 0
    sampled_idx         = 0
    raw_idx             = 0

    log.info("detect_video: %s  %dx%d  %.1f fps", req.video_path, width, height, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_idx % req.sample_every != 0:
            raw_idx += 1
            continue

        # Stale Tracks schließen
        still_active = []
        for t in active_tracks:
            if sampled_idx - last_seen.get(id(t), 0) > req.max_gap_frames:
                closed_tracks.append(t)
            else:
                still_active.append(t)
        active_tracks = still_active

        try:
            landmarks = active_backend.detect(frame)
        except Exception as e:
            log.warning("detect_video: %s failed frame %d – %s",
                        active_backend.backend_id, raw_idx, e)
            raw_idx     += 1
            sampled_idx += 1
            continue

        # Nur Body-Landmarks (idx 0–16) für Tracking verwenden
        # Alle Backends liefern bereits normalisierte Koordinaten (0–1)
        body_lm = [lm for lm in landmarks if lm.get('idx', 0) <= 16]
        if body_lm:
            vis_scores = [lm.get('visibility', lm.get('score', 1.0)) for lm in body_lm]
            visibility = float(sum(vis_scores) / max(len(vis_scores), 1))
            bbox = _lm_to_bbox(body_lm, margin=0.1)

            track = _match_track(bbox, active_tracks, req.iou_threshold)
            if track is None:
                track = {
                    'track_id':          str(next_id),
                    'frame_start':       raw_idx,
                    'frame_end':         raw_idx,
                    'frames':            [],
                    'bboxes':            [],
                    'visibility_scores': [],
                    'best_visibility':   0.0,
                    'best_frame_idx':    None,
                    'best_frame_b64':    None,
                }
                next_id += 1
                active_tracks.append(track)

            track['frames'].append(raw_idx)
            track['bboxes'].append(bbox)
            track['visibility_scores'].append(visibility)
            track['frame_end'] = raw_idx
            last_seen[id(track)] = sampled_idx

            if visibility > track['best_visibility']:
                track['best_visibility'] = visibility
                track['best_frame_idx']  = raw_idx
                # Crop speichern
                x1 = int(bbox[0] * width);  x2 = int(bbox[2] * width)
                y1 = int(bbox[1] * height); y2 = int(bbox[3] * height)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    track['best_frame_b64'] = base64.b64encode(buf).decode()

        sampled_idx += 1
        raw_idx     += 1

    cap.release()

    result = []
    for t in (closed_tracks + active_tracks):
        if len(t['frames']) < req.min_track_frames:
            continue
        vis = t['visibility_scores']
        result.append({
            'track_id':       t['track_id'],
            'frame_start':    t['frame_start'],
            'frame_end':      t['frame_end'],
            'frame_count':    len(t['frames']),
            'mean_visibility': float(sum(vis) / max(len(vis), 1)),
            'bboxes':         t['bboxes'],
            'best_frame_idx': t['best_frame_idx'],
            'best_frame_b64': t['best_frame_b64'],
        })

    log.info("detect_video: %d Track(s) in %s (backend=%s)",
             len(result), req.video_path, active_backend.backend_id)
    return {"tracks": result, "backend": active_backend.backend_id}


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
