"""
Filesystem video scanner.
Walks VIDEO_SCAN_ROOT and registers new MP4/MOV/AVI files in the database,
then runs person detection to populate DetectedPerson records.
"""
import io
import json
import logging
import os
import subprocess
from pathlib import Path

import cv2

from .models import DetectedPerson, VideoSource

log = logging.getLogger(__name__)


def _ffprobe_metadata(path: str) -> dict:
    """Return {fps, frame_count, width, height} via ffprobe, or empty dict on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return {}
        info = json.loads(result.stdout)
        stream = (info.get("streams") or [{}])[0]
        width  = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        # fps: "30000/1001" or "30/1"
        fps_raw = stream.get("r_frame_rate", "0/1")
        num, den = (int(x) for x in fps_raw.split("/"))
        fps = num / den if den else 0.0
        nb_frames = stream.get("nb_frames")
        if nb_frames:
            frame_count = int(nb_frames)
        else:
            # Fallback: duration * fps
            dur = float(stream.get("duration", 0) or 0)
            frame_count = int(dur * fps) if fps else 0
        return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
    except Exception:
        log.debug("ffprobe failed for %s", path, exc_info=True)
        return {}

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def scan_video_folder(root: str) -> int:
    """Scan folder recursively, add new videos to DB. Returns count of new entries."""
    root  = Path(root)
    added = 0

    for dirpath, _, filenames in os.walk(root):
        folder = str(Path(dirpath).relative_to(root))
        for fname in filenames:
            if Path(fname).suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            full_path = str(Path(dirpath) / fname)
            if VideoSource.objects.filter(path=full_path).exists():
                continue

            vs = VideoSource.objects.create(
                path     = full_path,
                filename = fname,
                folder   = folder or '.',
            )
            _populate_metadata(vs)
            added += 1

    return added


def _populate_metadata(vs: VideoSource):
    """Read duration/fps/resolution via ffprobe (fallback: OpenCV) and save a video thumbnail."""
    try:
        cap = cv2.VideoCapture(vs.path)
        if not cap.isOpened():
            return

        # Prefer ffprobe — more reliable across codecs/platforms
        meta = _ffprobe_metadata(vs.path)
        if meta and meta["fps"] > 0:
            fps         = meta["fps"]
            frame_count = meta["frame_count"]
            w           = meta["width"]
            h           = meta["height"]
        else:
            fps        = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vs.fps        = fps
        vs.duration_s = frame_count / fps if fps > 0 else None
        vs.resolution = f"{w}x{h}"

        # Grab a frame from 10 % into the video as thumbnail
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.1))
            ret, frame = cap.read()
            if ret:
                vs.thumbnail.save(
                    f"video_{vs.id}.jpg",
                    _bgr_to_jpeg_file(frame, max_size=480),
                    save=False,
                )

        cap.release()
        vs.save()

    except Exception:
        log.exception("_populate_metadata failed for %s", vs.path)


def detect_persons_for_video(vs: VideoSource) -> int:
    """
    Run GPU-based person detection on *vs* via the pose-worker and create
    DetectedPerson records.

    Safe to call multiple times – existing tracks for this video are
    deleted first so results stay consistent with the current detector.

    Returns the number of new DetectedPerson records created.
    """
    import base64
    import httpx
    import numpy as np

    POSE_WORKER_URL = os.environ.get("POSE_WORKER_URL", "http://pose-worker:8001")

    # Remove stale detections from a previous run
    DetectedPerson.objects.filter(video=vs).delete()

    log.info("detect_persons_for_video: calling pose-worker for %s", vs.path)
    try:
        resp = httpx.post(
            f"{POSE_WORKER_URL}/detect/video",
            json={"video_path": vs.path},
            timeout=3600.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            # ViTPose not available – fall back to local MediaPipe CPU
            log.warning("pose-worker /detect/video unavailable (503), falling back to local MediaPipe")
            return _detect_persons_local(vs)
        raise
    except Exception as e:
        log.warning("pose-worker /detect/video failed (%s), falling back to local MediaPipe", e)
        return _detect_persons_local(vs)

    tracks = resp.json().get("tracks", [])
    created = 0
    for t in tracks:
        dp = DetectedPerson(
            video       = vs,
            track_id    = t['track_id'],
            frame_start = t['frame_start'],
            frame_end   = t['frame_end'],
            frame_count = t['frame_count'],
            visibility  = t['mean_visibility'],
            meta        = {
                'bboxes':    t['bboxes'][::10],
                'mask_path': None,
            },
        )

        best_b64 = t.get('best_frame_b64')
        if best_b64:
            data = base64.b64decode(best_b64)
            arr  = np.frombuffer(data, np.uint8)
            crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if crop is not None and crop.size > 0:
                dp.thumbnail.save(
                    f"person_{vs.id}_{t['track_id']}.jpg",
                    _bgr_to_jpeg_file(crop, max_size=256),
                    save=False,
                )

        dp.save()
        created += 1

    log.info("detect_persons_for_video: %d track(s) created for video %s", created, vs.id)
    return created


def _detect_persons_local(vs: VideoSource) -> int:
    """Fallback: lokale MediaPipe CPU-Detection (falls pose-worker nicht verfügbar)."""
    from django.conf import settings
    from .detection.person_detector import detect_persons_in_video

    mask_dir = os.path.join(
        getattr(settings, 'MEDIA_ROOT', 'media'),
        'masks',
        str(vs.id),
    )
    tracks  = detect_persons_in_video(video_path=vs.path, mask_output_dir=mask_dir)
    created = 0
    for t in tracks:
        dp = DetectedPerson(
            video       = vs,
            track_id    = t.track_id,
            frame_start = t.frame_start,
            frame_end   = t.frame_end,
            frame_count = t.frame_count,
            visibility  = t.mean_visibility,
            meta        = {'bboxes': t.bboxes[::10], 'mask_path': t.mask_path},
        )
        if t.best_frame_crop is not None and t.best_frame_crop.size > 0:
            dp.thumbnail.save(
                f"person_{vs.id}_{t.track_id}.jpg",
                _bgr_to_jpeg_file(t.best_frame_crop, max_size=256),
                save=False,
            )
        dp.save()
        created += 1
    return created

    log.info("detect_persons_for_video: created %d DetectedPerson(s) for %s",
             created, vs.filename)
    return created


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _bgr_to_jpeg_file(bgr_image, max_size: int = 256):
    """Convert a BGR numpy array to a Django-compatible JPEG ContentFile."""
    from django.core.files.base import ContentFile
    from PIL import Image

    h, w = bgr_image.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        bgr_image = cv2.resize(
            bgr_image, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return ContentFile(buf.read())
