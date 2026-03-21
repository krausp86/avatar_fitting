"""
Filesystem video scanner.
Walks VIDEO_SCAN_ROOT and registers new MP4/MOV/AVI files in the database,
then runs person detection to populate DetectedPerson records.
"""
import io
import logging
import os
from pathlib import Path

import cv2

from .models import DetectedPerson, VideoSource

log = logging.getLogger(__name__)

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
    """Read duration/fps/resolution via OpenCV and save a video thumbnail."""
    try:
        cap = cv2.VideoCapture(vs.path)
        if not cap.isOpened():
            return

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
    Run person detection on *vs* and create DetectedPerson records.

    Safe to call multiple times – existing tracks for this video are
    deleted first so results stay consistent with the current detector.

    Returns the number of new DetectedPerson records created.
    """
    from django.conf import settings
    from .detection.person_detector import detect_persons_in_video

    # Remove stale detections from a previous run
    DetectedPerson.objects.filter(video=vs).delete()

    mask_dir = os.path.join(
        getattr(settings, 'MEDIA_ROOT', 'media'),
        'masks',
        str(vs.id),
    )

    tracks = detect_persons_in_video(
        video_path=vs.path,
        mask_output_dir=mask_dir,
    )

    created = 0
    for t in tracks:
        dp = DetectedPerson(
            video       = vs,
            track_id    = t.track_id,
            frame_start = t.frame_start,
            frame_end   = t.frame_end,
            frame_count = t.frame_count,
            visibility  = t.mean_visibility,
            meta        = {
                # Store a sparse sample of bounding boxes (every 10th detection)
                'bboxes':    t.bboxes[::10],
                'mask_path': t.mask_path,
            },
        )

        if t.best_frame_crop is not None and t.best_frame_crop.size > 0:
            dp.thumbnail.save(
                f"person_{vs.id}_{t.track_id}.jpg",
                _bgr_to_jpeg_file(t.best_frame_crop, max_size=256),
                save=False,
            )

        dp.save()
        created += 1

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
