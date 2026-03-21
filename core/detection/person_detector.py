"""
Person detection and tracking using MediaPipe Pose Landmarker (Tasks API, >=0.10).
"""
from __future__ import annotations

import io
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "/tmp/pose_landmarker_lite.task"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        log.info("Downloading MediaPipe pose model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("Model downloaded to %s", MODEL_PATH)


@dataclass
class PersonTrack:
    track_id: str
    frame_start: int
    frame_end: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[tuple] = field(default_factory=list)
    visibility_scores: List[float] = field(default_factory=list)
    best_visibility: float = 0.0
    best_frame_idx: Optional[int] = None
    best_frame_crop: Optional[np.ndarray] = None
    mask_path: Optional[str] = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def mean_visibility(self) -> float:
        return float(np.mean(self.visibility_scores)) if self.visibility_scores else 0.0


def detect_persons_in_video(
    video_path: str,
    mask_output_dir: Optional[str] = None,
    sample_every: int = 3,
    min_track_frames: int = 10,
    iou_threshold: float = 0.3,
    max_gap_frames: int = 30,
) -> List[PersonTrack]:
    try:
        import mediapipe as mp
    except ImportError:
        log.warning("mediapipe not installed – skipping detection")
        return []

    _ensure_model()

    BaseOptions          = mp.tasks.BaseOptions
    PoseLandmarker       = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode    = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=True,
        num_poses=1,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    active_tracks: List[PersonTrack] = []
    closed_tracks: List[PersonTrack] = []
    mask_buf: Dict[int, Dict[int, np.ndarray]] = {}
    last_seen: Dict[int, int] = {}
    next_id   = 0
    sampled_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        raw_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % sample_every != 0:
                raw_idx += 1
                continue

            # Close stale tracks
            still_active = []
            for t in active_tracks:
                if sampled_idx - last_seen.get(id(t), 0) > max_gap_frames:
                    closed_tracks.append(t)
                else:
                    still_active.append(t)
            active_tracks = still_active

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(raw_idx / fps * 1000)
            results   = landmarker.detect_for_video(mp_image, timestamp)

            if results.pose_landmarks:
                landmarks  = results.pose_landmarks[0]
                bbox       = _landmarks_to_bbox(landmarks, margin=0.1)
                visibility = _mean_visibility(landmarks)

                mask = None
                if results.segmentation_masks:
                    raw_mask = results.segmentation_masks[0].numpy_view()
                    mask = (raw_mask > 0.5).astype(np.uint8)

                track = _match_to_track(bbox, active_tracks, iou_threshold)
                if track is None:
                    track = PersonTrack(
                        track_id=str(next_id),
                        frame_start=raw_idx,
                        frame_end=raw_idx,
                    )
                    next_id += 1
                    mask_buf[id(track)] = {}
                    active_tracks.append(track)

                track.frames.append(raw_idx)
                track.bboxes.append(bbox)
                track.visibility_scores.append(visibility)
                track.frame_end = raw_idx
                last_seen[id(track)] = sampled_idx

                if mask is not None:
                    mask_buf[id(track)][raw_idx] = mask

                if visibility > track.best_visibility:
                    track.best_visibility = visibility
                    track.best_frame_idx  = raw_idx
                    track.best_frame_crop = _crop_bbox(frame, bbox, width, height)

            sampled_idx += 1
            raw_idx     += 1

    cap.release()

    result = []
    for t in (closed_tracks + active_tracks):
        if t.frame_count < min_track_frames:
            continue
        masks = mask_buf.get(id(t), {})
        if mask_output_dir and masks:
            os.makedirs(mask_output_dir, exist_ok=True)
            mask_path = os.path.join(mask_output_dir, f"track_{t.track_id}_masks.npz")
            np.savez_compressed(
                mask_path,
                frame_indices=np.array(sorted(masks.keys()), dtype=np.int32),
                masks=np.stack([masks[k] for k in sorted(masks.keys())]),
            )
            t.mask_path = mask_path
        result.append(t)

    log.info("detect_persons_in_video: %d track(s) in %s", len(result), os.path.basename(video_path))
    return result


def _landmarks_to_bbox(landmarks, margin=0.1):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return (max(0., x1-dx), max(0., y1-dy), min(1., x2+dx), min(1., y2+dy))


def _mean_visibility(landmarks) -> float:
    scores = [lm.visibility for lm in landmarks if lm.visibility is not None]
    return float(np.mean(scores)) if scores else 0.0


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    if inter == 0.:
        return 0.
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _match_to_track(bbox, active_tracks, threshold):
    best, best_score = None, threshold
    for t in active_tracks:
        if t.bboxes:
            score = _iou(bbox, t.bboxes[-1])
            if score > best_score:
                best, best_score = t, score
    return best


def _crop_bbox(frame, bbox, width, height):
    x1, y1 = int(bbox[0]*width), int(bbox[1]*height)
    x2, y2 = int(bbox[2]*width), int(bbox[3]*height)
    return frame[y1:y2, x1:x2].copy()
