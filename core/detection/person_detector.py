"""
Person detection and tracking for video files.

Uses MediaPipe Pose (single-person, best-in-frame) to detect persons
frame-by-frame and groups detections into tracks using bounding-box IoU.

Produces per track:
  - frame_start, frame_end, frame_count, visibility (for DetectedPerson)
  - best-frame thumbnail crop (BGR, for DB thumbnail)
  - binary segmentation masks saved as .npz (for Stage 1 silhouette loss)

Multi-person note:
  MediaPipe Pose detects only the most prominent person per frame.
  For videos with multiple subjects, use the Merge UI in the app to
  combine tracks that belong to the same identity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PersonTrack:
    track_id: str
    frame_start: int
    frame_end: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[tuple] = field(default_factory=list)      # (x1,y1,x2,y2) normalised [0,1]
    visibility_scores: List[float] = field(default_factory=list)
    best_visibility: float = 0.0
    best_frame_idx: Optional[int] = None                   # frame number with highest visibility
    best_frame_crop: Optional[np.ndarray] = None           # BGR crop of that frame
    mask_path: Optional[str] = None                        # path to .npz with all masks

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def mean_visibility(self) -> float:
        return float(np.mean(self.visibility_scores)) if self.visibility_scores else 0.0


# ─── Main entry point ─────────────────────────────────────────────────────────

def detect_persons_in_video(
    video_path: str,
    mask_output_dir: Optional[str] = None,
    sample_every: int = 3,
    min_track_frames: int = 10,
    iou_threshold: float = 0.3,
    max_gap_frames: int = 30,
) -> List[PersonTrack]:
    """
    Analyse a video and return a list of person tracks.

    Args:
        video_path:       Path to video file.
        mask_output_dir:  Directory for segmentation mask .npz files.
                          Pass None to skip saving masks.
        sample_every:     Process every Nth frame (3 → ~10 fps for 30 fps video).
        min_track_frames: Discard tracks shorter than this many sampled frames.
        iou_threshold:    Minimum bounding-box IoU to link a detection to an
                          existing track rather than starting a new one.
        max_gap_frames:   Close a track when it has not been updated for this
                          many *sampled* frames.

    Returns:
        List of PersonTrack objects (only tracks >= min_track_frames).
    """
    try:
        import mediapipe as mp
    except ImportError:
        log.warning("mediapipe is not installed – person detection skipped. "
                    "Install with: pip install mediapipe>=0.10.9")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # active_tracks: tracks still accepting new detections
    # closed_tracks: tracks that timed out or video ended
    active_tracks: List[PersonTrack] = []
    closed_tracks: List[PersonTrack] = []
    next_id = 0

    # Mask buffer: id(track) → {frame_idx: binary mask (H,W uint8)}
    mask_buf: Dict[int, Dict[int, np.ndarray]] = {}

    # last_seen[id(track)] = sampled frame index of last detection
    last_seen: Dict[int, int] = {}

    sampled_idx = 0   # counts only sampled frames (for gap detection)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        raw_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % sample_every != 0:
                raw_idx += 1
                continue

            # Close tracks that have gone stale
            still_active = []
            for t in active_tracks:
                if sampled_idx - last_seen[id(t)] > max_gap_frames:
                    closed_tracks.append(t)
                else:
                    still_active.append(t)
            active_tracks = still_active

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                bbox       = _landmarks_to_bbox(results.pose_landmarks, margin=0.1)
                visibility = _mean_visibility(results.pose_landmarks)
                mask       = (
                    (results.segmentation_mask > 0.5).astype(np.uint8)
                    if results.segmentation_mask is not None
                    else None
                )

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
                    track.best_visibility  = visibility
                    track.best_frame_idx   = raw_idx
                    track.best_frame_crop  = _crop_bbox(frame, bbox, width, height)

            sampled_idx += 1
            raw_idx     += 1

    cap.release()

    # Collect all tracks
    all_tracks = closed_tracks + active_tracks

    result = []
    for t in all_tracks:
        if t.frame_count < min_track_frames:
            continue

        # Save masks if requested
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

    log.info("detect_persons_in_video: %d track(s) found in %s",
             len(result), os.path.basename(video_path))
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _landmarks_to_bbox(landmarks, margin: float = 0.1) -> tuple:
    """Convert MediaPipe pose landmarks to (x1,y1,x2,y2) normalised [0,1]."""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(1.0, x2 + dx),
        min(1.0, y2 + dy),
    )


def _mean_visibility(landmarks) -> float:
    """Average landmark visibility score across all pose landmarks."""
    scores = [lm.visibility for lm in landmarks.landmark
              if hasattr(lm, 'visibility') and lm.visibility is not None]
    return float(np.mean(scores)) if scores else 0.0


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union of two normalised (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _match_to_track(
    bbox: tuple,
    active_tracks: List[PersonTrack],
    threshold: float,
) -> Optional[PersonTrack]:
    """Return the active track whose last bbox has the highest IoU with bbox,
    or None if all tracks score below threshold."""
    best, best_score = None, threshold
    for t in active_tracks:
        if t.bboxes:
            score = _iou(bbox, t.bboxes[-1])
            if score > best_score:
                best, best_score = t, score
    return best


def _crop_bbox(frame: np.ndarray, bbox: tuple, width: int, height: int) -> np.ndarray:
    """Return a BGR crop for the normalised bbox."""
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    return frame[y1:y2, x1:x2].copy()
