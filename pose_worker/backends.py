"""
Pose estimation backends – ML implementations (MediaPipe, RTMPose, ViTPose).
Runs inside pose-worker container.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from joint_definitions import (
    COCO17_CONNECTIONS,
    COCO17_JOINT_NAMES,
    COCO_WHOLEBODY_CONNECTIONS,
    COCO_WHOLEBODY_JOINT_NAMES,
    FACE_68_CONNECTIONS,
    LHAND_CONNECTIONS,
    RHAND_CONNECTIONS,
    MEDIAPIPE_CONNECTIONS,
    MEDIAPIPE_JOINT_NAMES,
)

log = logging.getLogger(__name__)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = "/tmp/pose_landmarker_heavy.task"

_MODEL_CACHE: Dict[str, Any] = {}


def _ensure_mediapipe_model():
    import urllib.request
    if not os.path.exists(MODEL_PATH):
        log.info("Downloading MediaPipe pose model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("Model downloaded to %s", MODEL_PATH)


def _draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks: List[Dict],
    connections: List[Tuple[int, int]],
) -> np.ndarray:
    vis_img = frame_bgr.copy()
    h, w = vis_img.shape[:2]
    idx_map = {lm['idx']: lm for lm in landmarks}

    for a, b in connections:
        la, lb = idx_map.get(a), idx_map.get(b)
        if la is None or lb is None:
            continue
        if la['visibility'] > 0.2 and lb['visibility'] > 0.2:
            pa = (int(max(0, min(1, la['x'])) * w), int(max(0, min(1, la['y'])) * h))
            pb = (int(max(0, min(1, lb['x'])) * w), int(max(0, min(1, lb['y'])) * h))
            cv2.line(vis_img, pa, pb, (80, 200, 255), 2)

    for lm in landmarks:
        vis = lm['visibility']
        oob = lm['out_of_bounds']
        px = int(max(0, min(1, lm['x'])) * w)
        py = int(max(0, min(1, lm['y'])) * h)
        color = (0, 140, 255) if oob else (0, int(255 * vis), int(255 * (1 - vis)))
        cv2.circle(vis_img, (px, py), 6, color, -1)
        cv2.circle(vis_img, (px, py), 6, (220, 220, 220), 1)
        if oob:
            cv2.drawMarker(vis_img, (px, py), (0, 140, 255), cv2.MARKER_CROSS, 14, 2)

    return vis_img


def _make_landmark(idx: int, name: str, x: float, y: float,
                   z: float, vis: float) -> Dict:
    return {
        'idx': idx,
        'name': name,
        'x': round(float(x), 4),
        'y': round(float(y), 4),
        'z': round(float(z), 4),
        'visibility': round(float(vis), 3),
        'out_of_bounds': not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0),
    }


class PoseBackend(ABC):
    backend_id:   str = ''
    display_name: str = ''
    available:    bool = False

    @classmethod
    def _get_model(cls):
        if cls.backend_id not in _MODEL_CACHE:
            log.info("Loading backend model: %s", cls.backend_id)
            _MODEL_CACHE[cls.backend_id] = cls._load_model()
            log.info("Backend model ready: %s", cls.backend_id)
        return _MODEL_CACHE[cls.backend_id]

    @classmethod
    @abstractmethod
    def _load_model(cls): ...

    @classmethod
    @abstractmethod
    def detect(cls, frame_bgr: np.ndarray) -> List[Dict]: ...

    @classmethod
    def connections(cls) -> List[Tuple[int, int]]:
        return []

    @classmethod
    def render(cls, frame_bgr: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        return _draw_skeleton(frame_bgr, landmarks, cls.connections())

    @classmethod
    def segmentation_mask(cls, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        return None


# ── MediaPipe Heavy ───────────────────────────────────────────────────────────

class MediaPipeBackend(PoseBackend):
    backend_id   = 'mediapipe'
    display_name = 'MediaPipe Heavy'

    @classmethod
    def _load_model(cls):
        import mediapipe as mp
        _ensure_mediapipe_model()
        return mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_segmentation_masks=True,
            num_poses=1,
        )

    @classmethod
    def _run(cls, frame_bgr: np.ndarray):
        import mediapipe as mp
        options = cls._get_model()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as lm:
            return lm.detect(mp_image)

    @classmethod
    def detect(cls, frame_bgr: np.ndarray) -> List[Dict]:
        result = cls._run(frame_bgr)
        if not result.pose_landmarks:
            return []
        landmarks = []
        for i, lm in enumerate(result.pose_landmarks[0]):
            vis = float(lm.visibility) if lm.visibility is not None else 0.0
            name = MEDIAPIPE_JOINT_NAMES[i] if i < len(MEDIAPIPE_JOINT_NAMES) else str(i)
            landmarks.append(_make_landmark(i, name, lm.x, lm.y,
                                            float(lm.z) if lm.z else 0.0, vis))
        return landmarks

    @classmethod
    def segmentation_mask(cls, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        result = cls._run(frame_bgr)
        if not result.segmentation_masks:
            return None
        mask = result.segmentation_masks[0].numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    @classmethod
    def connections(cls):
        return MEDIAPIPE_CONNECTIONS


# ── MMPose base (shared by RTMPose + ViTPose) ─────────────────────────────────

class _MMPoseBackend(PoseBackend):
    _mmpose_model_alias: str = ''
    _joint_names: List[str] = []

    @classmethod
    def _load_model(cls):
        from mmpose.apis import MMPoseInferencer
        from mmengine.registry import DefaultScope
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        log.info("MMPose using device: %s", device)
        # Some model configs (e.g. ViTPose td-hm) incorrectly default to mmdet scope.
        # Explicitly set mmpose scope before building.
        DefaultScope.get_instance('mmpose_default', scope_name='mmpose')
        return MMPoseInferencer(pose2d=cls._mmpose_model_alias, device=device)

    @classmethod
    def detect(cls, frame_bgr: np.ndarray) -> List[Dict]:
        inferencer = cls._get_model()
        h, w = frame_bgr.shape[:2]

        result_gen = inferencer(frame_bgr, return_vis=False, return_datasample=False)
        result = next(result_gen)

        predictions = result.get('predictions', [[]])
        if not predictions or not predictions[0]:
            return []

        pred = predictions[0][0]
        keypoints = pred.get('keypoints', [])
        scores    = pred.get('keypoint_scores', [])

        landmarks = []
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            x_norm = kp[0] / w
            y_norm = kp[1] / h
            name = cls._joint_names[i] if i < len(cls._joint_names) else f'kp_{i}'
            landmarks.append(_make_landmark(i, name, x_norm, y_norm, 0.0, float(score)))
        return landmarks


# ── RTMPose-L Wholebody ───────────────────────────────────────────────────────

class RTMPoseBackend(_MMPoseBackend):
    backend_id           = 'rtmpose'
    display_name         = 'RTMPose-L Wholebody'
    _mmpose_model_alias  = 'wholebody'
    _joint_names         = COCO_WHOLEBODY_JOINT_NAMES

    @classmethod
    def connections(cls):
        return COCO_WHOLEBODY_CONNECTIONS


# ── ViTPose-H ────────────────────────────────────────────────────────────────

class ViTPoseBackend(_MMPoseBackend):
    backend_id           = 'vitpose'
    display_name         = 'ViTPose-H'
    _mmpose_model_alias  = 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
    _joint_names         = COCO17_JOINT_NAMES

    @classmethod
    def connections(cls):
        return COCO17_CONNECTIONS


# ── Combined visualization ────────────────────────────────────────────────────

def draw_combined(frame_bgr: np.ndarray,
                  vit_landmarks: List[Dict],
                  rtm_landmarks: List[Dict]) -> np.ndarray:
    """Draw combined skeleton: ViTPose body (0-16) + RTMPose face (23-90) + hands (91-132)."""
    vis_img = frame_bgr.copy()
    h, w = vis_img.shape[:2]

    body_map = {lm['idx']: lm for lm in vit_landmarks if lm['idx'] <= 16}
    rtm_map  = {lm['idx']: lm for lm in rtm_landmarks}

    def _pt(lm):
        return (int(max(0, min(1, lm['x'])) * w), int(max(0, min(1, lm['y'])) * h))

    def _draw_edges(idx_map, connections, color, thickness):
        for a, b in connections:
            la, lb = idx_map.get(a), idx_map.get(b)
            if la is None or lb is None:
                continue
            if la['visibility'] > 0.15 and lb['visibility'] > 0.15:
                cv2.line(vis_img, _pt(la), _pt(lb), color, thickness)

    # Body – green (ViTPose)
    _draw_edges(body_map, COCO17_CONNECTIONS, (60, 220, 60), 3)
    # Face – cyan (RTMPose)
    _draw_edges(rtm_map, FACE_68_CONNECTIONS, (200, 210, 0), 1)
    # Hands – orange (RTMPose)
    _draw_edges(rtm_map, LHAND_CONNECTIONS, (0, 160, 255), 2)
    _draw_edges(rtm_map, RHAND_CONNECTIONS, (0, 100, 220), 2)

    # Dots – body
    for lm in body_map.values():
        cv2.circle(vis_img, _pt(lm), 5, (60, 220, 60), -1)
        cv2.circle(vis_img, _pt(lm), 5, (200, 255, 200), 1)

    # Dots – face
    for idx, lm in rtm_map.items():
        if 23 <= idx <= 90:
            cv2.circle(vis_img, _pt(lm), 2, (200, 210, 0), -1)

    # Dots – hands
    for idx, lm in rtm_map.items():
        if 91 <= idx <= 132:
            color = (0, 160, 255) if idx <= 111 else (0, 100, 220)
            cv2.circle(vis_img, _pt(lm), 3, color, -1)

    return vis_img


# ── Availability ──────────────────────────────────────────────────────────────

def _has(lib: str) -> bool:
    return importlib.util.find_spec(lib) is not None

MediaPipeBackend.available = _has('mediapipe')
RTMPoseBackend.available   = _has('mmpose')
ViTPoseBackend.available   = _has('mmpose')

ALL_BACKENDS: List[type] = [MediaPipeBackend, RTMPoseBackend, ViTPoseBackend]
BACKEND_BY_ID: Dict[str, type] = {b.backend_id: b for b in ALL_BACKENDS}
