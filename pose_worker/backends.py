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

import time
import threading

import cv2
import numpy as np

_MODEL_LOAD_LOCK = threading.Lock()

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
MODEL_PATH = "/data/models/pose_landmarker_heavy.task"

_MODEL_CACHE: Dict[str, Any] = {}


def _ensure_mediapipe_model():
    import urllib.request
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        log.info("Downloading MediaPipe pose model → %s …", MODEL_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("MediaPipe model cached at %s", MODEL_PATH)
    else:
        log.info("MediaPipe model found in cache: %s", MODEL_PATH)


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
            with _MODEL_LOAD_LOCK:
                if cls.backend_id not in _MODEL_CACHE:  # re-check after acquiring
                    log.info(">>> Loading model: %s – this may take 1-3 minutes …", cls.backend_id)
                    t0 = time.perf_counter()
                    try:
                        _MODEL_CACHE[cls.backend_id] = cls._load_model()
                        log.info(">>> Model ready: %s  (%.1f s)", cls.backend_id, time.perf_counter() - t0)
                    except Exception as e:
                        log.error(">>> Model load failed: %s – %s. Marking unavailable.", cls.backend_id, e)
                        cls.available = False
                        raise
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
    _mmpose_model_alias:      str = ''
    _mmpose_checkpoint_file:  str = ''   # filename to look for in /data/models/
    _joint_names: List[str] = []

    @classmethod
    def _find_local_checkpoint(cls) -> Optional[str]:
        """Return path to local checkpoint if it exists in /data/models/, else None."""
        if not cls._mmpose_checkpoint_file:
            return None
        path = os.path.join('/data/models', cls._mmpose_checkpoint_file)
        if os.path.exists(path):
            log.info("%s: using local checkpoint %s", cls.backend_id, path)
            return path
        log.info("%s: local checkpoint not found at %s – mim will download", cls.backend_id, path)
        return None

    @classmethod
    def _load_model(cls):
        from mmpose.apis import MMPoseInferencer
        from mmengine.registry import DefaultScope
        import torch, uuid
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        log.info("MMPose using device: %s", device)
        local_ckpt = cls._find_local_checkpoint()
        scope_name = str(uuid.uuid4())
        DefaultScope.get_instance(scope_name, scope_name='mmpose')
        try:
            if local_ckpt:
                return MMPoseInferencer(pose2d=cls._mmpose_model_alias,
                                        pose2d_weights=local_ckpt, device=device)
            return MMPoseInferencer(pose2d=cls._mmpose_model_alias, device=device)
        finally:
            DefaultScope.get_instance(str(uuid.uuid4()), scope_name='mmpose')

    @classmethod
    def detect(cls, frame_bgr: np.ndarray) -> List[Dict]:
        inferencer = cls._get_model()
        h, w = frame_bgr.shape[:2]

        t0 = time.perf_counter()
        result_gen = inferencer(frame_bgr, return_vis=False, return_datasample=False)
        result = next(result_gen)
        log.info("%s inference: %.0f ms  (%dx%d)", cls.backend_id,
                 (time.perf_counter() - t0) * 1000, w, h)

        predictions = result.get('predictions', [[]])
        if not predictions or not predictions[0]:
            log.warning("%s: no predictions returned", cls.backend_id)
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
        log.info("%s: %d landmarks detected", cls.backend_id, len(landmarks))
        return landmarks


# ── RTMPose-L Wholebody ───────────────────────────────────────────────────────

class RTMPoseBackend(_MMPoseBackend):
    backend_id                = 'rtmpose'
    display_name              = 'RTMPose-L Wholebody'
    _mmpose_model_alias       = 'wholebody'
    _mmpose_checkpoint_file   = 'rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth'
    _joint_names              = COCO_WHOLEBODY_JOINT_NAMES

    @classmethod
    def connections(cls):
        return COCO_WHOLEBODY_CONNECTIONS


# ── ViTPose-H ────────────────────────────────────────────────────────────────

class ViTPoseBackend(_MMPoseBackend):
    backend_id                = 'vitpose'
    display_name              = 'ViTPose-H'
    _mmpose_model_alias       = 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
    _mmpose_checkpoint_file   = 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
    _joint_names              = COCO17_JOINT_NAMES

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


# ── SMPLer-X (Expressive Body Regression) ────────────────────────────────────

class SMPLerXBackend(PoseBackend):
    """
    SMPLer-X: one-stage expressive human body estimation via MMPose 3D.
    Gibt SMPL-X Parameter (beta, body_pose, expression, jaw, hands, transl) zurück.

    Weights: /data/models/smpler_x/smpler_x_h32.pth.tar
    Modell-Alias für MMPose: 'smpler_x_h32'
    Referenz: https://github.com/caizhongang/SMPLer-X
    """
    backend_id   = 'smpler_x'
    display_name = 'SMPLer-X (Expressive Body)'

    _CHECKPOINT_NAMES = [
        'smpler_x/smpler_x_h32.pth.tar',
        'smpler_x_h32.pth.tar',
        'smpler_x/smpler_x_b32.pth.tar',
        'smpler_x_b32.pth.tar',
    ]
    # MMPose model aliases to try in order
    _MODEL_ALIASES = ['smpler_x_h32', 'smpler_x_b32']

    @classmethod
    def _find_checkpoint(cls) -> Optional[str]:
        for name in cls._CHECKPOINT_NAMES:
            path = os.path.join('/data/models', name)
            if os.path.exists(path):
                log.info("SMPLer-X: found checkpoint %s", path)
                return path
        log.warning("SMPLer-X: no checkpoint found in /data/models; "
                    "will try mim download (requires internet)")
        return None

    @classmethod
    def _load_model(cls):
        from mmpose.apis import MMPoseInferencer
        import torch, uuid
        from mmengine.registry import DefaultScope
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        log.info("SMPLer-X: loading model on device=%s", device)

        ckpt = cls._find_checkpoint()
        last_err = None
        for alias in cls._MODEL_ALIASES:
            try:
                scope_name = str(uuid.uuid4())
                DefaultScope.get_instance(scope_name, scope_name='mmpose')
                kwargs: Dict[str, Any] = dict(pose3d=alias, device=device)
                if ckpt:
                    kwargs['pose3d_weights'] = ckpt
                inferencer = MMPoseInferencer(**kwargs)
                log.info("SMPLer-X: loaded via alias '%s'", alias)
                return inferencer
            except Exception as e:
                log.warning("SMPLer-X: alias '%s' failed – %s", alias, e)
                last_err = e
        raise RuntimeError(f"SMPLer-X: all model aliases failed. Last error: {last_err}")

    @classmethod
    def regress(cls, frame_bgr: np.ndarray) -> Optional[Dict[str, List[float]]]:
        """
        Führt SMPL-X Regression durch und gibt die Parameter für die
        prominenteste Person zurück.

        Returns:
            Dict mit Keys: beta(10), body_pose(63), global_orient(3),
            transl(3), expression(10), jaw_pose(3),
            left_hand_pose(45), right_hand_pose(45)
            oder None wenn keine Person detektiert.
        """
        inferencer = cls._get_model()

        try:
            result_gen = inferencer(frame_bgr, return_vis=False, return_datasample=False)
            result = next(result_gen)
        except StopIteration:
            log.warning("SMPLer-X: no result (StopIteration)")
            return None
        except Exception as e:
            log.error("SMPLer-X inference error: %s", e)
            return None

        predictions = result.get('predictions', [[]])
        if not predictions or not predictions[0]:
            return None

        pred = predictions[0][0]

        # MMPose 3D-Inferenz: SMPL-X-Parameter können unter verschiedenen Keys stehen
        smplx: Dict[str, Any] = {}
        for key in ('smplx_param', 'smplx_params', 'param', 'smpl_param'):
            if key in pred and isinstance(pred[key], dict):
                smplx = pred[key]
                log.debug("SMPLer-X: found params under key '%s'", key)
                break

        if not smplx:
            # Fallback: Parameter direkt im pred-Dict suchen
            smplx = pred
            log.debug("SMPLer-X: params directly in pred dict")

        def _to_floats(v, n: int) -> List[float]:
            """Tensor/array/list → flat Python float list, zero-padded to length n."""
            if v is None:
                return [0.0] * n
            try:
                import numpy as _np
                if hasattr(v, 'detach'):
                    v = v.detach().cpu().numpy()
                arr = _np.asarray(v, dtype=float).flatten()
                result_arr = arr[:n].tolist()
                if len(result_arr) < n:
                    result_arr += [0.0] * (n - len(result_arr))
                return result_arr
            except Exception:
                return [0.0] * n

        def _get(keys, n):
            for k in keys:
                if k in smplx:
                    return _to_floats(smplx[k], n)
            return [0.0] * n

        return {
            'beta':            _get(['betas', 'beta', 'shape'], 10),
            'body_pose':       _get(['body_pose', 'pose'], 63),
            'global_orient':   _get(['global_orient', 'root_pose'], 3),
            'transl':          _get(['transl', 'translation'], 3),
            'expression':      _get(['expression', 'expr'], 10),
            'jaw_pose':        _get(['jaw_pose'], 3),
            'left_hand_pose':  _get(['left_hand_pose', 'lhand_pose'], 45),
            'right_hand_pose': _get(['right_hand_pose', 'rhand_pose'], 45),
        }

    @classmethod
    def detect(cls, frame_bgr: np.ndarray) -> List[Dict]:
        """detect() interface — liefert 3D-Keypoints als 2D-Projektion zurück."""
        inferencer = cls._get_model()
        try:
            result_gen = inferencer(frame_bgr, return_vis=False, return_datasample=False)
            result = next(result_gen)
        except StopIteration:
            return []
        predictions = result.get('predictions', [[]])
        if not predictions or not predictions[0]:
            return []
        pred = predictions[0][0]
        keypoints = pred.get('keypoints', [])
        scores = pred.get('keypoint_scores', [])
        h, w = frame_bgr.shape[:2]
        landmarks = []
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            # kp kann [x,y] oder [x,y,z] sein – erste 2 Werte als 2D verwenden
            x_norm = float(kp[0]) / w if float(kp[0]) > 1.0 else float(kp[0])
            y_norm = float(kp[1]) / h if float(kp[1]) > 1.0 else float(kp[1])
            landmarks.append(_make_landmark(i, f'smplx_{i}', x_norm, y_norm, 0.0, float(score)))
        return landmarks


# ── Availability ──────────────────────────────────────────────────────────────

def _has(lib: str) -> bool:
    try:
        import importlib
        importlib.import_module(lib)
        return True
    except Exception as e:
        log.warning("_has(%s): import failed – %s: %s", lib, type(e).__name__, e)
        return False

# ── 4D-Humans / HMR2 ─────────────────────────────────────────────────────────

class HMR2Backend:
    backend_id   = 'hmr2'
    display_name = '4D-Humans HMR2'
    available    = False
    _model       = None
    _cfg         = None

    CACHE_DIR = '/data/models/hmr2'

    @classmethod
    def _find_checkpoint(cls):
        """Find checkpoint file in CACHE_DIR, return path or None."""
        import glob
        from pathlib import Path
        patterns = [
            f'{cls.CACHE_DIR}/**/*.ckpt',
            f'{cls.CACHE_DIR}/**/*.pth',
            f'{cls.CACHE_DIR}/**/*.pt',
        ]
        for pattern in patterns:
            hits = glob.glob(pattern, recursive=True)
            if hits:
                ckpt = max(hits, key=lambda p: os.path.getsize(p))
                log.info("HMR2: found checkpoint %s", ckpt)
                return ckpt
        return None

    @classmethod
    def _load(cls):
        from hmr2.models import load_hmr2, download_models
        from pathlib import Path
        ckpt_dir = Path(cls.CACHE_DIR)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = cls._find_checkpoint()
        if ckpt is None:
            log.info("HMR2: no checkpoint found – downloading to %s …", ckpt_dir)
            download_models(str(ckpt_dir))
            ckpt = cls._find_checkpoint()
        if ckpt is None:
            raise RuntimeError(f"HMR2: no checkpoint found in {cls.CACHE_DIR} after download")

        log.info("HMR2: loading from %s", ckpt)
        model, cfg = load_hmr2(ckpt)
        model = model.eval()
        log.info("HMR2: model ready")
        return model, cfg

    @classmethod
    def regress(cls, frame_bgr: np.ndarray) -> Optional[Dict]:
        import torch
        from scipy.spatial.transform import Rotation

        if cls._model is None:
            cls._model, cls._cfg = cls._load()

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess: 256×256, ImageNet normalisation
        try:
            from hmr2.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
            mean = np.array(DEFAULT_MEAN, dtype=np.float32)
            std  = np.array(DEFAULT_STD,  dtype=np.float32)
        except ImportError:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
        img = (img - mean) / std                                   # (256, 256, 3)
        img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)   # (1, 3, 256, 256)

        with torch.no_grad():
            out = cls._model({'img': img_t})

        smpl = out['pred_smpl_params']

        # Rotation matrices → axis-angle
        body_rotmat   = smpl['body_pose'][0].cpu().numpy()     # (23, 3, 3)
        global_rotmat = smpl['global_orient'][0].cpu().numpy() # (1,  3, 3)

        # SMPL-X body: 21 joints (SMPL joints 0-20; joints 21-22 are hand roots → skip)
        body_aa   = Rotation.from_matrix(body_rotmat[:21]).as_rotvec().flatten().tolist()  # 63
        global_aa = Rotation.from_matrix(global_rotmat[0]).as_rotvec().tolist()            # 3
        betas     = smpl['betas'][0].cpu().numpy().tolist()                                # 10

        # Rough translation from weak-perspective camera
        pred_cam = out.get('pred_cam')
        if pred_cam is not None:
            s, tx, ty = pred_cam[0].cpu().numpy()
            transl = [float(tx), float(ty), float(10.0 / max(s, 0.1))]
        else:
            transl = [0.0, 0.0, 3.0]

        return {
            'betas':            betas,
            'body_pose':        body_aa,
            'global_orient':    global_aa,
            'transl':           transl,
            'expression':       [0.0] * 10,
            'jaw_pose':         [0.0] * 3,
            'left_hand_pose':   [0.0] * 12,
            'right_hand_pose':  [0.0] * 12,
        }


MediaPipeBackend.available = _has('mediapipe')
log.info("MediaPipe available: %s", MediaPipeBackend.available)
RTMPoseBackend.available   = _has('mmpose')
ViTPoseBackend.available   = _has('mmpose')
log.info("MMPose available: %s", RTMPoseBackend.available)

# SMPLer-X: verfügbar wenn mmpose UND smplx installiert
_smplx_pkg_available = _has('smplx')
SMPLerXBackend.available = RTMPoseBackend.available and _smplx_pkg_available
log.info("SMPLer-X available: %s (mmpose=%s, smplx=%s)",
         SMPLerXBackend.available, RTMPoseBackend.available, _smplx_pkg_available)

# HMR2
HMR2Backend.available = _has('hmr2')
log.info("HMR2 available: %s", HMR2Backend.available)

ALL_BACKENDS: List[type] = [MediaPipeBackend, RTMPoseBackend, ViTPoseBackend, SMPLerXBackend]
BACKEND_BY_ID: Dict[str, type] = {b.backend_id: b for b in ALL_BACKENDS}
