"""
Stage 1 – SMPL-X Fitting

Estimates body shape β and per-frame pose θ_t from video frames.

Two-phase optimisation:
  Phase 1 – Shape:  β on static frames (low joint velocity).
  Phase 2 – Pose:   θ_t per frame with β frozen, batched forward pass.

Outputs saved to avatar.data_path:
  poses.npz      – theta_t, global_orient_t, transl_t, T_bones_t
  metadata.json  – beta, camera_intrinsics, fitting_quality (merged)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

log = logging.getLogger(__name__)


# ── Joint-index mapping ────────────────────────────────────────────────────────
#
# SMPL-X body joints 0-21 (pelvis + 21 body joints).
# Maps  smplx_joint_idx  →  mediapipe_landmark_idx
# Only joints visible in MediaPipe Pose are included.

_SMPLX_TO_MP: Dict[int, int] = {
    1:  23,   # left_hip       → mp left_hip
    2:  24,   # right_hip      → mp right_hip
    4:  25,   # left_knee      → mp left_knee
    5:  26,   # right_knee     → mp right_knee
    7:  27,   # left_ankle     → mp left_ankle
    8:  28,   # right_ankle    → mp right_ankle
    10: 31,   # left_foot      → mp left_foot_index
    11: 32,   # right_foot     → mp right_foot_index
    16: 11,   # left_shoulder  → mp left_shoulder
    17: 12,   # right_shoulder → mp right_shoulder
    18: 13,   # left_elbow     → mp left_elbow
    19: 14,   # right_elbow    → mp right_elbow
    20: 15,   # left_wrist     → mp left_wrist
    21: 16,   # right_wrist    → mp right_wrist
}

# Ordered lists for vectorised indexing
_SMPLX_IDX = sorted(_SMPLX_TO_MP.keys())           # indices into smplx joints
_MP_IDX    = [_SMPLX_TO_MP[j] for j in _SMPLX_IDX] # indices into mp landmarks

MAX_FIT_FRAMES = 150   # cap frames to keep memory manageable


# ── Config & result types ──────────────────────────────────────────────────────

@dataclass
class Stage1Config:
    num_betas:                  int   = 10
    n_shape_epochs:             int   = 100
    n_pose_epochs:              int   = 100
    lr_shape:                   float = 5e-3
    lr_pose:                    float = 5e-3
    w_keypoint:                 float = 1.0
    w_silhouette:               float = 0.5
    w_shape_prior:              float = 0.01
    w_pose_prior:               float = 0.001
    static_velocity_threshold:  float = 0.05


@dataclass
class Stage1Result:
    beta:              np.ndarray   # (num_betas,)
    theta_t:           np.ndarray   # (N, 63)  body pose axis-angle
    global_orient_t:   np.ndarray   # (N, 3)
    transl_t:          np.ndarray   # (N, 3)
    T_bones_t:         np.ndarray   # (N, 55, 4, 4)  bone transforms
    camera_intrinsics: dict
    fitting_quality:   dict


ProgressCB = Callable[[dict], None]


# ── Entry point ────────────────────────────────────────────────────────────────

def run_stage1(
    avatar,
    config_dict: dict,
    progress_cb: Optional[ProgressCB] = None,
) -> Stage1Result:
    """
    Run Stage 1 SMPL-X fitting.

    Args:
        avatar:      Avatar Django model instance (.group, .data_path).
        config_dict: Fitting config from the FittingJob.
        progress_cb: Called with a progress dict every 10 epochs.

    Returns:
        Stage1Result containing β, per-frame θ, T_bones and camera params.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Stage1Config(
        static_velocity_threshold = config_dict.get('static_threshold', 0.05),
        num_betas                 = config_dict.get('num_betas', 10),
    )
    log.info("Stage 1 on device=%s  num_betas=%d", device, cfg.num_betas)

    smplx_model = _load_smplx(cfg, device)

    frames, masks, (H, W) = _load_person_frames(avatar)
    if not frames:
        raise RuntimeError(
            "No frames found – run person detection first "
            f"(avatar.group={avatar.group})."
        )
    N = len(frames)
    log.info("Stage 1: %d frames  %dx%d", N, W, H)

    cam = _default_intrinsics(W, H)

    _cb(progress_cb, 0, cfg.n_shape_epochs + cfg.n_pose_epochs,
        0.0, {}, note='Extracting 2D keypoints…')

    kp_2d = _extract_keypoints(frames)   # (N, n_joints, 3)  x_norm, y_norm, vis

    static_idx = _select_static_frames(kp_2d, cfg.static_velocity_threshold)
    if not static_idx:
        static_idx = list(range(min(10, N)))
    log.info("Stage 1: %d/%d static frames selected", len(static_idx), N)

    # ── Phase 1: shape estimation on static frames ────────────────────────────

    beta        = nn.Parameter(torch.zeros(1, cfg.num_betas, device=device))
    glob_orient = nn.Parameter(torch.zeros(1, 3, device=device))
    transl_ref  = nn.Parameter(torch.tensor([[0.0, 0.0, 3.0]], device=device))

    opt_shape = torch.optim.Adam([beta, glob_orient, transl_ref], lr=cfg.lr_shape)
    total_epochs = cfg.n_shape_epochs + cfg.n_pose_epochs

    for epoch in range(cfg.n_shape_epochs):
        opt_shape.zero_grad()

        out     = smplx_model(betas=beta, global_orient=glob_orient, transl=transl_ref)
        j2d     = _project(out.joints[0, _SMPLX_IDX], cam)       # (n_joints, 2) px
        kp_mean = torch.tensor(
            kp_2d[static_idx].mean(axis=0), dtype=torch.float32, device=device
        )  # mean keypoints over static frames → stable shape target

        loss_kp = _kp_loss(j2d, kp_mean, cam)
        loss_sp = (beta ** 2).mean()
        loss    = cfg.w_keypoint * loss_kp + cfg.w_shape_prior * loss_sp
        loss.backward()
        opt_shape.step()

        if epoch % 10 == 0:
            preview = None
            if epoch % 50 == 0:
                preview = _make_preview(frames[static_idx[0]],
                                        out.joints[0, _SMPLX_IDX].detach(), cam)
            _cb(progress_cb, epoch, total_epochs, float(loss), {
                'keypoint':    round(float(loss_kp), 4),
                'shape_prior': round(float(loss_sp), 4),
            }, preview_jpg=preview)

    beta_fixed = beta.detach()

    # ── Phase 2: pose estimation per frame (batched) ──────────────────────────

    body_pose  = nn.Parameter(torch.zeros(N, 63, device=device))
    orient_all = nn.Parameter(torch.zeros(N, 3, device=device))
    transl_all = nn.Parameter(transl_ref.detach().expand(N, -1).clone())

    opt_pose = torch.optim.Adam([body_pose, orient_all, transl_all], lr=cfg.lr_pose)

    kp_tensor = torch.tensor(kp_2d, dtype=torch.float32, device=device)  # (N, n_j, 3)
    sil_ok    = _pytorch3d_available()

    loss_kp = loss_sil = loss_pp = torch.tensor(0.0)   # init for final quality

    for epoch in range(cfg.n_pose_epochs):
        opt_pose.zero_grad()

        out = smplx_model(
            betas        = beta_fixed.expand(N, -1),
            body_pose    = body_pose,
            global_orient= orient_all,
            transl       = transl_all,
        )
        # out.joints: (N, n_total_joints, 3)
        # out.A:      (N, 55, 4, 4)

        j2d_batch = _project_batch(out.joints[:, _SMPLX_IDX, :], cam)  # (N, n_j, 2)

        loss_kp  = _kp_loss_batch(j2d_batch, kp_tensor, cam)
        loss_pp  = (body_pose ** 2).mean()
        loss_sil = _silhouette_loss_batch(out.vertices, smplx_model.faces_tensor,
                                          masks, cam, device) if sil_ok else \
                   torch.zeros(1, device=device)

        loss = (cfg.w_keypoint   * loss_kp +
                cfg.w_silhouette * loss_sil +
                cfg.w_pose_prior * loss_pp)
        loss.backward()
        opt_pose.step()

        if epoch % 10 == 0:
            preview = None
            if epoch % 50 == 0:
                preview = _make_preview(frames[0],
                                        out.joints[0, _SMPLX_IDX].detach(), cam)
            _cb(progress_cb, cfg.n_shape_epochs + epoch, total_epochs, float(loss), {
                'keypoint':   round(float(loss_kp), 4),
                'silhouette': round(float(loss_sil), 4),
                'pose_prior': round(float(loss_pp), 4),
            }, preview_jpg=preview)

    # ── Final forward pass: collect T_bones_t ────────────────────────────────

    with torch.no_grad():
        out_final = smplx_model(
            betas         = beta_fixed.expand(N, -1),
            body_pose     = body_pose.detach(),
            global_orient = orient_all.detach(),
            transl        = transl_all.detach(),
        )

    try:
        T_bones_t = out_final.A.cpu().numpy()    # (N, 55, 4, 4)
    except AttributeError:
        log.warning("output.A not available – computing T_bones_t from joints")
        T_bones_t = _compute_T_bones_from_joints(out_final.joints.detach().cpu().numpy())

    return Stage1Result(
        beta             = beta_fixed.squeeze(0).cpu().numpy(),
        theta_t          = body_pose.detach().cpu().numpy(),
        global_orient_t  = orient_all.detach().cpu().numpy(),
        transl_t         = transl_all.detach().cpu().numpy(),
        T_bones_t        = T_bones_t,
        camera_intrinsics= cam,
        fitting_quality  = {
            'stage1_keypoint_err': round(float(loss_kp), 4),
            'n_frames':            N,
            'n_static_frames':     len(static_idx),
        },
    )


def save_stage1_result(result: Stage1Result, data_path: str) -> None:
    """Write Stage 1 outputs into the avatar data folder."""
    os.makedirs(data_path, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_path, 'poses.npz'),
        theta_t         = result.theta_t,
        global_orient_t = result.global_orient_t,
        transl_t        = result.transl_t,
        T_bones_t       = result.T_bones_t,
    )

    meta_path = os.path.join(data_path, 'metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta['beta']               = result.beta.tolist()
    meta['camera_intrinsics']  = result.camera_intrinsics
    meta.setdefault('fitting_quality', {}).update(result.fitting_quality)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info("Stage 1 results saved → %s", data_path)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_person_frames(
    avatar,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], Tuple[int, int]]:
    """
    Load BGR frames and binary segmentation masks for all DetectedPerson tracks
    assigned to the avatar's PersonGroup.

    Returns:
        frames  – list of BGR images (≤ MAX_FIT_FRAMES)
        masks   – list of binary masks, same length (None where unavailable)
        (H, W)  – video frame dimensions
    """
    from core.models import DetectedPerson

    if avatar.group is None:
        raise RuntimeError("Avatar has no PersonGroup assigned.")

    persons = list(avatar.group.persons.select_related('video').all())
    if not persons:
        raise RuntimeError("PersonGroup has no DetectedPerson tracks.")

    H, W = _video_hw(persons[0].video.path)

    # Build flat list of (video_path, frame_idx, mask_or_None)
    candidates: List[Tuple[str, int, Optional[np.ndarray]]] = []
    for dp in persons:
        mask_lookup = _load_mask_lookup(dp.meta.get('mask_path'))
        for fi in range(dp.frame_start, dp.frame_end + 1, 3):
            candidates.append((dp.video.path, fi, mask_lookup.get(fi)))

    # Evenly sample down to MAX_FIT_FRAMES
    if len(candidates) > MAX_FIT_FRAMES:
        step       = len(candidates) // MAX_FIT_FRAMES
        candidates = candidates[::step][:MAX_FIT_FRAMES]

    frames, masks = [], []
    open_caps: Dict[str, cv2.VideoCapture] = {}

    for video_path, frame_idx, mask in candidates:
        cap = open_caps.setdefault(video_path, cv2.VideoCapture(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            masks.append(mask)

    for cap in open_caps.values():
        cap.release()

    return frames, masks, (H, W)


def _load_mask_lookup(mask_path: Optional[str]) -> Dict[int, np.ndarray]:
    """Load a masks .npz file into a {frame_idx: mask} dict."""
    if not mask_path or not os.path.exists(mask_path):
        return {}
    data = np.load(mask_path)
    return {int(fi): m for fi, m in zip(data['frame_indices'], data['masks'])}


def _video_hw(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


# ── SMPL-X ────────────────────────────────────────────────────────────────────

def _load_smplx(cfg: Stage1Config, device: torch.device):
    from django.conf import settings
    import smplx

    model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
    model = smplx.create(
        model_path    = model_dir,
        model_type    = 'smplx',
        gender        = 'neutral',
        num_betas     = cfg.num_betas,
        use_pca       = False,       # full body pose parameters
        flat_hand_mean= True,
    ).to(device)
    model.eval()
    return model


# ── Keypoint extraction ────────────────────────────────────────────────────────

def _extract_keypoints(frames: List[np.ndarray]) -> np.ndarray:
    """
    Run MediaPipe Pose on each frame.

    Returns (N, n_joints, 3) with (x_norm, y_norm, visibility) for each
    joint in _MP_IDX order (same order as _SMPLX_IDX).
    """
    n_joints = len(_MP_IDX)

    try:
        import mediapipe as mp
    except ImportError:
        log.warning("mediapipe not available – using zero keypoints")
        return np.zeros((len(frames), n_joints, 3), dtype=np.float32)

    mp_pose = mp.solutions.pose
    kp_all  = []

    with mp_pose.Pose(
        static_image_mode      = True,
        model_complexity       = 1,
        min_detection_confidence = 0.5,
    ) as pose:
        for frame in frames:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            row     = np.zeros((n_joints, 3), dtype=np.float32)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                for i, mp_idx in enumerate(_MP_IDX):
                    p = lm[mp_idx]
                    row[i] = [p.x, p.y, p.visibility]
            kp_all.append(row)

    return np.array(kp_all, dtype=np.float32)   # (N, n_joints, 3)


def _select_static_frames(kp_2d: np.ndarray, threshold: float) -> List[int]:
    """
    Return frame indices where mean joint velocity is below threshold.
    Velocity is mean L2 displacement (normalised coords) between consecutive frames.
    """
    if len(kp_2d) < 2:
        return list(range(len(kp_2d)))

    xy    = kp_2d[:, :, :2]                               # (N, n_j, 2)
    vis   = kp_2d[:, :, 2]                                # (N, n_j)
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=-1)  # (N-1, n_j)
    valid = (vis[:-1] > 0.5) & (vis[1:] > 0.5)
    delta = np.where(valid, delta, np.nan)
    vel   = np.nanmean(delta, axis=1)                      # (N-1,)

    return [t for t, v in enumerate(vel)
            if not np.isnan(v) and v < threshold]


# ── Camera ─────────────────────────────────────────────────────────────────────

def _default_intrinsics(W: int, H: int) -> dict:
    """Approximate pinhole camera from image dimensions."""
    f = float(max(W, H))
    return {'fx': f, 'fy': f, 'cx': W / 2.0, 'cy': H / 2.0, 'W': W, 'H': H}


def _project(joints_3d: torch.Tensor, cam: dict) -> torch.Tensor:
    """Project (n_joints, 3) → pixel coords (n_joints, 2)."""
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2].clamp(min=0.1)
    return torch.stack([
        X / Z * cam['fx'] + cam['cx'],
        Y / Z * cam['fy'] + cam['cy'],
    ], dim=-1)


def _project_batch(joints_3d: torch.Tensor, cam: dict) -> torch.Tensor:
    """Project (N, n_joints, 3) → pixel coords (N, n_joints, 2)."""
    X, Y, Z = joints_3d[..., 0], joints_3d[..., 1], joints_3d[..., 2].clamp(min=0.1)
    return torch.stack([
        X / Z * cam['fx'] + cam['cx'],
        Y / Z * cam['fy'] + cam['cy'],
    ], dim=-1)


# ── Loss functions ─────────────────────────────────────────────────────────────

def _kp_loss(j2d: torch.Tensor, kp: torch.Tensor, cam: dict) -> torch.Tensor:
    """
    Visibility-weighted keypoint loss in pixel space.

    j2d: (n_joints, 2)  – projected SMPL-X joints in pixels
    kp:  (n_joints, 3)  – MediaPipe (x_norm, y_norm, visibility)
    """
    vis       = kp[:, 2:3].clamp(0.0, 1.0)
    target_px = kp[:, :2] * torch.tensor(
        [[cam['W'], cam['H']]], dtype=j2d.dtype, device=j2d.device
    )
    diff = (j2d - target_px) ** 2
    return (diff.sum(dim=-1) * vis.squeeze(-1)).mean()


def _kp_loss_batch(j2d: torch.Tensor, kp: torch.Tensor, cam: dict) -> torch.Tensor:
    """
    Batched visibility-weighted keypoint loss.

    j2d: (N, n_joints, 2) pixels
    kp:  (N, n_joints, 3) mediapipe (x_norm, y_norm, vis)
    """
    vis       = kp[..., 2:3].clamp(0.0, 1.0)
    target_px = kp[..., :2] * torch.tensor(
        [[cam['W'], cam['H']]], dtype=j2d.dtype, device=j2d.device
    )
    diff = (j2d - target_px) ** 2
    return (diff.sum(dim=-1) * vis.squeeze(-1)).mean()


# ── Silhouette loss (PyTorch3D) ────────────────────────────────────────────────

def _pytorch3d_available() -> bool:
    try:
        import pytorch3d   # noqa: F401
        return True
    except ImportError:
        return False


def _silhouette_loss_batch(
    vertices:     torch.Tensor,        # (N, V, 3)
    faces_tensor: torch.Tensor,        # (F, 3)
    masks:        List[Optional[np.ndarray]],
    cam:          dict,
    device:       torch.device,
) -> torch.Tensor:
    """
    Differentiable silhouette loss using PyTorch3D soft rasterisation.
    Falls back to zero if no frame has a valid mask.

    Uses MSE between rendered soft silhouette and the binary detection mask.
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        MeshRasterizer,
        RasterizationSettings,
        SoftSilhouetteShader,
        MeshRenderer,
        BlendParams,
    )

    H, W = int(cam['H']), int(cam['W'])
    fx, fy = cam['fx'], cam['fy']
    cx, cy = cam['cx'], cam['cy']
    N = vertices.shape[0]

    # PyTorch3D uses NDC coords; convert intrinsics
    # focal_length and principal_point in NDC: divide by image half-size
    half_w, half_h = W / 2.0, H / 2.0
    focal_ndc = torch.tensor(
        [[fx / half_w, fy / half_h]], dtype=torch.float32, device=device
    ).expand(N, -1)
    pp_ndc = torch.tensor(
        [[(cx - half_w) / half_w, (cy - half_h) / half_h]],
        dtype=torch.float32, device=device,
    ).expand(N, -1)

    cameras = PerspectiveCameras(
        focal_length    = focal_ndc,
        principal_point = pp_ndc,
        in_ndc          = True,
        device          = device,
    )

    raster_settings = RasterizationSettings(
        image_size     = (H, W),
        blur_radius    = 1e-4,
        faces_per_pixel= 50,
    )
    renderer = MeshRenderer(
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader     = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4)),
    )

    faces_batch = faces_tensor.unsqueeze(0).expand(N, -1, -1)
    meshes      = Meshes(verts=vertices, faces=faces_batch)

    images = renderer(meshes)          # (N, H, W, 4)
    sil    = images[..., 3]            # alpha channel → silhouette (N, H, W)

    total = torch.zeros(1, device=device)
    count = 0
    for t, mask in enumerate(masks):
        if mask is None:
            continue
        gt = torch.tensor(mask, dtype=torch.float32, device=device)
        total = total + nn.functional.mse_loss(sil[t], gt)
        count += 1

    return total / count if count > 0 else total


# ── T_bones fallback ──────────────────────────────────────────────────────────

def _compute_T_bones_from_joints(joints: np.ndarray) -> np.ndarray:
    """
    Approximate T_bones_t as identity transforms centred on each joint.
    Used only when output.A is unavailable.

    joints: (N, n_joints, 3)
    Returns: (N, n_joints, 4, 4) – translation-only transforms
    """
    N, J, _ = joints.shape
    T = np.tile(np.eye(4), (N, J, 1, 1))
    T[:, :, :3, 3] = joints
    return T


# ── Preview ────────────────────────────────────────────────────────────────────

def _make_preview(
    frame_bgr: np.ndarray,
    joints_2d_3d: torch.Tensor,   # (n_joints, 3) in 3D – will be projected
    cam: dict,
) -> Optional[str]:
    """
    Project joints onto frame and return a base64-encoded JPEG string.
    """
    try:
        j2d = _project(joints_2d_3d.cpu(), cam).numpy()
        W, H = cam['W'], cam['H']
        vis  = frame_bgr.copy()
        for x, y in j2d:
            if 0 <= int(x) < W and 0 <= int(y) < H:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        scale = min(512 / W, 512 / H)
        out   = cv2.resize(vis, (int(W * scale), int(H * scale)))
        buf   = io.BytesIO()
        Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(
            buf, format='JPEG', quality=70
        )
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        log.debug("_make_preview failed", exc_info=True)
        return None


# ── Progress helper ────────────────────────────────────────────────────────────

def _cb(
    cb:           Optional[ProgressCB],
    epoch:        int,
    total_epochs: int,
    loss:         float,
    loss_terms:   dict,
    preview_jpg:  Optional[str] = None,
    note:         str = '',
) -> None:
    if cb is None:
        return
    payload = {
        'type':        'progress',
        'stage':       '1',
        'stage_name':  'SMPL-X Fitting',
        'epoch':        epoch,
        'total_epochs': total_epochs,
        'loss':         loss,
        'loss_terms':   loss_terms,
        'preview_jpg':  preview_jpg,
        'mesh_obj':     None,
        'texture_jpg':  None,
        'heatmap_jpg':  None,
    }
    if note:
        payload['note'] = note
    try:
        cb(payload)
    except Exception:
        log.exception("progress callback raised")
