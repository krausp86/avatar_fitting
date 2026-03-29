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

# Primary: ViTPose COCO-17 keypoints (same source as Stage 0 shape_fit)
# These are stored in PersonFrameKeypoints.body_landmarks as {idx, x, y, visibility}
# with x,y normalised 0-1.
_SMPLX_TO_COCO: Dict[int, int] = {
    1:  11,   # left_hip        → COCO left_hip
    2:  12,   # right_hip       → COCO right_hip
    4:  13,   # left_knee       → COCO left_knee
    5:  14,   # right_knee      → COCO right_knee
    7:  15,   # left_ankle      → COCO left_ankle
    8:  16,   # right_ankle     → COCO right_ankle
    16:  5,   # left_shoulder   → COCO left_shoulder
    17:  6,   # right_shoulder  → COCO right_shoulder
    18:  7,   # left_elbow      → COCO left_elbow
    19:  8,   # right_elbow     → COCO right_elbow
    20:  9,   # left_wrist      → COCO left_wrist
    21: 10,   # right_wrist     → COCO right_wrist
}

# Virtual joints: currently empty.
# Head/neck keypoints (nose→head, ears→neck) were tried but the anatomical
# offset (skull top ≠ nose, neck base ≠ ears) created competing gradients
# that worsened fitting. Proper head constraints need RTMPose face landmarks
# with explicit offset modelling.
_SMPLX_VIRTUAL: Dict[int, Tuple[int, int]] = {}

# Fallback: MediaPipe PoseLandmarker indices (used when DB cache is unavailable)
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

# Ordered SMPL-X joint indices (regular 1:1 mapping + virtual midpoints)
_SMPLX_IDX = sorted(set(_SMPLX_TO_COCO.keys()) | set(_SMPLX_VIRTUAL.keys()))
_MP_IDX    = [_SMPLX_TO_MP[j] for j in _SMPLX_IDX if j in _SMPLX_TO_MP]

MAX_FIT_FRAMES = 150   # cap frames to keep memory manageable
# On CPU-only machines load fewer frames to avoid OOM.  GPU has enough VRAM for 150.
import torch as _torch
MAX_FIT_FRAMES = 50 if not _torch.cuda.is_available() else 150
del _torch


# ── Config & result types ──────────────────────────────────────────────────────

@dataclass
class Stage1Config:
    num_betas:                  int   = 10
    gender:                     str   = 'neutral'   # 'neutral' | 'male' | 'female'
    max_frames:                 int   = 150          # cap on training frames
    n_warmup_epochs:            int   = 50           # orient+transl only
    n_shape_epochs:             int   = 200          # beta+transl, pose frozen
    n_poseref_epochs:           int   = 100          # pose_ref+orient+transl, beta frozen
    n_pose_epochs:              int   = 300          # per-frame pose (Phase 2)
    lr_shape:                   float = 5e-3
    lr_pose:                    float = 5e-3
    w_keypoint:                 float = 1.0
    w_shape_prior:              float = 0.01
    w_pose_prior:               float = 0.001
    w_temporal:                 float = 0.1          # adjacent-frame pose smoothness
    static_velocity_threshold:  float = 0.05


@dataclass
class Stage1Result:
    beta:              np.ndarray   # (num_betas,)
    gender:            str
    theta_t:           np.ndarray   # (N, 63)  body pose axis-angle
    global_orient_t:   np.ndarray   # (N, 3)
    transl_t:          np.ndarray   # (N, 3)
    T_bones_t:         np.ndarray   # (N, 55, 4, 4)  bone transforms
    camera_intrinsics: dict
    fitting_quality:   dict
    kp_2d:             Optional[np.ndarray] = None  # (N, n_joints, 3) observed keypoints
    person_frame_pairs: Optional[list] = None       # [(person_id, frame_idx, video_path), …]


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
    # Check for pre-fitted PersonShape – if available, betas are known and the
    # shape phase can be skipped entirely.
    _person_shape = None
    try:
        if avatar.group_id:
            from ..models import PersonShape
            _person_shape = PersonShape.objects.filter(
                group_id=avatar.group_id, status='done'
            ).first()
    except Exception:
        pass

    if _person_shape:
        log.info("Stage 1: using pre-fitted PersonShape betas (focal_scale=%.3f)",
                 _person_shape.focal_scale)
    else:
        log.warning("Stage 1: no PersonShape found for group – fitting beta from scratch")

    cfg = Stage1Config(
        static_velocity_threshold = config_dict.get('static_threshold', 0.05),
        num_betas                 = config_dict.get('num_betas', 10),
        gender                    = config_dict.get('gender', 'neutral'),
        max_frames                = config_dict.get('max_frames', MAX_FIT_FRAMES),
        n_warmup_epochs           = config_dict.get('n_warmup_epochs', 50),
        n_poseref_epochs          = config_dict.get('n_poseref_epochs', 100),
        n_pose_epochs             = config_dict.get('n_pose_epochs', 300),
        # Skip shape phase when PersonShape is already available
        n_shape_epochs            = 0 if _person_shape else config_dict.get('n_shape_epochs', 200),
    )
    log.info("Stage 1 on device=%s  num_betas=%d  n_shape_epochs=%d",
             device, cfg.num_betas, cfg.n_shape_epochs)

    smplx_model = _load_smplx(cfg, device, batch_size=1)

    total_epochs = cfg.n_warmup_epochs + cfg.n_shape_epochs + cfg.n_poseref_epochs + cfg.n_pose_epochs
    _cb(progress_cb, 0, total_epochs, 0.0, {}, note='Lade Video-Frames…')

    frames, masks, (H, W), temporal_valid, person_frame_pairs = _load_person_frames(
        avatar, max_frames=cfg.max_frames, load_masks=False
    )
    if not frames:
        raise RuntimeError(
            "No frames found – run person detection first "
            f"(avatar.group={avatar.group})."
        )
    N = len(frames)
    log.info("Stage 1: %d frames  %dx%d", N, W, H)

    # ROMP initialisation — gives β + θ as warm start (graceful fallback to None)
    from .romp_init import romp_init_frames
    from django.conf import settings as _django_settings
    _romp = romp_init_frames(
        frames,
        model_path=getattr(_django_settings, 'ROMP_MODEL_PATH', None),
        smpl_path=getattr(_django_settings, 'ROMP_SMPL_PATH',   None),
    )

    # Use focal_scale from Stage 0 PersonShape if available; otherwise default.
    # shape_fit uses fx_base = max(W,H) * 1.2 as its base, and PersonShape.focal_scale
    # is a multiplier on top of that base.  stage1 must apply the same 1.2 factor so
    # that the focal length is consistent between the two stages.
    _focal_scale = float(_person_shape.focal_scale) if _person_shape else 1.0
    cam = _default_intrinsics(W, H, focal_scale=1.2 * _focal_scale)
    log.info("Stage 1: focal = max(W,H)*1.2*%.3f = %.1f px", _focal_scale, cam['fx'])

    _cb(progress_cb, 0, total_epochs, 0.0, {},
        note=f'{N} Frames geladen ({W}×{H}) – extrahiere Keypoints…')

    # Prefer DB-cached ViTPose keypoints (same source as Stage 0 / debug_pose).
    # Fall back to running MediaPipe locally if the cache is incomplete.
    kp_2d = _extract_keypoints_db(person_frame_pairs)
    if kp_2d is None:
        log.info("Stage 1: DB keypoints unavailable – running MediaPipe fallback")
        kp_2d = _extract_keypoints(frames)   # (N, n_joints, 3) fallback

    # ── Diagnostics: check how many frames had a visible person ──────────────
    mean_vis = float(kp_2d[:, :, 2].mean())
    detected_frames = int((kp_2d[:, :, 2].max(axis=1) > 0.3).sum())
    log.info(
        "Stage 1 keypoints: %d/%d frames with detection, mean visibility=%.3f",
        detected_frames, N, mean_vis,
    )
    _cb(progress_cb, 0, total_epochs, 0.0, {},
        note=f'Keypoints: {detected_frames}/{N} Frames erkannt, vis={mean_vis:.3f}')
    if detected_frames == 0:
        raise RuntimeError(
            "MediaPipe detected no person in any frame. "
            "Check that the PersonGroup contains frames where the person is clearly visible, "
            "and that MediaPipe is correctly installed. "
            f"(mean_visibility={mean_vis:.4f}, frames={N})"
        )
    if mean_vis < 0.1:
        log.warning(
            "Very low mean keypoint visibility (%.3f) – fitting quality will be poor. "
            "Try re-running person detection on better-lit / less-occluded footage.",
            mean_vis,
        )

    # Frames where at least one joint was detected with sufficient confidence
    frame_detected = kp_2d[:, :, 2].max(axis=1) > 0.3   # (N,) bool numpy

    static_idx = _select_static_frames(kp_2d, cfg.static_velocity_threshold)
    # Only use static frames that actually have a detected person
    static_idx = [i for i in static_idx if frame_detected[i]]
    if not static_idx:
        # Fallback: any detected frame
        static_idx = [i for i in range(N) if frame_detected[i]][:10]
    if not static_idx:
        static_idx = list(range(min(10, N)))   # last resort
    log.info(
        "Stage 1: %d/%d static frames selected (%d/%d frames have detections)",
        len(static_idx), N, int(frame_detected.sum()), N,
    )

    # ── Phase 1: shape + reference-pose estimation on static frames ─────────────
    # Crucially we also optimise body_pose_ref so the model matches the actual
    # pose in the video rather than forcing T-pose onto posed-person keypoints.

    # ── Phase 1 shared setup ──────────────────────────────────────────────────

    # glob_orient starts at zero (identity rotation).
    # SMPL-X canonical Y+ = up; image Y+ = down → negate Y in _project/_project_batch.
    # Starting at [π,0,0] (the old approach) places the parameter at the axis-angle
    # singularity (||θ||=π) where Jacobian ∂R/∂θ is ill-conditioned → slow/failed convergence.
    # shape_fit.py uses the same zero-init + explicit Y-negation convention.
    if _person_shape and len(_person_shape.betas) == cfg.num_betas:
        _beta_init = torch.tensor([_person_shape.betas], dtype=torch.float32, device=device)
    elif _romp is not None:
        _beta_init = torch.tensor(
            [_romp['beta'][:cfg.num_betas]], dtype=torch.float32, device=device
        )
        log.info("Stage 1: beta init from ROMP (no PersonShape available)")
    else:
        _beta_init = torch.zeros(1, cfg.num_betas, device=device)
    beta          = nn.Parameter(_beta_init)
    glob_orient   = nn.Parameter(torch.zeros(1, 3, device=device))

    # VPoser pose prior (optional – falls back to direct 63-dim if unavailable)
    from .shape_fit import _try_load_vposer
    _vposer    = _try_load_vposer(device)
    _use_vpose = _vposer is not None
    log.info("Stage 1: use_vposer=%s", _use_vpose)
    _cb(progress_cb, 0, 1, 0.0, {},
        note=f'Pose prior: {"✓ VPoser V02_05" if _use_vpose else "⚠ L2 Fallback (VPoser nicht gefunden)"}')

    _POSE_DIM = 32 if _use_vpose else 63

    def _decode_pose(z: torch.Tensor) -> torch.Tensor:
        """z: (B, _POSE_DIM) → (B, 63) body_pose axis-angle."""
        if _use_vpose:
            decoded = _vposer.decode(z)
            bp = decoded.get('pose_body', decoded.get('pose_body_matrot'))
            return bp.reshape(z.shape[0], 63)
        return z

    z_ref         = nn.Parameter(torch.zeros(1, _POSE_DIM, device=device))
    # Alias so Phase 1a/1b code (which calls .detach()) keeps working unchanged
    body_pose_ref = z_ref

    # Central reference frame: the static frame with highest mean keypoint visibility.
    # All Phase 1b/1c optimisation uses this single frame so that beta and pose_ref
    # are not entangled across frames with differing body positions.
    center_idx = max(static_idx, key=lambda i: float(kp_2d[i, :, 2].mean()))
    kp_center_np = kp_2d[center_idx]                      # (n_j, 3)
    vis_mask     = kp_center_np[:, 2] > 0.4
    kp_center    = torch.tensor(kp_center_np, dtype=torch.float32, device=device)

    # Initialise z_ref from ROMP body pose of the center frame.
    # ROMP theta: (N, 72) = global_orient(3) | body_pose(69 for SMPL 23 joints).
    # SMPL-X body_pose has 21 joints = 63 dims; first 63 of ROMP's 69-dim body_pose
    # correspond to the shared joints — a good approximation for warm-starting VPoser.
    if _romp is not None:
        _romp_theta_idx = min(center_idx, len(_romp['thetas']) - 1)
        _romp_bp63 = torch.tensor(
            _romp['thetas'][_romp_theta_idx][3:66],   # skip global_orient, take 63 dims
            dtype=torch.float32, device=device,
        ).unsqueeze(0)  # (1, 63)
        if _use_vpose:
            with torch.no_grad():
                _enc = _vposer.encode(_romp_bp63)
                _z_mean = _enc.mean if hasattr(_enc, 'mean') else _enc
                z_ref.data.copy_(_z_mean.clamp(-2, 2))
            log.info("Stage 1: z_ref initialized from ROMP via VPoser")
        else:
            z_ref.data.copy_(_romp_bp63)
            log.info("Stage 1: z_ref initialized from ROMP body_pose")

    # Phase 1a uses the center frame PLUS its guaranteed temporal neighbours so
    # that glob_orient and transl_ref are anchored to real motion, not a single
    # possibly atypical frame.  Using consecutive indices in kp_2d guarantees
    # adjacency regardless of how many total frames are in the dataset.
    prev_idx   = max(0, center_idx - 1)
    next_idx   = min(N - 1, center_idx + 1)
    triplet_kp = [
        torch.tensor(kp_2d[i], dtype=torch.float32, device=device)
        for i in dict.fromkeys([prev_idx, center_idx, next_idx])   # deduplicate at boundaries
    ]
    log.info("Stage 1 reference frame: %d  (triplet: %d-%d-%d)",
             center_idx, prev_idx, center_idx, next_idx)

    # Z initialisation from center-frame keypoint scale
    _shoulder_l = _SMPLX_IDX.index(16) if 16 in _SMPLX_IDX else None
    _shoulder_r = _SMPLX_IDX.index(17) if 17 in _SMPLX_IDX else None
    _ankle_l    = _SMPLX_IDX.index(7)  if 7  in _SMPLX_IDX else None
    _ankle_r    = _SMPLX_IDX.index(8)  if 8  in _SMPLX_IDX else None

    ankle_vis = (
        (_ankle_l is not None and vis_mask[_ankle_l]) or
        (_ankle_r is not None and vis_mask[_ankle_r])
    )
    if _shoulder_l is not None and vis_mask[_shoulder_l] and ankle_vis:
        ankle_idx = _ankle_l if (_ankle_l is not None and vis_mask[_ankle_l]) else _ankle_r
        dy_px  = abs(kp_center_np[_shoulder_l, 1] - kp_center_np[ankle_idx, 1]) * cam['H']
        # 1.311 m = SMPL-X neutral shoulder(Y=0.085) to ankle(Y=-1.226) distance.
        # With -Y projection: dy_px = 1.311 / Z * fy  →  Z = 1.311 * fy / dy_px
        Z_init = float(np.clip(cam['fy'] * 1.311 / max(dy_px, 1.0), 1.0, 15.0))
        log.info("Stage 1: Z init shoulder-ankle: dy=%.1f px → Z=%.2f m", dy_px, Z_init)
    elif (_shoulder_l is not None and _shoulder_r is not None
            and vis_mask[_shoulder_l] and vis_mask[_shoulder_r]):
        dx_px = abs(kp_center_np[_shoulder_l, 0] - kp_center_np[_shoulder_r, 0]) * cam['W']
        if dx_px > 10:
            # 0.316 m = SMPL-X neutral shoulder width (X: 0.164 to -0.152)
            Z_init = float(np.clip(cam['fx'] * 0.316 / max(dx_px, 1.0), 1.0, 15.0))
            log.info("Stage 1: Z init shoulder width: dx=%.1f px → Z=%.2f m", dx_px, Z_init)
        else:
            Z_init = 3.0
    else:
        Z_init = 3.0
        log.info("Stage 1: Z init fallback Z=%.2f m", Z_init)

    # Initialise XY translation so the body starts projected at the keypoint
    # centroid rather than at the image centre.  Without this, warm-up has to
    # walk from (cx,cy) to the actual person position, which takes many epochs.
    _vis_kp = kp_center_np[vis_mask]   # (n_vis, 3)
    if len(_vis_kp) >= 2:
        kp_cx_px = float(_vis_kp[:, 0].mean()) * cam['W']
        kp_cy_px = float(_vis_kp[:, 1].mean()) * cam['H']
        X_init   =  (kp_cx_px - cam['cx']) / cam['fx'] * Z_init
        # With -Y projection: y_px = -Y/Z*fy + cy  →  Y = (cy - y_px)*Z/fy
        Y_init   = -(kp_cy_px - cam['cy']) / cam['fy'] * Z_init
        log.info("Stage 1: XY init from keypoint centroid: X=%.3f m  Y=%.3f m", X_init, Y_init)
    else:
        X_init = Y_init = 0.0
    transl_ref = nn.Parameter(torch.tensor([[X_init, Y_init, Z_init]], device=device))

    # Shoulder span scale constraint (computed from center frame, pose-invariant)
    _sh_l_idx = _SMPLX_IDX.index(16) if 16 in _SMPLX_IDX else None
    _sh_r_idx = _SMPLX_IDX.index(17) if 17 in _SMPLX_IDX else None
    _use_scale_loss = (
        _sh_l_idx is not None and _sh_r_idx is not None
        and vis_mask[_sh_l_idx] and vis_mask[_sh_r_idx]
    )
    _obs_shoulder_dx_px = (
        abs(kp_center_np[_sh_l_idx, 0] - kp_center_np[_sh_r_idx, 0]) * cam['W']
        if _use_scale_loss else 1.0
    )

    def _scale_loss(j2d):
        if not _use_scale_loss:
            return torch.zeros(1, device=device)
        proj_dx = (j2d[_sh_l_idx, 0] - j2d[_sh_r_idx, 0]).abs()
        return ((proj_dx - _obs_shoulder_dx_px) / max(_obs_shoulder_dx_px, 1.0)) ** 2

    def _forward_center(betas, orient, transl, pose_or_z):
        """Forward pass for the center reference frame.
        pose_or_z: either decoded (63-dim) or latent (32-dim) – decoded here if needed."""
        bp  = _decode_pose(pose_or_z) if pose_or_z.shape[-1] == _POSE_DIM else pose_or_z
        out = smplx_model(betas=betas, global_orient=orient, transl=transl, body_pose=bp)
        j2d     = _project(out.joints[0, _SMPLX_IDX], cam)
        loss_kp = _kp_loss(j2d, kp_center, cam)
        return out, j2d, loss_kp

    total_epochs = (cfg.n_warmup_epochs + cfg.n_shape_epochs +
                    cfg.n_poseref_epochs + cfg.n_pose_epochs)

    # ── Phase 1a: Warm-up — orient + transl only ──────────────────────────────
    # Fits the body position/orientation using the center frame PLUS its two
    # temporal neighbours (guaranteed adjacent in the video stream).  Running
    # three forward passes with a shared transl anchors the result to real
    # motion continuity — the single shared position must be plausible for
    # all three consecutive frames simultaneously.
    log.info("Stage 1a warm-up: %d epochs (orient + transl, triplet frames)", cfg.n_warmup_epochs)
    _cb(progress_cb, 0, total_epochs, 0.0, {},
        note=f'Phase 1a: Orient+Transl warm-up ({cfg.n_warmup_epochs} Epochen)…')
    opt_warmup = torch.optim.Adam([glob_orient, transl_ref], lr=cfg.lr_shape)
    for epoch in range(cfg.n_warmup_epochs):
        opt_warmup.zero_grad()
        loss_kp_total = torch.zeros(1, device=device)
        _bp_ref_detached = _decode_pose(z_ref.detach())
        for kp_t in triplet_kp:
            out_t = smplx_model(betas=beta.detach(), global_orient=glob_orient,
                                transl=transl_ref, body_pose=_bp_ref_detached)
            j2d_t = _project(out_t.joints[0, _SMPLX_IDX], cam)
            loss_kp_total = loss_kp_total + _kp_loss(j2d_t, kp_t, cam)
        loss_kp_total = loss_kp_total / len(triplet_kp)
        # Scale constraint from center frame only
        out_c = smplx_model(betas=beta.detach(), global_orient=glob_orient,
                            transl=transl_ref, body_pose=_bp_ref_detached)
        j2d_c = _project(out_c.joints[0, _SMPLX_IDX], cam)
        loss   = cfg.w_keypoint * loss_kp_total + 0.5 * _scale_loss(j2d_c)
        loss.backward()
        opt_warmup.step()
        if epoch % 10 == 0:
            _cb(progress_cb, epoch, total_epochs, float(loss),
                {'keypoint': round(float(loss_kp_total), 4)})

    # ── Phase 1b: Shape — beta + transl, orient + pose FROZEN ────────────────
    # Uses ONLY the center frame from here on.  With orientation already
    # correct, beta receives a clean gradient signal without pose interference.
    log.info("Stage 1b shape: %d epochs (beta + transl, center frame)", cfg.n_shape_epochs)
    _cb(progress_cb, cfg.n_warmup_epochs, total_epochs, 0.0, {},
        note=f'Phase 1b: Shape β ({cfg.n_shape_epochs} Epochen)…')
    glob_orient_frozen = glob_orient.detach()
    opt_shape = torch.optim.Adam([beta, transl_ref], lr=cfg.lr_shape)
    for epoch in range(cfg.n_shape_epochs):
        opt_shape.zero_grad()
        out, j2d, loss_kp = _forward_center(
            beta, glob_orient_frozen, transl_ref, body_pose_ref.detach()
        )
        loss_sp = (beta ** 2).mean()
        loss    = (cfg.w_keypoint    * loss_kp +
                   cfg.w_shape_prior * loss_sp +
                   0.5               * _scale_loss(j2d))
        loss.backward()
        opt_shape.step()
        if epoch % 10 == 0:
            preview = None
            if epoch % 50 == 0:
                preview = _make_preview(frames[center_idx],
                                        out.joints[0, _SMPLX_IDX].detach(), cam)
            _cb(progress_cb, cfg.n_warmup_epochs + epoch, total_epochs, float(loss), {
                'keypoint':    round(float(loss_kp), 4),
                'shape_prior': round(float(loss_sp), 4),
            }, preview_jpg=preview)

    beta_fixed = beta.detach()

    # ── Phase 1c: Pose-ref — body_pose_ref + orient + transl, beta FROZEN ────
    # With shape fixed, find the reference pose that best explains static frames.
    log.info("Stage 1c pose-ref: %d epochs (pose_ref + orient + transl)", cfg.n_poseref_epochs)
    _cb(progress_cb, cfg.n_warmup_epochs + cfg.n_shape_epochs, total_epochs, 0.0, {},
        note=f'Phase 1c: Pose-Ref ({cfg.n_poseref_epochs} Epochen)…')
    glob_orient   = nn.Parameter(glob_orient_frozen.clone())   # unfreeze orient
    opt_poseref   = torch.optim.Adam(
        [z_ref, glob_orient, transl_ref], lr=cfg.lr_shape
    )
    for epoch in range(cfg.n_poseref_epochs):
        opt_poseref.zero_grad()
        out, j2d, loss_kp = _forward_center(
            beta_fixed, glob_orient, transl_ref, z_ref
        )
        loss_pp = z_ref.pow(2).mean()
        loss    = (cfg.w_keypoint   * loss_kp +
                   cfg.w_pose_prior * loss_pp +
                   0.5              * _scale_loss(j2d))
        loss.backward()
        opt_poseref.step()
        if epoch % 10 == 0:
            preview = None
            if epoch % 50 == 0:
                preview = _make_preview(frames[center_idx],
                                        out.joints[0, _SMPLX_IDX].detach(), cam)
            _cb(progress_cb,
                cfg.n_warmup_epochs + cfg.n_shape_epochs + epoch,
                total_epochs, float(loss), {
                    'keypoint':   round(float(loss_kp), 4),
                    'pose_prior': round(float(loss_pp), 4),
                }, preview_jpg=preview)

    # ── Phase 2: pose estimation per frame (batched) ──────────────────────────
    # Need a separate model instance with batch_size=N; smplx bakes batch_size
    # into internal buffers at creation time, so a batch_size=1 model cannot
    # be used for N>1 forward passes without triggering a torch.cat size error.
    smplx_model = _load_smplx(cfg, device, batch_size=N)

    # Initialise per-frame pose from Phase 1 reference (warm start in latent space)
    z_all      = nn.Parameter(z_ref.detach().expand(N, -1).clone())
    orient_all = nn.Parameter(glob_orient.detach().expand(N, -1).clone())
    transl_all = nn.Parameter(transl_ref.detach().expand(N, -1).clone())

    opt_pose = torch.optim.Adam([z_all, orient_all, transl_all], lr=cfg.lr_pose)

    kp_tensor       = torch.tensor(kp_2d, dtype=torch.float32, device=device)   # (N, n_j, 3)
    # Which consecutive frame-pairs are within the same continuous clip.
    # Cross-clip boundaries must NOT get a temporal smoothness penalty.
    temp_valid_mask = torch.tensor(temporal_valid, dtype=torch.bool, device=device)  # (N-1,)
    # Boolean mask: only frames where MediaPipe detected at least one joint.
    # Frames without a person must not contribute to the keypoint loss —
    # their zero-visibility entries would dilute gradients from good frames.
    det_mask   = torch.tensor(frame_detected, dtype=torch.bool, device=device)  # (N,)

    # Outlier weight per frame: frames whose visible-keypoint centroid is far
    # from the median centroid across all frames get down-weighted.
    # This prevents one bad detection (e.g. arms obscuring body) from pulling
    # the per-frame pose to a completely wrong position.
    _kp_xy  = kp_2d[:, :, :2]                          # (N, n_j, 2)
    _kp_vis = kp_2d[:, :, 2] > 0.3                     # (N, n_j) bool
    _cx = np.array([
        float(_kp_xy[i, _kp_vis[i], 0].mean()) if _kp_vis[i].any() else 0.5
        for i in range(N)
    ])
    _cy = np.array([
        float(_kp_xy[i, _kp_vis[i], 1].mean()) if _kp_vis[i].any() else 0.5
        for i in range(N)
    ])
    _med_cx, _med_cy = float(np.median(_cx[frame_detected])), float(np.median(_cy[frame_detected]))
    _dist = np.sqrt((_cx - _med_cx) ** 2 + (_cy - _med_cy) ** 2)
    # Soft weight: 1.0 for frames near median, decays for outliers (sigma ≈ 0.15 normalised)
    _frame_w = np.exp(-(_dist / 0.15) ** 2).astype(np.float32)
    _frame_w[~frame_detected] = 0.0
    frame_weights = torch.tensor(_frame_w, dtype=torch.float32, device=device)  # (N,)
    n_outliers = int((_frame_w < 0.5).sum())
    log.info("Phase 2: %d/%d frames detected, %d outlier frames down-weighted",
             int(det_mask.sum()), N, n_outliers)
    _cb(progress_cb,
        cfg.n_warmup_epochs + cfg.n_shape_epochs + cfg.n_poseref_epochs,
        total_epochs, 0.0, {},
        note=f'Phase 2: Per-Frame Pose ({cfg.n_pose_epochs} Epochen, {int(det_mask.sum())}/{N} Frames)…')

    loss_kp = loss_pp = torch.tensor(0.0)   # init for final quality

    for epoch in range(cfg.n_pose_epochs):
        opt_pose.zero_grad()

        _body_pose = _decode_pose(z_all)   # (N, 63)
        out = smplx_model(
            betas        = beta_fixed.expand(N, -1),
            body_pose    = _body_pose,
            global_orient= orient_all,
            transl       = transl_all,
        )
        # out.joints: (N, n_total_joints, 3)
        # out.A:      (N, 55, 4, 4)

        j2d_batch = _project_batch(out.joints[:, _SMPLX_IDX, :], cam)  # (N, n_j, 2)

        loss_kp  = _kp_loss_batch(j2d_batch, kp_tensor, cam,
                                   frame_mask=det_mask, frame_weights=frame_weights)
        loss_pp  = z_all.pow(2).mean()   # Gaussian prior (latent space if VPoser)

        # Temporal smoothness in latent space (or direct axis-angle without VPoser)
        if N > 1 and temp_valid_mask.any():
            pose_delta   = z_all[1:] - z_all[:-1]             # (N-1, _POSE_DIM)
            orient_delta = orient_all[1:] - orient_all[:-1]   # (N-1, 3)
            loss_temp    = (pose_delta[temp_valid_mask].pow(2).mean() +
                            orient_delta[temp_valid_mask].pow(2).mean())
        else:
            loss_temp = torch.zeros(1, device=device)

        loss = (cfg.w_keypoint   * loss_kp +
                cfg.w_pose_prior * loss_pp +
                cfg.w_temporal   * loss_temp)
        loss.backward()
        opt_pose.step()

        if epoch % 10 == 0:
            preview = None
            if epoch % 50 == 0:
                preview = _make_preview(frames[0],
                                        out.joints[0, _SMPLX_IDX].detach(), cam)
            _cb(progress_cb,
                cfg.n_warmup_epochs + cfg.n_shape_epochs + cfg.n_poseref_epochs + epoch,
                total_epochs, float(loss), {
                'keypoint':   round(float(loss_kp), 4),
                'pose_prior': round(float(loss_pp), 4),
                'temporal':   round(float(loss_temp), 4),
            }, preview_jpg=preview)

    # ── Final forward pass: collect T_bones_t ────────────────────────────────

    with torch.no_grad():
        out_final = smplx_model(
            betas         = beta_fixed.expand(N, -1),
            body_pose     = _decode_pose(z_all.detach()),
            global_orient = orient_all.detach(),
            transl        = transl_all.detach(),
        )

    try:
        T_bones_t = out_final.A.cpu().numpy()    # (N, 55, 4, 4)
    except AttributeError:
        log.warning("output.A not available – computing T_bones_t from joints")
        T_bones_t = _compute_T_bones_from_joints(out_final.joints.detach().cpu().numpy())

    # Evict the large batch=N SMPL-X model from the in-process cache so GPU/RAM
    # is released before save_stage1_result / generate_stage1_previews run.
    # (tasks.py also calls _smplx_cache.clear() in the finally block, but doing
    # it here means the memory is freed before the next allocation, not after.)
    if N > 1:
        stale = [k for k in _smplx_cache if k[3] == N]
        for k in stale:
            del _smplx_cache[k]
        if stale and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stale:
            log.info("Stage 1: evicted batch=%d SMPL-X model from cache", N)

    return Stage1Result(
        beta             = beta_fixed.squeeze(0).cpu().numpy(),
        gender           = cfg.gender,
        theta_t          = _decode_pose(z_all.detach()).cpu().numpy(),
        global_orient_t  = orient_all.detach().cpu().numpy(),
        transl_t         = transl_all.detach().cpu().numpy(),
        T_bones_t        = T_bones_t,
        camera_intrinsics= cam,
        fitting_quality  = {
            'stage1_keypoint_err': round(float(loss_kp), 4),
            'n_frames':            N,
            'n_static_frames':     len(static_idx),
        },
        kp_2d              = kp_2d,   # (N, n_joints, 3) observed keypoints for preview overlay
        person_frame_pairs = person_frame_pairs,
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
    meta['gender']             = result.gender
    meta['camera_intrinsics']  = result.camera_intrinsics
    meta.setdefault('fitting_quality', {}).update(result.fitting_quality)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Pre-generate T-pose mesh so the web container can serve it without smplx
    _save_mesh_obj(result.beta, result.gender, data_path)

    log.info("Stage 1 results saved → %s", data_path)


def _save_mesh_obj(beta: np.ndarray, gender: str, data_path: str) -> None:
    """Generate T-pose SMPL-X mesh and save as mesh.obj."""
    try:
        import torch
        from django.conf import settings

        model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
        cfg_tmp = Stage1Config(gender=gender, num_betas=len(beta))
        model   = _load_smplx(cfg_tmp, torch.device('cpu'), batch_size=1)

        with torch.no_grad():
            out = model(betas=torch.tensor([beta.tolist()], dtype=torch.float32))

        verts = out.vertices[0].numpy()   # (V, 3)
        faces = model.faces               # (F, 3)

        obj_path = os.path.join(data_path, 'mesh.obj')
        with open(obj_path, 'w') as f:
            for v in verts:
                f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

        log.info("mesh.obj saved → %s", obj_path)
    except Exception:
        log.exception("Failed to save mesh.obj — smplx may not be available")


def generate_stage1_previews(result: Stage1Result, avatar, data_path: str,
                              n_previews: int = 9) -> None:
    """
    Render the fitted SMPL-X mesh overlaid on n_previews evenly-spaced video
    frames and save them as previews/preview_00.jpg … preview_08.jpg.
    """
    try:
        import torch
        import smplx as smplx_lib
        # MUST use the same max_frames as the fitting so that frames[i] matches
        # result.kp_2d[i] and result.transl_t[i] etc.
        N_fit = len(result.transl_t)
        frames, _, _, _, _ = _load_person_frames(avatar, max_frames=N_fit)
        if not frames:
            log.warning("generate_stage1_previews: no frames found")
            return

        N = len(frames)
        n = min(n_previews, N)
        indices = [int(i * (N - 1) / max(n - 1, 1)) for i in range(n)]

        from django.conf import settings
        cfg_tmp = Stage1Config(gender=result.gender, num_betas=len(result.beta))
        model   = _load_smplx(cfg_tmp, torch.device('cpu'), batch_size=1)
        faces = model.faces  # (F, 3) numpy int32/int64

        cam_orig = result.camera_intrinsics
        beta_t   = torch.tensor([result.beta.tolist()], dtype=torch.float32)

        preview_dir = os.path.join(data_path, 'previews')
        os.makedirs(preview_dir, exist_ok=True)

        for out_i, fi in enumerate(indices):
            frame = frames[fi].copy()
            H_f, W_f = frame.shape[:2]

            # Scale intrinsics to match the actual (possibly downscaled) frame size.
            # Fitting was done in cam_orig space; rendering must use the same scale.
            sx  = W_f / cam_orig['W']
            sy  = H_f / cam_orig['H']
            cam = {'fx': cam_orig['fx'] * sx, 'fy': cam_orig['fy'] * sy,
                   'cx': cam_orig['cx'] * sx, 'cy': cam_orig['cy'] * sy,
                   'W': W_f, 'H': H_f}

            with torch.no_grad():
                out = model(
                    betas         = beta_t,
                    body_pose     = torch.tensor([result.theta_t[fi].tolist()],
                                                 dtype=torch.float32),
                    global_orient = torch.tensor([result.global_orient_t[fi].tolist()],
                                                 dtype=torch.float32),
                    transl        = torch.tensor([result.transl_t[fi].tolist()],
                                                 dtype=torch.float32),
                )

            verts = out.vertices[0].numpy()   # (V, 3)
            Z     = np.maximum(verts[:, 2], 0.1)
            px    = ( verts[:, 0] / Z * cam['fx'] + cam['cx']).astype(np.int32)
            py    = (-verts[:, 1] / Z * cam['fy'] + cam['cy']).astype(np.int32)  # -Y: SMPL-X Y-up → image Y-down

            # Vectorised backface culling
            v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
            ex = (px[v1] - px[v0]).astype(np.int64)
            ey = (py[v1] - py[v0]).astype(np.int64)
            fx_ = (px[v2] - px[v0]).astype(np.int64)
            fy_ = (py[v2] - py[v0]).astype(np.int64)
            front = (ex * fy_ - ey * fx_) > 0
            z_ok  = (verts[v0, 2] > 0.1) & (verts[v1, 2] > 0.1) & (verts[v2, 2] > 0.1)
            vis_faces = faces[front & z_ok]

            # Draw semi-transparent mesh overlay
            overlay = frame.copy()
            mesh_color = (90, 140, 210)   # BGR warm blue
            for face in vis_faces:
                pts = np.array([[px[face[0]], py[face[0]]],
                                [px[face[1]], py[face[1]]],
                                [px[face[2]], py[face[2]]]], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], mesh_color)

            frame = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

            # Draw observed keypoints (from result.kp_2d) as green dots
            if result.kp_2d is not None and fi < len(result.kp_2d):
                for kp_row in result.kp_2d[fi]:
                    if kp_row[2] > 0.3:
                        cx_ = int(kp_row[0] * W_f)
                        cy_ = int(kp_row[1] * H_f)
                        cv2.circle(frame, (cx_, cy_), 5, (0, 255, 64), -1)

            out_path = os.path.join(preview_dir, f'preview_{out_i:02d}.jpg')
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log.info("Preview %d/%d → %s", out_i + 1, n, out_path)

    except Exception:
        log.exception("generate_stage1_previews failed")


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_person_frames(
    avatar,
    max_frames: int = MAX_FIT_FRAMES,
    load_masks: bool = True,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], Tuple[int, int], np.ndarray]:
    """
    Load BGR frames and binary segmentation masks for all DetectedPerson tracks
    assigned to the avatar's PersonGroup.

    Returns:
        frames         – list of BGR images (≤ max_frames)
        masks          – list of binary masks, same length (None where unavailable)
        (H, W)         – video frame dimensions
        temporal_valid – bool array (len(frames)-1,): True if frames[i] and
                         frames[i+1] are from the same continuous clip and the
                         temporal smoothness loss should be applied between them.
    """
    from core.models import DetectedPerson

    if avatar.group is None:
        raise RuntimeError("Avatar has no PersonGroup assigned.")

    persons = list(avatar.group.persons.select_related('video').all())
    if not persons:
        raise RuntimeError("PersonGroup has no DetectedPerson tracks.")

    H, W = _video_hw(persons[0].video.path)

    # Build flat list of (video_path, frame_idx, mask_path_or_None, track_id).
    # Masks are NOT loaded here to avoid pulling the entire NPZ into RAM
    # before sampling (354 persons × thousands of frames × mask arrays = OOM).
    candidates: List[Tuple[str, int, Optional[str], int]] = []
    for track_id, dp in enumerate(persons):
        mask_path = dp.meta.get('mask_path') or None
        for fi in range(dp.frame_start, dp.frame_end + 1, 3):
            candidates.append((dp.video.path, fi, mask_path, track_id))

    # Evenly sample down to max_frames (preserves temporal order within each track)
    if len(candidates) > max_frames:
        step       = len(candidates) // max_frames
        candidates = candidates[::step][:max_frames]

    if load_masks:
        _needed: Dict[str, set] = {}
        for _, fi, mp, _ in candidates:
            if mp:
                _needed.setdefault(mp, set()).add(fi)
        _mask_cache: Dict[str, Dict[int, np.ndarray]] = {
            mp: _load_mask_lookup(mp, needed_indices=idx_set)
            for mp, idx_set in _needed.items()
        }
    else:
        _mask_cache = {}

    def _get_mask(mask_path: Optional[str], frame_idx: int) -> Optional[np.ndarray]:
        if not mask_path or not load_masks:
            return None
        return _mask_cache.get(mask_path, {}).get(frame_idx)

    frames, masks, track_ids, person_frame_pairs = [], [], [], []
    open_caps: Dict[str, cv2.VideoCapture] = {}

    # On CPU, downscale frames to max 640px wide to reduce RAM usage.
    # MediaPipe and the preview renderer both work at any resolution.
    # Keypoint coords are normalised (0-1) so they are resolution-independent.
    _max_w = 640 if not torch.cuda.is_available() else 0   # 0 = no downscale

    for video_path, frame_idx, mask_path, track_id in candidates:
        cap = open_caps.setdefault(video_path, cv2.VideoCapture(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            if _max_w and frame.shape[1] > _max_w:
                scale = _max_w / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
            masks.append(_get_mask(mask_path, frame_idx))
            track_ids.append(track_id)
            person_frame_pairs.append((persons[track_id].id, frame_idx, video_path))

    for cap in open_caps.values():
        cap.release()

    # temporal_valid[i] = True  ↔  frame i and frame i+1 belong to the same track
    tids = np.array(track_ids, dtype=np.int32)
    temporal_valid = tids[:-1] == tids[1:] if len(tids) > 1 else np.array([], dtype=bool)

    return frames, masks, (H, W), temporal_valid, person_frame_pairs


def _load_mask_lookup(
    mask_path: Optional[str],
    needed_indices: Optional[set] = None,
) -> Dict[int, np.ndarray]:
    """Load a masks .npz file into a {frame_idx: mask} dict.

    Supports two NPZ formats:
    - New (per-frame keys): arrays stored as 'f<frame_idx>' — allows random
      access without decompressing the entire file (avoids 10–30 GB RAM spikes).
    - Legacy (stacked array): 'frame_indices' + 'masks' — loads everything.
      Only used for old files created before the per-frame format was introduced.

    If needed_indices is given, only those frame indices are loaded.
    """
    if not mask_path or not os.path.exists(mask_path):
        return {}
    data = np.load(mask_path)
    result = {}

    # New per-frame format: keys like 'f123'
    # Each key is decompressed independently — no large allocation.
    if any(k.startswith('f') and k[1:].isdigit() for k in data.files):
        for fi in (needed_indices if needed_indices is not None
                   else [int(k[1:]) for k in data.files if k.startswith('f') and k[1:].isdigit()]):
            key = f'f{fi}'
            if key in data.files:
                result[fi] = data[key]
        return result

    # Legacy format: 'frame_indices' + 'masks' stacked array.
    # This loads the entire masks array into RAM — unavoidable with this format.
    if 'masks' in data.files:
        for fi, m in zip(data['frame_indices'], data['masks']):
            fi_int = int(fi)
            if needed_indices is None or fi_int in needed_indices:
                result[fi_int] = m
    return result


def _video_hw(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


# ── SMPL-X ────────────────────────────────────────────────────────────────────

_smplx_cache: dict = {}   # key: (model_dir, gender, num_betas) – one model per config


def _load_smplx(cfg: Stage1Config, device: torch.device, batch_size: int = 1):
    from django.conf import settings
    import smplx

    model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
    key = (model_dir, cfg.gender, cfg.num_betas, batch_size)
    if key not in _smplx_cache:
        # Evict any cached model with the same config but a different batch_size
        # to avoid keeping two large SMPL-X instances in RAM/VRAM simultaneously.
        stale = [k for k in _smplx_cache
                 if k[0] == model_dir and k[1] == cfg.gender and k[2] == cfg.num_betas
                 and k[3] != batch_size]
        for k in stale:
            del _smplx_cache[k]
        _smplx_cache[key] = smplx.create(
            model_path    = model_dir,
            model_type    = 'smplx',
            gender        = cfg.gender,
            num_betas     = cfg.num_betas,
            use_pca       = False,
            flat_hand_mean= True,
            batch_size    = batch_size,
        ).eval()
    return _smplx_cache[key].to(device)


# ── Keypoint extraction ────────────────────────────────────────────────────────

def _extract_keypoints(frames: List[np.ndarray]) -> np.ndarray:
    """
    Run MediaPipe PoseLandmarker (Tasks API, mediapipe >= 0.10) on each frame.

    Returns (N, n_joints, 3) with (x_norm, y_norm, visibility) for each
    joint in _SMPLX_IDX order (zeros for joints without a MediaPipe mapping).
    """
    n_joints = len(_SMPLX_IDX)
    # Map: (slot_in_SMPLX_IDX, mediapipe_landmark_idx) for joints with MP mapping
    _mp_slots = [(i, _SMPLX_TO_MP[smplx_j])
                 for i, smplx_j in enumerate(_SMPLX_IDX) if smplx_j in _SMPLX_TO_MP]

    try:
        import mediapipe as mp
    except ImportError:
        log.warning("mediapipe not available – using zero keypoints")
        return np.zeros((len(frames), n_joints, 3), dtype=np.float32)

    # Same model file used by person_detector.py
    from core.detection.person_detector import _ensure_model, MODEL_PATH
    _ensure_model()

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    kp_all = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as lm_model:
        for frame in frames:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = lm_model.detect(mp_image)
            row      = np.zeros((n_joints, 3), dtype=np.float32)
            if results.pose_landmarks:
                lm = results.pose_landmarks[0]   # first detected person
                for slot, mp_idx in _mp_slots:
                    p = lm[mp_idx]
                    # Out-of-bounds landmarks are extrapolated and unreliable
                    if 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0:
                        row[slot] = [p.x, p.y, p.visibility]
                    # else: row stays zero (visibility=0, ignored in loss)
            kp_all.append(row)

    return np.array(kp_all, dtype=np.float32)   # (N, n_joints, 3)


def _extract_keypoints_db(
    person_frame_pairs: List[Tuple],
) -> Optional[np.ndarray]:
    """
    Load ViTPose COCO-17 keypoints from PersonFrameKeypoints DB cache.
    For frames not yet cached, compute them via combined_analyze and save to DB.

    Returns (N, n_joints, 3) array with (x_norm, y_norm, visibility) for joints
    in _SMPLX_IDX order, or None on complete failure.
    """
    from core.models import PersonFrameKeypoints
    from core.shape_tasks import _compute_keypoints

    n_joints = len(_SMPLX_IDX)
    N        = len(person_frame_pairs)
    result   = np.zeros((N, n_joints, 3), dtype=np.float32)

    # Bulk-fetch cached rows
    person_ids = [str(pid) for pid, _, _ in person_frame_pairs]
    frame_idxs = [fi for _, fi, _ in person_frame_pairs]
    rows = {
        (str(kp.person_id), kp.frame_idx): kp
        for kp in PersonFrameKeypoints.objects.filter(
            person_id__in=set(person_ids),
            frame_idx__in=set(frame_idxs),
        )
    }

    hit_count = computed_count = 0
    for i, (person_id, frame_idx, video_path) in enumerate(person_frame_pairs):
        key = (str(person_id), frame_idx)
        kp  = rows.get(key)

        # On cache miss: compute via pose worker and save for next time
        if kp is None:
            kp_data = _compute_keypoints(video_path, frame_idx)
            if kp_data and kp_data.get('body_landmarks'):
                try:
                    kp = PersonFrameKeypoints.objects.create(
                        person_id      = person_id,
                        frame_idx      = frame_idx,
                        body_landmarks = kp_data['body_landmarks'],
                        rtm_landmarks  = kp_data.get('rtm_landmarks', []),
                        seg_mask_b64   = kp_data.get('seg_mask_b64', ''),
                    )
                    computed_count += 1
                except Exception:
                    pass  # race condition – row might already exist

        if kp is None or not kp.body_landmarks:
            continue

        lm_map = {d['idx']: d for d in kp.body_landmarks}
        for j, smplx_j in enumerate(_SMPLX_IDX):
            if smplx_j in _SMPLX_TO_COCO:
                coco_j = _SMPLX_TO_COCO[smplx_j]
                d = lm_map.get(coco_j)
                if d and d.get('visibility', 0) > 0:
                    x, y = d['x'], d['y']
                    # Discard extrapolated joints outside the visible frame area
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        result[i, j] = [x, y, d['visibility']]
            elif smplx_j in _SMPLX_VIRTUAL:
                # Virtual joint: average of two COCO landmarks
                coco_a, coco_b = _SMPLX_VIRTUAL[smplx_j]
                da = lm_map.get(coco_a)
                db = lm_map.get(coco_b)
                vis_a = da.get('visibility', 0) if da else 0.0
                vis_b = db.get('visibility', 0) if db else 0.0
                mx = (da['x'] + db['x']) / 2 if (vis_a > 0 and vis_b > 0) else (da['x'] if vis_a > 0 else db['x'])
                my = (da['y'] + db['y']) / 2 if (vis_a > 0 and vis_b > 0) else (da['y'] if vis_a > 0 else db['y'])
                mv = min(vis_a, vis_b) if (vis_a > 0 and vis_b > 0) else ((vis_a or vis_b) * 0.5)
                if mv > 0 and 0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0:
                    result[i, j] = [mx, my, mv]
        hit_count += 1

    log.info("_extract_keypoints_db: %d/%d frames  (%d cached, %d newly computed)",
             hit_count, N, hit_count - computed_count, computed_count)
    if hit_count == 0:
        log.warning("No keypoints at all – falling back to MediaPipe")
        return None
    return result


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

def _default_intrinsics(W: int, H: int, focal_scale: float = 1.0) -> dict:
    """Approximate pinhole camera from image dimensions, optionally scaled by Stage 0 result."""
    f = float(max(W, H)) * focal_scale
    return {'fx': f, 'fy': f, 'cx': W / 2.0, 'cy': H / 2.0, 'W': W, 'H': H}


def _project(joints_3d: torch.Tensor, cam: dict) -> torch.Tensor:
    """Project (n_joints, 3) → pixel coords (n_joints, 2).
    SMPL-X canonical Y+ = up; image Y+ = down → negate Y (matches shape_fit.py).
    """
    X, Y, Z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2].clamp(min=0.1)
    return torch.stack([
         X / Z * cam['fx'] + cam['cx'],
        -Y / Z * cam['fy'] + cam['cy'],
    ], dim=-1)


def _project_batch(joints_3d: torch.Tensor, cam: dict) -> torch.Tensor:
    """Project (N, n_joints, 3) → pixel coords (N, n_joints, 2).
    SMPL-X canonical Y+ = up; image Y+ = down → negate Y (matches shape_fit.py).
    """
    X, Y, Z = joints_3d[..., 0], joints_3d[..., 1], joints_3d[..., 2].clamp(min=0.1)
    return torch.stack([
         X / Z * cam['fx'] + cam['cx'],
        -Y / Z * cam['fy'] + cam['cy'],
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


def _kp_loss_batch(j2d: torch.Tensor, kp: torch.Tensor, cam: dict,
                   frame_mask: torch.Tensor = None,
                   frame_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Batched visibility-weighted keypoint loss.

    j2d:           (N, n_joints, 2) pixels
    kp:            (N, n_joints, 3) (x_norm, y_norm, vis)
    frame_mask:    (N,) bool  – frames without detection are excluded entirely
    frame_weights: (N,) float – outlier down-weighting (0..1); if None, uniform
    """
    vis       = kp[..., 2:3].clamp(0.0, 1.0)
    target_px = kp[..., :2] * torch.tensor(
        [[cam['W'], cam['H']]], dtype=j2d.dtype, device=j2d.device
    )
    diff = (j2d - target_px) ** 2
    per_frame = (diff.sum(dim=-1) * vis.squeeze(-1)).mean(dim=-1)  # (N,)

    if frame_weights is not None:
        w = frame_weights
        if frame_mask is not None:
            w = w * frame_mask.float()
        total_w = w.sum().clamp(min=1e-6)
        return (per_frame * w).sum() / total_w
    if frame_mask is not None and frame_mask.any():
        return per_frame[frame_mask].mean()
    return per_frame.mean()


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
    Scales intrinsics to match the actual frame size (which may differ from cam['W'/'H']
    when frames were downscaled on CPU for memory efficiency).
    """
    try:
        H_f, W_f = frame_bgr.shape[:2]
        sx = W_f / cam['W']
        sy = H_f / cam['H']
        cam_f = {'fx': cam['fx']*sx, 'fy': cam['fy']*sy,
                 'cx': cam['cx']*sx, 'cy': cam['cy']*sy,
                 'W': W_f, 'H': H_f}
        j2d = _project(joints_2d_3d.cpu(), cam_f).numpy()
        vis = frame_bgr.copy()
        for x, y in j2d:
            if 0 <= int(x) < W_f and 0 <= int(y) < H_f:
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        scale = min(512 / W_f, 512 / H_f)
        out   = cv2.resize(vis, (int(W_f * scale), int(H_f * scale)))
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
