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
    # NOTE: SMPL-X joint 15 is the skull-base, NOT the nose.
    # Adding nose here caused systematic vertical offset → removed.
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
    gender:                     str   = 'neutral'   # 'neutral' | 'male' | 'female'
    max_frames:                 int   = 150          # cap on training frames
    n_warmup_epochs:            int   = 50           # orient+transl only
    n_shape_epochs:             int   = 200          # beta+transl, pose frozen
    n_poseref_epochs:           int   = 100          # pose_ref+orient+transl, beta frozen
    n_pose_epochs:              int   = 300          # per-frame pose (Phase 2)
    lr_shape:                   float = 5e-3
    lr_pose:                    float = 5e-3
    w_keypoint:                 float = 1.0
    w_silhouette:               float = 0.5
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
        gender                    = config_dict.get('gender', 'neutral'),
        max_frames                = config_dict.get('max_frames', 150),
    )
    log.info("Stage 1 on device=%s  num_betas=%d", device, cfg.num_betas)

    smplx_model = _load_smplx(cfg, device, batch_size=1)

    frames, masks, (H, W), temporal_valid = _load_person_frames(avatar, max_frames=cfg.max_frames)
    if not frames:
        raise RuntimeError(
            "No frames found – run person detection first "
            f"(avatar.group={avatar.group})."
        )
    N = len(frames)
    log.info("Stage 1: %d frames  %dx%d", N, W, H)

    cam = _default_intrinsics(W, H)

    _cb(progress_cb, 0,
        cfg.n_warmup_epochs + cfg.n_shape_epochs + cfg.n_poseref_epochs + cfg.n_pose_epochs,
        0.0, {}, note='Extracting 2D keypoints…')

    kp_2d = _extract_keypoints(frames)   # (N, n_joints, 3)  x_norm, y_norm, vis

    # ── Diagnostics: check how many frames had a visible person ──────────────
    mean_vis = float(kp_2d[:, :, 2].mean())
    detected_frames = int((kp_2d[:, :, 2].max(axis=1) > 0.3).sum())
    log.info(
        "Stage 1 keypoints: %d/%d frames with detection, mean visibility=%.3f",
        detected_frames, N, mean_vis,
    )
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

    # Initialise global orientation to 180° around X-axis so the body is upright
    # in camera space.  SMPL-X canonical pose has Y+ = up, but the projection
    # formula uses OpenCV convention (Y+ = down in image).  Without this flip the
    # body projects upside-down and the optimizer cannot recover.
    _pi = torch.tensor([[np.pi, 0.0, 0.0]], dtype=torch.float32, device=device)

    beta          = nn.Parameter(torch.zeros(1, cfg.num_betas, device=device))
    glob_orient   = nn.Parameter(_pi.clone())
    body_pose_ref = nn.Parameter(torch.zeros(1, 63, device=device))

    # Central reference frame: the static frame with highest mean keypoint visibility.
    # All Phase 1b/1c optimisation uses this single frame so that beta and pose_ref
    # are not entangled across frames with differing body positions.
    center_idx = max(static_idx, key=lambda i: float(kp_2d[i, :, 2].mean()))
    kp_center_np = kp_2d[center_idx]                      # (n_j, 3)
    vis_mask     = kp_center_np[:, 2] > 0.4
    kp_center    = torch.tensor(kp_center_np, dtype=torch.float32, device=device)

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
        Z_init = float(np.clip(cam['fy'] * 0.85 / max(dy_px, 1.0), 1.0, 8.0))
        log.info("Stage 1: Z init shoulder-ankle: dy=%.1f px → Z=%.2f m", dy_px, Z_init)
    elif (_shoulder_l is not None and _shoulder_r is not None
            and vis_mask[_shoulder_l] and vis_mask[_shoulder_r]):
        dx_px = abs(kp_center_np[_shoulder_l, 0] - kp_center_np[_shoulder_r, 0]) * cam['W']
        if dx_px > 10:
            Z_init = float(np.clip(cam['fx'] * 0.38 / max(dx_px, 1.0), 1.0, 8.0))
            log.info("Stage 1: Z init shoulder width: dx=%.1f px → Z=%.2f m", dx_px, Z_init)
        else:
            Z_init = 3.0
    else:
        Z_init = 3.0
        log.info("Stage 1: Z init fallback Z=%.2f m", Z_init)

    transl_ref = nn.Parameter(torch.tensor([[0.0, 0.0, Z_init]], device=device))

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

    def _forward_center(betas, orient, transl, pose):
        """Forward pass for the center reference frame."""
        out     = smplx_model(betas=betas, global_orient=orient,
                              transl=transl, body_pose=pose)
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
    opt_warmup = torch.optim.Adam([glob_orient, transl_ref], lr=cfg.lr_shape)
    for epoch in range(cfg.n_warmup_epochs):
        opt_warmup.zero_grad()
        loss_kp_total = torch.zeros(1, device=device)
        for kp_t in triplet_kp:
            out_t = smplx_model(betas=beta.detach(), global_orient=glob_orient,
                                transl=transl_ref, body_pose=body_pose_ref.detach())
            j2d_t = _project(out_t.joints[0, _SMPLX_IDX], cam)
            loss_kp_total = loss_kp_total + _kp_loss(j2d_t, kp_t, cam)
        loss_kp_total = loss_kp_total / len(triplet_kp)
        # Scale constraint from center frame only
        out_c = smplx_model(betas=beta.detach(), global_orient=glob_orient,
                            transl=transl_ref, body_pose=body_pose_ref.detach())
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
    glob_orient   = nn.Parameter(glob_orient_frozen.clone())   # unfreeze orient
    opt_poseref   = torch.optim.Adam(
        [body_pose_ref, glob_orient, transl_ref], lr=cfg.lr_shape
    )
    for epoch in range(cfg.n_poseref_epochs):
        opt_poseref.zero_grad()
        out, j2d, loss_kp = _forward_center(
            beta_fixed, glob_orient, transl_ref, body_pose_ref
        )
        loss_pp = (body_pose_ref ** 2).mean()
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

    # Initialise per-frame pose from Phase 1 reference pose (warm start)
    body_pose  = nn.Parameter(body_pose_ref.detach().expand(N, -1).clone())
    orient_all = nn.Parameter(glob_orient.detach().expand(N, -1).clone())
    transl_all = nn.Parameter(transl_ref.detach().expand(N, -1).clone())

    opt_pose = torch.optim.Adam([body_pose, orient_all, transl_all], lr=cfg.lr_pose)

    kp_tensor       = torch.tensor(kp_2d, dtype=torch.float32, device=device)   # (N, n_j, 3)
    # Which consecutive frame-pairs are within the same continuous clip.
    # Cross-clip boundaries must NOT get a temporal smoothness penalty.
    temp_valid_mask = torch.tensor(temporal_valid, dtype=torch.bool, device=device)  # (N-1,)
    # Boolean mask: only frames where MediaPipe detected at least one joint.
    # Frames without a person must not contribute to the keypoint loss —
    # their zero-visibility entries would dilute gradients from good frames.
    det_mask   = torch.tensor(frame_detected, dtype=torch.bool, device=device)  # (N,)
    log.info("Phase 2: %d/%d frames have detections and will contribute to loss",
             int(det_mask.sum()), N)
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

        loss_kp  = _kp_loss_batch(j2d_batch, kp_tensor, cam, frame_mask=det_mask)
        loss_pp  = (body_pose ** 2).mean()
        loss_sil = _silhouette_loss_batch(out.vertices, smplx_model.faces_tensor,
                                          masks, cam, device) if sil_ok else \
                   torch.zeros(1, device=device)

        # Temporal smoothness: penalise large pose jumps between adjacent frames,
        # but ONLY within the same continuous clip (not across track boundaries).
        if N > 1 and temp_valid_mask.any():
            pose_delta   = body_pose[1:] - body_pose[:-1]      # (N-1, 63)
            orient_delta = orient_all[1:] - orient_all[:-1]    # (N-1, 3)
            loss_temp    = (pose_delta[temp_valid_mask].pow(2).mean() +
                            orient_delta[temp_valid_mask].pow(2).mean())
        else:
            loss_temp = torch.zeros(1, device=device)

        loss = (cfg.w_keypoint   * loss_kp +
                cfg.w_silhouette * loss_sil +
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
                'silhouette': round(float(loss_sil), 4),
                'pose_prior': round(float(loss_pp), 4),
                'temporal':   round(float(loss_temp), 4),
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
        gender           = cfg.gender,
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
        import smplx as smplx_lib
        from django.conf import settings

        model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
        model = smplx_lib.create(
            model_path     = model_dir,
            model_type     = 'smplx',
            gender         = gender,
            num_betas      = len(beta),
            use_pca        = False,
            flat_hand_mean = True,
            batch_size     = 1,
        )
        model.eval()

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
        from django.conf import settings

        frames, _, _, _ = _load_person_frames(avatar)
        if not frames:
            log.warning("generate_stage1_previews: no frames found")
            return

        N = len(frames)
        n = min(n_previews, N)
        indices = [int(i * (N - 1) / max(n - 1, 1)) for i in range(n)]

        model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
        model = smplx_lib.create(
            model_path     = model_dir,
            model_type     = 'smplx',
            gender         = result.gender,
            num_betas      = len(result.beta),
            use_pca        = False,
            flat_hand_mean = True,
            batch_size     = 1,
        )
        model.eval()
        faces = model.faces  # (F, 3) numpy int32/int64

        cam      = result.camera_intrinsics
        beta_t   = torch.tensor([result.beta.tolist()], dtype=torch.float32)

        preview_dir = os.path.join(data_path, 'previews')
        os.makedirs(preview_dir, exist_ok=True)

        for out_i, fi in enumerate(indices):
            frame = frames[fi].copy()
            H_f, W_f = frame.shape[:2]

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
            px    = (verts[:, 0] / Z * cam['fx'] + cam['cx']).astype(np.int32)
            py    = (verts[:, 1] / Z * cam['fy'] + cam['cy']).astype(np.int32)

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

            # Draw fitted 2D keypoints on top
            kp = _extract_keypoints([frames[fi]])[0]   # (n_j, 3)
            for j in range(len(_MP_IDX)):
                if kp[j, 2] > 0.4:
                    cx_ = int(kp[j, 0] * W_f)
                    cy_ = int(kp[j, 1] * H_f)
                    cv2.circle(frame, (cx_, cy_), 4, (0, 255, 128), -1)

            out_path = os.path.join(preview_dir, f'preview_{out_i:02d}.jpg')
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log.info("Preview %d/%d → %s", out_i + 1, n, out_path)

    except Exception:
        log.exception("generate_stage1_previews failed")


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_person_frames(
    avatar,
    max_frames: int = MAX_FIT_FRAMES,
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

    # Build flat list of (video_path, frame_idx, mask_or_None, track_id)
    candidates: List[Tuple[str, int, Optional[np.ndarray], int]] = []
    for track_id, dp in enumerate(persons):
        mask_lookup = _load_mask_lookup(dp.meta.get('mask_path'))
        for fi in range(dp.frame_start, dp.frame_end + 1, 3):
            candidates.append((dp.video.path, fi, mask_lookup.get(fi), track_id))

    # Evenly sample down to max_frames (preserves temporal order within each track)
    if len(candidates) > max_frames:
        step       = len(candidates) // max_frames
        candidates = candidates[::step][:max_frames]

    frames, masks, track_ids = [], [], []
    open_caps: Dict[str, cv2.VideoCapture] = {}

    for video_path, frame_idx, mask, track_id in candidates:
        cap = open_caps.setdefault(video_path, cv2.VideoCapture(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            masks.append(mask)
            track_ids.append(track_id)

    for cap in open_caps.values():
        cap.release()

    # temporal_valid[i] = True  ↔  frame i and frame i+1 belong to the same track
    tids = np.array(track_ids, dtype=np.int32)
    temporal_valid = tids[:-1] == tids[1:] if len(tids) > 1 else np.array([], dtype=bool)

    return frames, masks, (H, W), temporal_valid


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

def _load_smplx(cfg: Stage1Config, device: torch.device, batch_size: int = 1):
    from django.conf import settings
    import smplx

    model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
    model = smplx.create(
        model_path    = model_dir,
        model_type    = 'smplx',
        gender        = cfg.gender,
        num_betas     = cfg.num_betas,
        use_pca       = False,       # full body pose parameters
        flat_hand_mean= True,
        batch_size    = batch_size,
    ).to(device)
    model.eval()
    return model


# ── Keypoint extraction ────────────────────────────────────────────────────────

def _extract_keypoints(frames: List[np.ndarray]) -> np.ndarray:
    """
    Run MediaPipe PoseLandmarker (Tasks API, mediapipe >= 0.10) on each frame.

    Returns (N, n_joints, 3) with (x_norm, y_norm, visibility) for each
    joint in _MP_IDX order (same order as _SMPLX_IDX).
    """
    n_joints = len(_MP_IDX)

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
                for i, mp_idx in enumerate(_MP_IDX):
                    p = lm[mp_idx]
                    # Out-of-bounds landmarks are extrapolated and unreliable
                    if 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0:
                        row[i] = [p.x, p.y, p.visibility]
                    # else: row stays zero (visibility=0, ignored in loss)
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


def _kp_loss_batch(j2d: torch.Tensor, kp: torch.Tensor, cam: dict,
                   frame_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Batched visibility-weighted keypoint loss.

    j2d:        (N, n_joints, 2) pixels
    kp:         (N, n_joints, 3) mediapipe (x_norm, y_norm, vis)
    frame_mask: (N,) bool – only detected frames contribute to the mean.
                Frames without a detected person are excluded entirely so
                they don't dilute the gradient of well-detected frames.
    """
    vis       = kp[..., 2:3].clamp(0.0, 1.0)
    target_px = kp[..., :2] * torch.tensor(
        [[cam['W'], cam['H']]], dtype=j2d.dtype, device=j2d.device
    )
    diff = (j2d - target_px) ** 2
    per_frame = (diff.sum(dim=-1) * vis.squeeze(-1)).mean(dim=-1)  # (N,)

    if frame_mask is not None and frame_mask.any():
        return per_frame[frame_mask].mean()
    return per_frame.mean()


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
