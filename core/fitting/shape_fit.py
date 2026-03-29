"""
Multi-frame SMPL-X shape fitting for a PersonGroup.

Shared SMPL-X shape (beta, 10 params) across N frames from P clips.
Per-frame: glob_orient, transl, body_pose.

Hip joints
──────────
ViTPose/RTMPose COCO-17 hip landmarks (11 = left_hip, 12 = right_hip) are placed at
the kinematic joint centre (femoral head area), which corresponds directly to SMPL-X
joints 1 & 2.  No correction factor is needed — hips get full weight in the keypoint
loss like all other joints.

The midpoint of the two COCO hips is additionally constrained to SMPL-X pelvis
(joint 0), giving an independent pelvis-centre anchor that helps when one hip is
partially occluded.

A hip/shoulder width ratio constraint shares information across frames.
"""
from __future__ import annotations

import base64
import logging
import os
import random
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ── Joint mapping ─────────────────────────────────────────────────────────────

_COCO_IDX  = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
_SMPLX_IDX = [16, 17, 18, 19, 20, 21,  1,  2,  4,  5,  7,  8]
# All joints get equal weight — ViTPose hips are at the kinematic joint centre.
_WEIGHTS   = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

_K         = len(_COCO_IDX)       # 12
_HIP_L_I   = _COCO_IDX.index(11)  # 6
_HIP_R_I   = _COCO_IDX.index(12)  # 7
_SHO_L_I   = _COCO_IDX.index(5)   # 0
_SHO_R_I   = _COCO_IDX.index(6)   # 1
_ANK_L_I   = _COCO_IDX.index(15)
_ANK_R_I   = _COCO_IDX.index(16)

# Skeleton edges as pairs of indices into _COCO_IDX list
# _COCO_IDX = [5,6,7,8,9,10,11,12,13,14,15,16]
#               0 1 2 3 4  5  6  7  8  9 10 11
_SKEL_EDGES = [
    (0, 1),   # shoulders
    (0, 2), (1, 3),   # shoulder → elbow
    (2, 4), (3, 5),   # elbow → wrist
    (6, 7),   # hips
    (6, 8), (7, 9),   # hip → knee
    (8, 10), (9, 11), # knee → ankle
    (0, 6), (1, 7),   # torso sides
]

_smplx_cache:  dict = {}
_vposer_cache: dict = {}


def _try_load_vposer(device: torch.device):
    """
    Load VPoser V02_05 if the weights directory exists and human_body_prior is installed.
    Returns the VPoser model (eval, on device) or None if unavailable.
    """
    from django.conf import settings as django_settings
    vposer_dir = getattr(django_settings, 'VPOSER_MODEL_DIR', None)
    if not vposer_dir or not os.path.isdir(vposer_dir):
        log.info("VPoser weights not found at %s – using direct body_pose optimisation", vposer_dir)
        return None

    key = str(device)
    if key in _vposer_cache:
        return _vposer_cache[key].to(device)

    vp = _load_vposer_weights(vposer_dir, device)
    _vposer_cache[key] = vp
    log.info("VPoser loaded from %s", vposer_dir)
    return vp


def _get_vposer_class():
    """Import VPoser class, trying multiple package layouts."""
    # GitHub / newer pip version
    try:
        from human_body_prior.models.vposer_model import VPoser
        return VPoser
    except ImportError:
        pass
    # Older PyPI version
    try:
        from human_body_prior.train.vposer_smpl import VPoser
        return VPoser
    except ImportError:
        pass
    raise ImportError("Cannot find VPoser class in human_body_prior – "
                      "install via: pip install git+https://github.com/nghorbani/human_body_prior")


def _load_vposer_weights(vposer_dir: str, device: torch.device):
    """Try multiple loading strategies for different human_body_prior versions."""
    VPoserClass = _get_vposer_class()

    # Strategy 1: load_model helper
    try:
        from human_body_prior.tools.model_loader import load_model
        vp, _ = load_model(vposer_dir, model_code=VPoserClass,
                           remove_words_in_model_weights='vp_model.')
        return vp.eval().to(device)
    except Exception as _e1:
        log.debug("VPoser strategy 1 (load_model) failed: %s", _e1)

    # Strategy 2: load_vposer helper
    try:
        from human_body_prior.tools.model_loader import load_vposer
        vp, _ = load_vposer(vposer_dir)
        return vp.eval().to(device)
    except Exception as _e2:
        log.debug("VPoser strategy 2 (load_vposer) failed: %s", _e2)

    # Strategy 3: manual checkpoint scan + direct weight load
    import glob as _glob
    ckpts = (
        _glob.glob(os.path.join(vposer_dir, 'snapshots', '**', '*.pt'), recursive=True)
        + _glob.glob(os.path.join(vposer_dir, '*.pt'))
    )
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoint found under {vposer_dir}")
    ckpt = sorted(ckpts)[-1]
    log.info("VPoser: loading checkpoint %s", ckpt)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('vp_model.', ''): v for k, v in state.items()}
    vp = VPoserClass(num_neurons=512, latentD=32, data_shape=[1, 21, 3])
    vp.load_state_dict(state, strict=False)
    return vp.eval().to(device)


def _load_smplx(device: torch.device, batch_size: int):
    import smplx
    from django.conf import settings as django_settings
    model_dir = getattr(django_settings, 'SMPLX_MODEL_DIR', 'models')
    key = (model_dir, batch_size)
    if key not in _smplx_cache:
        # Evict any cached model with a different batch_size to avoid
        # accumulating multiple large SMPL-X instances in GPU VRAM.
        stale = [k for k in _smplx_cache if k[0] == model_dir and k[1] != batch_size]
        for k in stale:
            del _smplx_cache[k]
        m = smplx.create(
            model_path=model_dir, model_type='smplx',
            gender='neutral', num_betas=10, batch_size=batch_size,
            use_face_contour=False,
        ).eval()
        _smplx_cache[key] = m
    return _smplx_cache[key].to(device)


def _make_preview(kp_px_n: np.ndarray, j2d_n: np.ndarray,
                  W: int, H: int) -> Optional[str]:
    """
    Render observed (green) vs model (red) keypoints on a dark canvas.

    kp_px_n: (K, 3)  [x_px, y_px, vis]  – observed
    j2d_n:   (K, 2)  [x_px, y_px]       – model projected

    Returns 'data:image/jpeg;base64,...' or None on failure.
    """
    try:
        scale = min(480 / max(W, 1), 270 / max(H, 1), 1.0)
        sw, sh = max(1, int(W * scale)), max(1, int(H * scale))
        canvas = np.full((sh, sw, 3), 28, dtype=np.uint8)

        # Skeleton from observed keypoints (dim green)
        for i, j in _SKEL_EDGES:
            if kp_px_n[i, 2] > 0.3 and kp_px_n[j, 2] > 0.3:
                p1 = (int(kp_px_n[i, 0] * scale), int(kp_px_n[i, 1] * scale))
                p2 = (int(kp_px_n[j, 0] * scale), int(kp_px_n[j, 1] * scale))
                cv2.line(canvas, p1, p2, (0, 100, 30), 1)

        # Skeleton from model keypoints (dim red)
        for i, j in _SKEL_EDGES:
            p1 = (int(j2d_n[i, 0] * scale), int(j2d_n[i, 1] * scale))
            p2 = (int(j2d_n[j, 0] * scale), int(j2d_n[j, 1] * scale))
            cv2.line(canvas, p1, p2, (30, 30, 120), 1)

        # Observed keypoints – green dots
        for px, py, vis in kp_px_n:
            if vis > 0.3:
                x, y = int(px * scale), int(py * scale)
                if 0 <= x < sw and 0 <= y < sh:
                    cv2.circle(canvas, (x, y), 5, (30, 220, 60), -1)
                    cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)

        # Model keypoints – red dots
        for px, py in j2d_n:
            x, y = int(px * scale), int(py * scale)
            if 0 <= x < sw and 0 <= y < sh:
                cv2.circle(canvas, (x, y), 5, (60, 60, 220), -1)
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)

        _, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    except Exception:
        log.debug("_make_preview failed", exc_info=True)
        return None


# ── Main fitting entry point ──────────────────────────────────────────────────

def run_shape_fit(
    frames_data:      list[dict],
    n_phase1_epochs:  int = 300,
    n_phase2_epochs:  int = 500,
    progress_cb=None,
) -> dict:
    """
    Fit shared SMPL-X shape (beta) across N frames.

    frames_data items must have:
        W, H          – frame pixel dimensions
        body_landmarks– ViTPose COCO-17 list of {idx, x, y, visibility}
        clip_id       – str, for temporal loss grouping
        sample_order  – int, position within clip's sample sequence

    Returns dict with betas, focal_scale, kp_loss, n_frames, n_clips.
    """
    N = len(frames_data)
    if N == 0:
        raise ValueError("No frames provided for shape fitting")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("shape_fit: N=%d frames, device=%s", N, device)

    smplx_model = _load_smplx(device, N)

    # ── Build observation tensors ─────────────────────────────────────────────
    kp_px     = np.zeros((N, _K, 3), dtype=np.float32)   # [x_px, y_px, vis]
    pelvis_obs = np.zeros((N, 3),    dtype=np.float32)   # midpoint of hips
    Ws = np.zeros(N, dtype=np.float32)
    Hs = np.zeros(N, dtype=np.float32)

    for n, fd in enumerate(frames_data):
        W, H = int(fd['W']), int(fd['H'])
        Ws[n], Hs[n] = W, H
        lm_map = {d['idx']: d for d in fd.get('body_landmarks', [])}
        for k, coco_idx in enumerate(_COCO_IDX):
            d = lm_map.get(coco_idx)
            if d:
                kp_px[n, k] = [d['x'] * W, d['y'] * H, d['visibility']]
        hl, hr = kp_px[n, _HIP_L_I], kp_px[n, _HIP_R_I]
        vis = min(hl[2], hr[2])
        pelvis_obs[n] = [(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2, vis]

    # Observed hip/shoulder ratio (for the ratio constraint)
    obs_ratios, ratio_valid = [], []
    for n in range(N):
        hl, hr = kp_px[n, _HIP_L_I], kp_px[n, _HIP_R_I]
        sl, sr = kp_px[n, _SHO_L_I], kp_px[n, _SHO_R_I]
        ok = hl[2] > 0.3 and hr[2] > 0.3 and sl[2] > 0.3 and sr[2] > 0.3
        hip_dx = abs(hl[0] - hr[0])
        sho_dx = abs(sl[0] - sr[0])
        if ok and sho_dx > 5:
            obs_ratios.append(hip_dx / sho_dx)
            ratio_valid.append(True)
        else:
            obs_ratios.append(0.0)
            ratio_valid.append(False)

    mean_ratio = float(np.mean([r for r, v in zip(obs_ratios, ratio_valid) if v])) \
                 if any(ratio_valid) else 1.1

    # Base intrinsics (without focal_scale)
    fx_base = np.maximum(Ws, Hs) * 1.2
    fy_base = fx_base.copy()
    cx_base = Ws / 2.0
    cy_base = Hs / 2.0

    # Convert to tensors
    kp_t         = torch.tensor(kp_px,      dtype=torch.float32, device=device)  # (N, K, 3)
    pelvis_t     = torch.tensor(pelvis_obs, dtype=torch.float32, device=device)  # (N, 3)
    w_t          = torch.tensor(_WEIGHTS,   dtype=torch.float32, device=device)  # (K,)
    fx_base_t    = torch.tensor(fx_base,    dtype=torch.float32, device=device)  # (N,)
    fy_base_t    = torch.tensor(fy_base,    dtype=torch.float32, device=device)
    cx_t         = torch.tensor(cx_base,    dtype=torch.float32, device=device)
    cy_t         = torch.tensor(cy_base,    dtype=torch.float32, device=device)
    ratio_mask_t = torch.tensor(ratio_valid, dtype=torch.bool,   device=device)  # (N,)

    # ── Adjacent pairs for temporal loss ─────────────────────────────────────
    adj_pairs: list[tuple[int, int]] = []
    clip_frames: dict[str, list] = defaultdict(list)
    for n, fd in enumerate(frames_data):
        clip_frames[fd.get('clip_id', 'default')].append((fd.get('sample_order', n), n))
    for items in clip_frames.values():
        items.sort()
        for i in range(len(items) - 1):
            adj_pairs.append((items[i][1], items[i + 1][1]))

    # ── Initial depth estimate per frame ─────────────────────────────────────
    z_init = np.full(N, 3.0, dtype=np.float32)
    for n in range(N):
        sho_l = kp_px[n, _SHO_L_I]
        ank_l = kp_px[n, _ANK_L_I]
        ank_r = kp_px[n, _ANK_R_I]
        if sho_l[2] > 0.3 and (ank_l[2] > 0.3 or ank_r[2] > 0.3):
            ank_y = ank_l[1] if ank_l[2] > 0.3 else ank_r[1]
            dy_px = abs(sho_l[1] - ank_y)
            if dy_px > 1:
                # 1.311 m = SMPL-X neutral shoulder(Y=0.085) to ankle(Y=-1.226) distance
                z_init[n] = float(np.clip(fy_base[n] * 1.311 / dy_px, 1.0, 15.0))
        elif kp_px[n, _SHO_R_I][2] > 0.3 and sho_l[2] > 0.3:
            dx_px = abs(sho_l[0] - kp_px[n, _SHO_R_I][0])
            if dx_px > 10:
                # 0.316 m = SMPL-X neutral shoulder width
                z_init[n] = float(np.clip(fx_base[n] * 0.316 / dx_px, 1.0, 15.0))

    # ── VPoser (optional pose prior) ─────────────────────────────────────────
    vposer     = _try_load_vposer(device)
    use_vposer = vposer is not None
    log.info("shape_fit: use_vposer=%s", use_vposer)

    # ── Parameters ───────────────────────────────────────────────────────────
    beta        = nn.Parameter(torch.zeros(1, 10, device=device))
    glob_orient = nn.Parameter(torch.zeros(N, 3, device=device))
    transl      = nn.Parameter(torch.tensor(
        np.column_stack([np.zeros(N), np.zeros(N), z_init]),
        dtype=torch.float32, device=device,
    ))
    # log-space focal scale; init = 1.0
    log_focal   = nn.Parameter(torch.tensor(0.0, device=device))

    if use_vposer:
        # 32-dim latent per frame; VPoser decoder maps to 63-dim body_pose
        pose_z = nn.Parameter(torch.zeros(N, 32, device=device))
    else:
        pose_z = nn.Parameter(torch.zeros(N, 63, device=device))  # direct axis-angle

    total_epochs = n_phase1_epochs + n_phase2_epochs

    # ── Helper closures ───────────────────────────────────────────────────────

    def _fx():
        return fx_base_t * torch.exp(log_focal)   # (N,)

    def _fy():
        return fy_base_t * torch.exp(log_focal)

    def project_batch(j3d):
        """(N, J, 3) → (N, J, 2)  pixel coords with Y-flip."""
        z   = j3d[:, :, 2:3].clamp(min=0.1)
        x2d =  j3d[:, :, 0:1] * _fx().view(N, 1, 1) / z + cx_t.view(N, 1, 1)
        y2d = -j3d[:, :, 1:2] * _fy().view(N, 1, 1) / z + cy_t.view(N, 1, 1)
        return torch.cat([x2d, y2d], dim=2)

    def kp_loss_body(j2d_sel):
        """Weighted 2-D reprojection loss for the 12 body joints."""
        vis  = kp_t[:, :, 2:3]                              # (N, K, 1)
        diff = (j2d_sel - kp_t[:, :, :2]).pow(2).sum(2)    # (N, K)
        return (diff * vis[:, :, 0] * w_t.view(1, _K)).mean()

    def pelvis_loss(j2d_pelvis):
        """Pelvis midpoint (SMPL-X joint 0) vs avg of observed COCO hips."""
        vis  = pelvis_t[:, 2:3]                              # (N, 1)
        diff = (j2d_pelvis - pelvis_t[:, :2]).pow(2).sum(1) # (N,)
        return (diff * vis[:, 0]).mean()

    def ratio_loss(j3d_full):
        """Hip-shoulder width ratio constraint (3-D), using SMPL-X joints directly."""
        if not ratio_mask_t.any():
            return torch.zeros(1, device=device)
        hip_w  = (j3d_full[:, 1] - j3d_full[:, 2]).norm(dim=1)
        sho_w  = (j3d_full[:, 16] - j3d_full[:, 17]).norm(dim=1).clamp(min=1e-4)
        pred_r = hip_w / sho_w
        tgt    = torch.tensor(mean_ratio, dtype=torch.float32, device=device)
        return (pred_r[ratio_mask_t] - tgt).pow(2).mean()

    def _get_body_pose() -> torch.Tensor:
        """Return (N, 63) body_pose, decoded via VPoser if available."""
        if use_vposer:
            decoded = vposer.decode(pose_z)
            bp = decoded.get('pose_body', decoded.get('pose_body_matrot'))
            return bp.reshape(N, 63)
        return pose_z

    def _pose_prior() -> torch.Tensor:
        """Gaussian prior – in latent space if using VPoser, else on raw axis-angle."""
        return pose_z.pow(2).mean()

    def temporal_loss():
        if not adj_pairs:
            return torch.zeros(1, device=device)
        diffs = [(pose_z[i] - pose_z[j]).pow(2).mean() +
                 (glob_orient[i] - glob_orient[j]).pow(2).mean()
                 for i, j in adj_pairs]
        return torch.stack(diffs).mean()

    # Pick one frame index for previews (consistent within a run, random across runs)
    _preview_n = random.randint(0, N - 1)

    def _report(phase_name, ep, total_phase, ep_all, loss_val, j2d_np=None):
        if not progress_cb or ep % 20 != 0:
            return
        preview = None
        if j2d_np is not None and ep % 50 == 0:
            preview = _make_preview(kp_px[_preview_n], j2d_np[_preview_n],
                                    int(Ws[_preview_n]), int(Hs[_preview_n]))
        progress_cb({
            'phase':        phase_name,
            'epoch':        ep,
            'total_phase':  total_phase,
            'epoch_all':    ep_all,
            'total_all':    total_epochs,
            'loss':         round(float(loss_val), 6),
            'focal_scale':  round(float(torch.exp(log_focal).item()), 3),
            'preview_jpg':  preview,
        })

    # ── Phase 1: all params including body_pose ───────────────────────────────
    # Crucially body_pose is NOT frozen: with a zero T-pose, projected joints
    # won't match real video poses, so optimising beta against T-pose produces
    # a shape that compensates for the wrong pose (systematically wrong).
    # Instead we allow body_pose to vary with a pose prior, and use a stronger
    # beta prior so betas stay physically plausible.
    # Shape fit: prior only prevents explosion, not constrain the pose.
    # VPoser latent space is well-conditioned so even a small weight is enough.
    pose_prior_w = 5e-3 if use_vposer else 1e-3

    log.info("shape_fit phase 1: %d epochs  vposer=%s", n_phase1_epochs, use_vposer)
    opt1   = torch.optim.Adam(
        [beta, glob_orient, transl, pose_z, log_focal], lr=5e-3)
    sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=100, gamma=0.3)
    loss   = torch.zeros(1, device=device)

    for ep in range(n_phase1_epochs):
        opt1.zero_grad()
        out    = smplx_model(betas=beta.expand(N, -1),
                             global_orient=glob_orient,
                             transl=transl,
                             body_pose=_get_body_pose())
        j3d    = out.joints[:, :22]                              # (N, 22, 3)
        j2d    = project_batch(j3d[:, _SMPLX_IDX])              # (N, K, 2)
        j2d_p  = project_batch(j3d[:, :1])[:, 0]                # (N, 2) pelvis

        loss = (kp_loss_body(j2d)
                + pelvis_loss(j2d_p)
                + 0.3          * ratio_loss(j3d)
                + 0.5          * beta.pow(2).mean()   # shape stays plausible
                + pose_prior_w * _pose_prior()
                + 0.05         * log_focal.pow(2))
        loss.backward()
        opt1.step()
        sched1.step()
        _report('Shape + Orient', ep, n_phase1_epochs, ep, loss,
                j2d_np=j2d.detach().cpu().numpy())

    log.info("shape_fit phase 1 done: loss=%.5f focal=%.3f",
             float(loss), torch.exp(log_focal).item())

    # ── Phase 2: all parameters including body pose ───────────────────────────
    log.info("shape_fit phase 2: %d epochs  vposer=%s", n_phase2_epochs, use_vposer)
    opt2   = torch.optim.Adam(
        [beta, glob_orient, transl, pose_z, log_focal], lr=2e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=150, gamma=0.3)

    for ep in range(n_phase2_epochs):
        opt2.zero_grad()
        out   = smplx_model(betas=beta.expand(N, -1),
                            global_orient=glob_orient,
                            transl=transl,
                            body_pose=_get_body_pose())
        j3d   = out.joints[:, :22]
        j2d   = project_batch(j3d[:, _SMPLX_IDX])
        j2d_p = project_batch(j3d[:, :1])[:, 0]

        loss = (kp_loss_body(j2d)
                + pelvis_loss(j2d_p)
                + 0.3          * ratio_loss(j3d)
                + 0.5          * beta.pow(2).mean()
                + pose_prior_w * _pose_prior()
                + 0.2          * temporal_loss()
                + 0.05         * log_focal.pow(2))
        loss.backward()
        opt2.step()
        sched2.step()
        _report('Full Pose', ep, n_phase2_epochs, n_phase1_epochs + ep, loss,
                j2d_np=j2d.detach().cpu().numpy())

    final_loss = float(loss.item())
    log.info("shape_fit done: loss=%.5f focal=%.3f",
             final_loss, torch.exp(log_focal).item())

    result = {
        'betas':          beta.detach().cpu().numpy()[0].tolist(),
        'hip_correction': 1.0,   # kept for DB/API compatibility; no longer optimised
        'focal_scale':    round(float(torch.exp(log_focal).item()), 4),
        'kp_loss':        round(final_loss, 6),
        'n_frames':       N,
        'n_clips':        len(clip_frames),
    }

    # Explicitly free GPU tensors and optimizer state; the caller (shape_tasks)
    # will also clear _smplx_cache and call torch.cuda.empty_cache().
    del beta, glob_orient, transl, pose_z, log_focal
    del opt2, sched2, kp_t, pelvis_t, fx_base_t, fy_base_t, cx_t, cy_t
    return result


# ── T-pose render ─────────────────────────────────────────────────────────────

def render_tpose(betas_list: list, W: int = 400, H: int = 512) -> np.ndarray:
    """
    Render SMPL-X T-pose (zero body_pose) from a 3/4-elevated angle.
    Returns BGR numpy array (H, W, 3).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_smplx(device, 1)
    beta   = torch.tensor([betas_list], dtype=torch.float32, device=device)

    with torch.no_grad():
        out   = model(betas=beta,
                      global_orient=torch.zeros(1, 3, device=device),
                      transl=torch.zeros(1, 3, device=device),
                      body_pose=torch.zeros(1, 63, device=device))
        verts = out.vertices[0].cpu().numpy()   # (V, 3)
        faces = model.faces                     # (F, 3)

    # Rotate: yaw +20° then pitch -18° (slight bird's-eye 3/4 view)
    R = _rot_y(20.0) @ _rot_x(-18.0)
    verts = verts @ R.T

    # Centre on body midpoint, scale to fill canvas
    lo, hi    = verts.min(0), verts.max(0)
    centre    = (lo + hi) / 2
    verts    -= centre
    body_h    = float(hi[1] - lo[1])
    scale     = H * 0.85 / max(body_h, 1e-3)

    return _ortho_render(verts, faces, W, H, scale)


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]], dtype=np.float32)


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)


def _ortho_render(verts: np.ndarray, faces: np.ndarray,
                  W: int, H: int, scale: float,
                  bg: tuple = (245, 245, 245)) -> np.ndarray:
    """Orthographic projection + CPU painter's algorithm."""
    cx, cy = W / 2.0, H * 0.52   # slightly below vertical centre

    x2d = ( verts[:, 0] * scale + cx)
    y2d = (-verts[:, 1] * scale + cy)   # Y-flip
    z_v = verts[:, 2]

    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    px0, py0 = x2d[f0], y2d[f0]
    px1, py1 = x2d[f1], y2d[f1]
    px2, py2 = x2d[f2], y2d[f2]

    cross_z = (px1 - px0) * (py2 - py0) - (py1 - py0) * (px2 - px0)

    M = 10
    in_screen = (
        (np.minimum(px0, np.minimum(px1, px2)) < W + M) &
        (np.maximum(px0, np.maximum(px1, px2)) > -M) &
        (np.minimum(py0, np.minimum(py1, py2)) < H + M) &
        (np.maximum(py0, np.maximum(py1, py2)) > -M)
    )
    mask_ccw = in_screen & (cross_z < 0)
    mask_cw  = in_screen & (cross_z > 0)
    vis_mask = mask_ccw if mask_ccw.sum() >= mask_cw.sum() else mask_cw

    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    if vis_mask.sum() == 0:
        return canvas

    sf     = faces[vis_mask]
    mean_z = (z_v[sf[:, 0]] + z_v[sf[:, 1]] + z_v[sf[:, 2]]) / 3
    order  = np.argsort(-mean_z)
    sf     = sf[order]

    x_pts = np.clip(x2d[sf], -32000, 32000).astype(np.int32)
    y_pts = np.clip(y2d[sf], -32000, 32000).astype(np.int32)
    pts   = np.stack([x_pts, y_pts], axis=2)

    skin_bgr = (130, 168, 210)
    cv2.fillPoly(canvas, list(pts), skin_bgr)
    return canvas
