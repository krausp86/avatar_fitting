"""
Single-frame SMPL-X fitting for pose debug overlay.

Fits SMPL-X body pose parameters to COCO-17 2D keypoints (ViTPose output)
and renders the mesh over the original frame using a CPU painter's algorithm.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# Reuse VPoser loader from shape_fit (same cache, no double-load)
def _try_load_vposer(device):
    try:
        from .shape_fit import _try_load_vposer as _sf_load
        return _sf_load(device)
    except Exception:
        return None

log = logging.getLogger(__name__)

# COCO-17 keypoint idx → SMPL-X joint idx
_COCO_TO_SMPLX: dict[int, int] = {
    3:  15,   # left_ear  → head  (back/side visible; pair midpoint ≈ head joint)
    4:  15,   # right_ear → head
    5:  16,   # left_shoulder
    6:  17,   # right_shoulder
    7:  18,   # left_elbow
    8:  19,   # right_elbow
    9:  20,   # left_wrist
    10: 21,   # right_wrist
    11:  1,   # left_hip
    12:  2,   # right_hip
    13:  4,   # left_knee
    14:  5,   # right_knee
    15:  7,   # left_ankle
    16:  8,   # right_ankle
}
_COCO_IDX  = sorted(_COCO_TO_SMPLX)
_SMPLX_IDX = [_COCO_TO_SMPLX[c] for c in _COCO_IDX]

# Skeleton edges for joint overlay (pairs of indices into _SMPLX_IDX list)
_JOINT_ORDER = {smplx_j: i for i, smplx_j in enumerate(_SMPLX_IDX)}
_SKEL_EDGES = [
    (1,  2),  (1,  4),  (2,  5),   # hips + knees
    (4,  7),  (5,  8),             # knees + ankles
    (16, 17), (16, 18), (17, 19),  # shoulders + elbows
    (18, 20), (19, 21),            # elbows + wrists
]

_smplx_cache: dict = {}
_vposer_cache: dict = {}


def _load_vposer(device: torch.device):
    """Load VPoser V02_05 from VPOSER_MODEL_DIR; returns model or None."""
    from django.conf import settings
    vposer_dir = getattr(settings, 'VPOSER_MODEL_DIR', None)
    if not vposer_dir:
        return None
    key = str(vposer_dir)
    if key not in _vposer_cache:
        try:
            from human_body_prior.tools.model_loader import load_model
            from human_body_prior.models.vposer_model import VPoser
            vp, _ = load_model(
                vposer_dir, model_code=VPoser,
                remove_words_in_model_weights='vp_model.',
                disable_grad=True, comp_device='cpu',
            )
            vp = vp.eval()
            _vposer_cache[key] = vp
            log.info("VPoser loaded from %s", vposer_dir)
        except Exception as e:
            log.warning("VPoser not available: %s", e)
            _vposer_cache[key] = None
    vp = _vposer_cache[key]
    return vp.to(device) if vp is not None else None


def _load_smplx(device: torch.device):
    import smplx
    from django.conf import settings
    model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
    if model_dir not in _smplx_cache:
        m = smplx.create(
            model_path=model_dir, model_type='smplx',
            gender='neutral', num_betas=10, batch_size=1,
            use_face_contour=False,
        ).eval()
        _smplx_cache[model_dir] = m
    return _smplx_cache[model_dir].to(device)


def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → axis-angle (3,) via Rodrigues."""
    import cv2 as _cv2
    aa, _ = _cv2.Rodrigues(R)
    return aa.flatten()


def _analytical_init(kp_px, smplx_model, fx, fy, cx, cy, z_init, device):
    """
    Solve PnP to get glob_orient + transl from visible keypoints.

    Given 2D projections (ViTPose) and known 3D structure (SMPL-X neutral pose),
    solvePnP recovers the full 6DOF pose including the torso plane normal —
    without needing to know Z in advance.

    Coordinate systems:
      SMPL-X world: Y-up.  OpenCV camera: Y-down.
      Transform: flip Y in object points before solvePnP,
                 then flip Y back in tvec to get SMPL-X transl.
      glob_orient: R_smplx = D @ R_pnp @ D  where D = diag(1,-1,1).

    Falls back to zero-init when fewer than 4 keypoints are visible.
    """
    import cv2 as _cv2

    zero_orient = torch.zeros(1, 3, device=device)
    zero_transl = torch.tensor([[0.0, 0.0, z_init]], dtype=torch.float32, device=device)

    # ── Neutral-pose joint positions (Y-up, model space) ─────────────────────
    with torch.no_grad():
        neutral = smplx_model(
            betas=torch.zeros(1, 10, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
        )
    j = neutral.joints[0].cpu().numpy()   # (J, 3)

    # ── Collect 2D↔3D correspondences for all visible keypoints ──────────────
    # _COCO_TO_SMPLX maps COCO idx → SMPL-X joint idx
    _COCO_TO_SMPLX_J = {
        5: 16, 6: 17,   # shoulders
        7: 18, 8: 19,   # elbows
        9: 20, 10: 21,  # wrists
        11: 1, 12: 2,   # hips
        13: 4, 14: 5,   # knees
        15: 7, 16: 8,   # ankles
    }
    obj_pts, img_pts = [], []
    for coco_idx, smplx_j in _COCO_TO_SMPLX_J.items():
        if coco_idx not in _COCO_IDX:
            continue
        i = _COCO_IDX.index(coco_idx)
        if kp_px[i, 2] > 0.3:
            p3d = j[smplx_j].copy().astype(np.float64)
            p3d[1] *= -1   # Y-up → Y-down (OpenCV convention)
            obj_pts.append(p3d)
            img_pts.append([float(kp_px[i, 0]), float(kp_px[i, 1])])

    if len(obj_pts) < 4:
        log.info("pnp_init: only %d visible points – using zero init", len(obj_pts))
        return zero_orient, zero_transl

    obj_pts = np.array(obj_pts, dtype=np.float64)   # (N, 3)
    img_pts = np.array(img_pts, dtype=np.float64)   # (N, 2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ── Solve PnP ─────────────────────────────────────────────────────────────
    # SQPNP is the most stable solver for 4+ non-coplanar points.
    # Fall back to EPNP if SQPNP is unavailable (older OpenCV).
    for flags in (getattr(_cv2, 'SOLVEPNP_SQPNP', None),
                  _cv2.SOLVEPNP_EPNP):
        if flags is None:
            continue
        ok, rvec, tvec = _cv2.solvePnP(obj_pts, img_pts, K,
                                        np.zeros(4), flags=flags)
        if ok:
            break
    else:
        log.warning("pnp_init: solvePnP failed – using zero init")
        return zero_orient, zero_transl

    rvec = rvec.flatten()
    tvec = tvec.flatten()

    # ── Convert back to SMPL-X convention (Y-up) ─────────────────────────────
    # D = diag(1,-1,1);  R_smplx = D @ R_pnp @ D
    R_pnp, _ = _cv2.Rodrigues(rvec)
    D = np.diag([1.0, -1.0, 1.0])
    R_smplx = D @ R_pnp @ D
    aa_smplx = _rotmat_to_aa(R_smplx.astype(np.float32))

    t_smplx = tvec.copy()
    t_smplx[1] *= -1   # flip Y back

    log.info("pnp_init: %d pts  aa=(%.2f,%.2f,%.2f)  t=(%.2f,%.2f,%.2f)",
             len(obj_pts), *aa_smplx.tolist(), *t_smplx.tolist())

    orient_t = torch.tensor(aa_smplx[None], dtype=torch.float32, device=device)
    transl_t = torch.tensor(t_smplx[None],  dtype=torch.float32, device=device)
    return orient_t, transl_t


def fit_and_render(
    frame_bgr:       np.ndarray,
    body_landmarks:  list[dict],
    intrinsics:      Optional[dict] = None,
    seg_mask:        Optional[np.ndarray] = None,
    fixed_betas:     Optional[list] = None,
    romp_init:       Optional[dict] = None,
    n_orient_epochs: int = 600,
    n_pose_epochs:   int = 900,
    progress_cb=None,
) -> tuple[np.ndarray, dict]:
    """
    Fit SMPL-X to ViTPose COCO-17 landmarks and render the mesh over frame.

    Args:
        progress_cb: Optional callable(phase, epoch, total_epochs, loss)
                     called every 20 epochs for progress reporting.
    Returns:
        (rendered_bgr, quality_dict)
    """
    H, W = frame_bgr.shape[:2]

    if intrinsics is None:
        f = max(W, H) * 1.2
        intrinsics = {'fx': f, 'fy': f, 'cx': W / 2.0, 'cy': H / 2.0}

    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smplx_model = _load_smplx(device)

    # Build keypoint tensor (12, 3) = [x_px, y_px, visibility]
    lm_map = {d['idx']: d for d in body_landmarks}
    kp_px = np.zeros((len(_COCO_IDX), 3), dtype=np.float32)
    for i, coco_idx in enumerate(_COCO_IDX):
        d = lm_map.get(coco_idx)
        if d:
            kp_px[i] = [d['x'] * W, d['y'] * H, d['visibility']]

    n_visible = int((kp_px[:, 2] > 0.3).sum())
    log.info("single_frame_fit: %d/%d keypoints visible, device=%s", n_visible, len(_COCO_IDX), device)

    kp = torch.tensor(kp_px, dtype=torch.float32, device=device)
    if fixed_betas is not None and len(fixed_betas) == 10:
        beta = torch.tensor([fixed_betas], dtype=torch.float32, device=device)
        log.info("single_frame_fit: using fixed PersonShape betas")
    elif romp_init is not None:
        beta = torch.tensor([romp_init['beta'][:10]], dtype=torch.float32, device=device)
        log.info("single_frame_fit: beta init from ROMP")
    else:
        beta = torch.zeros(1, 10, device=device)

    vposer = _load_vposer(device)
    log.info("single_frame_fit: VPoser=%s", "loaded" if vposer is not None else "unavailable")

    z_init = _estimate_z(kp_px, fx, fy, W, H)
    log.info("single_frame_fit: Z_init=%.2f m, device=%s", z_init, device)

    pnp_orient, transl_init = _analytical_init(kp_px, smplx_model, fx, fy, cx, cy, z_init, device)
    log.info("single_frame_fit: analytical init transl=(%.2f, %.2f, %.2f)",
             *transl_init.flatten().tolist())

    _I_LSHO, _I_RSHO = _COCO_IDX.index(5),  _COCO_IDX.index(6)
    _I_LHIP, _I_RHIP = _COCO_IDX.index(11), _COCO_IDX.index(12)

    _torso_vis  = False
    obs_winding = 0.0

    if romp_init is not None:
        # ROMP directly regresses global orientation — more robust than PnP+winding,
        # especially for side-profile and partial-body views where the winding test
        # degenerates. Use ROMP orientation and skip PnP+winding entirely.
        orient_init = torch.tensor(
            romp_init['thetas'][0][0:3], dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_chirality = 0   # winding unknown; chirality_loss disabled below
        log.info("single_frame_fit: orient init from ROMP (skipping PnP+winding)")
    else:
        orient_init = pnp_orient

        def _aa_to_R(aa):
            import cv2 as _cv2
            R, _ = _cv2.Rodrigues(aa.detach().cpu().numpy().flatten())
            return torch.tensor(R, dtype=torch.float32, device=device)
        def _R_to_aa(R):
            import cv2 as _cv2
            aa, _ = _cv2.Rodrigues(R.cpu().numpy())
            return torch.tensor(aa.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

        R_pnp = _aa_to_R(orient_init)
        ax = (R_pnp @ torch.tensor([0., 1., 0.], device=device)).cpu().numpy()
        ax = ax / (np.linalg.norm(ax) + 1e-8)
        R_flip_np = (2.0 * np.outer(ax, ax) - np.eye(3, dtype=np.float32)).astype(np.float32)
        R_flip = torch.tensor(R_flip_np, dtype=torch.float32, device=device)
        orient_flip = _R_to_aa(R_flip @ _aa_to_R(orient_init))
        log.info("single_frame_fit: body up-axis=(%+.2f,%+.2f,%+.2f)", *ax.tolist())

        def _torso_winding(pts_2d):
            lsho = pts_2d[_I_LSHO, :2]
            rsho = pts_2d[_I_RSHO, :2]
            lhip = pts_2d[_I_LHIP, :2]
            v1 = rsho - lsho
            v2 = lhip - lsho
            return float(v1[0] * v2[1] - v1[1] * v2[0])

        _torso_vis = all(kp_px[i, 2] > 0.3 for i in [_I_LSHO, _I_RSHO, _I_LHIP, _I_RHIP])
        obs_winding = 0.0
        if _torso_vis:
            obs_winding = _torso_winding(kp_px)
            with torch.no_grad():
                out_c = smplx_model(betas=beta, global_orient=orient_init,
                                    transl=transl_init,
                                    body_pose=torch.zeros(1, 63, device=device))
                j = out_c.joints[0]
                proj_pts = np.zeros((len(_COCO_IDX), 3), dtype=np.float32)
                for ii, smplx_j in enumerate(_SMPLX_IDX):
                    z_c = max(float(j[smplx_j, 2].item()), 0.1)
                    proj_pts[ii, 0] = float(j[smplx_j, 0].item()) * fx / z_c + cx
                    proj_pts[ii, 1] = -float(j[smplx_j, 1].item()) * fy / z_c + cy
                    proj_pts[ii, 2] = 1.0
            model_winding = _torso_winding(proj_pts)
            obs_front   = obs_winding   < 0
            model_front = model_winding < 0
            log.info("single_frame_fit: torso winding obs=%.1f (%s)  model=%.1f (%s)",
                     obs_winding, 'front' if obs_front else 'back',
                     model_winding, 'front' if model_front else 'back')
            if obs_front != model_front:
                orient_init = orient_flip
                log.info("single_frame_fit: winding mismatch → applying body-axis 180° flip")
        else:
            log.info("single_frame_fit: torso keypoints not fully visible — skipping winding test")

        obs_chirality = int(np.sign(-obs_winding)) if _torso_vis else 0

    glob_orient = nn.Parameter(orient_init)
    transl      = nn.Parameter(transl_init)

    # Single body_pose parameter — no anatomical segments.
    body_pose = nn.Parameter(torch.zeros(1, 63, device=device))

    # Warm-start from ROMP body_pose if available.
    # ROMP theta: (72,) = global_orient(3) | SMPL body_pose(69, joints 1-23)
    # SMPL-X body_pose uses 21 joints (63 dims) — the first 63 of SMPL's 69 match.
    if romp_init is not None:
        with torch.no_grad():
            bp = torch.tensor(romp_init['thetas'][0][3:66], dtype=torch.float32, device=device)
            body_pose.data.copy_(bp[:63].unsqueeze(0))
        log.info("single_frame_fit: body_pose initialized from ROMP")

    # ── Torso-centroid alignment ───────────────────────────────────────────────
    _TORSO_IDX = [_COCO_IDX.index(c) for c in (5, 6, 11, 12)]  # lsho,rsho,lhip,rhip
    with torch.no_grad():
        out_c = smplx_model(betas=beta, global_orient=glob_orient,
                            transl=transl, body_pose=body_pose)
        j = out_c.joints[0]
        torso_vis = [i for i in _TORSO_IDX if kp_px[i, 2] > 0.3]
        if len(torso_vis) >= 2:
            z_c     = j[_SMPLX_IDX, 2].clamp(min=0.1).cpu().numpy()
            proj_x  = j[_SMPLX_IDX, 0].cpu().numpy() * fx / z_c + cx
            proj_y  = -j[_SMPLX_IDX, 1].cpu().numpy() * fy / z_c + cy
            w       = np.array([kp_px[i, 2] if i in torso_vis else 0.0
                                for i in range(len(_COCO_IDX))], dtype=np.float32)
            obs_cx  = (kp_px[:, 0] * w).sum() / w.sum()
            obs_cy  = (kp_px[:, 1] * w).sum() / w.sum()
            mdl_cx  = (proj_x * w).sum() / w.sum()
            mdl_cy  = (proj_y * w).sum() / w.sum()
            z_now   = float(transl[0, 2].item())
            transl[0, 0] += (obs_cx - mdl_cx) * z_now / fx
            transl[0, 1] -= (obs_cy - mdl_cy) * z_now / fy
            log.info("torso centroid shift: Δpx=(%.1f, %.1f) → Δt=(%.3f, %.3f)",
                     obs_cx - mdl_cx, obs_cy - mdl_cy,
                     (obs_cx - mdl_cx) * z_now / fx,
                     -(obs_cy - mdl_cy) * z_now / fy)

    def project(j3d):
        # SMPL-X is Y-up, image coords are Y-down → negate Y in projection
        z = j3d[:, 2:3].clamp(min=0.1)
        return torch.cat([j3d[:, :1] * fx / z + cx,
                         -j3d[:, 1:2] * fy / z + cy], dim=1)

    def kp_loss(j2d, kp_t):
        vis = kp_t[:, 2:3]
        return ((j2d - kp_t[:, :2]) ** 2 * vis).mean()

    # Shoulder-span scale constraint (stabilises depth, same as stage1)
    sho_l_i = _COCO_IDX.index(5)
    sho_r_i = _COCO_IDX.index(6)
    use_scale = kp_px[sho_l_i, 2] > 0.3 and kp_px[sho_r_i, 2] > 0.3
    obs_sho_dx = abs(kp_px[sho_l_i, 0] - kp_px[sho_r_i, 0]) if use_scale else 1.0

    def scale_loss(j2d):
        if not use_scale:
            return torch.zeros(1, device=device)
        pred_dx = (j2d[sho_l_i, 0] - j2d[sho_r_i, 0]).abs()
        return ((pred_dx - obs_sho_dx) / max(obs_sho_dx, 1.0)) ** 2

    # Left-right ordinal loss: penalise leg/shoulder crossing.
    # For each L/R pair: if observed right is to the right of left, the model
    # must also have right to the right — else hinge penalty (relu(-sign * pred_dx)).
    # Only active when both joints of a pair are visible (vis > 0.3).
    _LR_PAIRS = [
        (_COCO_IDX.index(5),  _COCO_IDX.index(6)),   # left_shoulder, right_shoulder
        (_COCO_IDX.index(11), _COCO_IDX.index(12)),   # left_hip,      right_hip
        (_COCO_IDX.index(13), _COCO_IDX.index(14)),   # left_knee,     right_knee
        (_COCO_IDX.index(15), _COCO_IDX.index(16)),   # left_ankle,    right_ankle
    ]
    _lr_signs = []  # (pair_idx, obs_sign) — precomputed from kp_px
    for l_i, r_i in _LR_PAIRS:
        if kp_px[l_i, 2] > 0.3 and kp_px[r_i, 2] > 0.3:
            obs_dx = kp_px[r_i, 0] - kp_px[l_i, 0]
            # Only use when joints are clearly separated — from a side view,
            # L/R joints overlap in X and obs_sign is pure noise.
            if abs(obs_dx) > 20.0:
                _lr_signs.append((l_i, r_i, float(np.sign(obs_dx))))

    def lr_order_loss(j2d):
        if not _lr_signs:
            return torch.zeros(1, device=device)
        total = torch.zeros(1, device=device)
        for l_i, r_i, obs_sign in _lr_signs:
            pred_dx = j2d[r_i, 0] - j2d[l_i, 0]   # positive = right is to the right
            total = total + torch.relu(-obs_sign * pred_dx)
        return total / len(_lr_signs)

    # ── Silhouette loss (CPU path) ────────────────────────────────────────────
    # Project SMPL-X vertices to 2D, sample the MediaPipe mask via differentiable
    # bilinear grid_sample, penalise vertices outside the body mask.
    if seg_mask is not None:
        # Gaussian blur fills mask holes and softens edges — makes the loss robust
        # to small detection errors and gives smoother gradients near boundaries.
        import cv2 as _cv2_sil
        _seg_smooth = _cv2_sil.GaussianBlur(seg_mask.astype(np.float32), (21, 21), 7)
        _sil_gt_cpu = torch.tensor(_seg_smooth, dtype=torch.float32, device=device)
        # Precompute 4D mask once — avoids allocating a new (1,1,H,W) tensor every epoch
        _sil_mask4d = _sil_gt_cpu.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W), no grad
        log.info("single_frame_fit: CPU silhouette loss enabled (mask %dx%d, smoothed)",
                 seg_mask.shape[1], seg_mask.shape[0])
    else:
        _sil_gt_cpu  = None
        _sil_mask4d  = None

    # Fixed random vertex sample indices — same subset every epoch for stable gradients.
    # Using 1000 of 10,475 vertices keeps the backward graph small enough for CPU.
    _N_SIL_VERTS = 1000
    _sil_vert_idx = torch.randperm(10475)[:_N_SIL_VERTS] if _sil_mask4d is not None else None

    def _sil_loss_cpu(out):
        """CPU silhouette loss — sampled vertices only to keep backward graph manageable."""
        if _sil_mask4d is None:
            return torch.zeros(1, device=device)
        verts  = out.vertices[0, _sil_vert_idx]        # (1000, 3), Y-up
        z      = verts[:, 2:3].clamp(min=0.1)
        x_px   =  verts[:, :1] * fx / z + cx          # (1000, 1)
        y_px   = -verts[:, 1:2] * fy / z + cy
        x_n    = x_px / (W / 2.0) - 1.0
        y_n    = y_px / (H / 2.0) - 1.0
        grid   = torch.cat([x_n, y_n], dim=1).view(1, 1, -1, 2)
        samp   = nn.functional.grid_sample(
            _sil_mask4d, grid, mode='bilinear', align_corners=False,
            padding_mode='zeros').view(-1)              # (1000,) ∈ [0, 1]
        return torch.relu(0.5 - samp).pow(2).mean()

    total_epochs = n_orient_epochs + n_pose_epochs

    def _report(phase_name, ep_phase, total_phase, ep_all, loss_val):
        if progress_cb and ep_phase % 20 == 0:
            progress_cb({
                'phase':       phase_name,
                'epoch':       ep_phase,
                'total_phase': total_phase,
                'epoch_all':   ep_all,
                'total_all':   total_epochs,
                'loss':        round(float(loss_val), 5),
            })

    # Winding loss: keeps the torso-plane orientation locked during Phase 1.
    # Uses the same cross-product as the winding test above.
    # obs_winding < 0 (front-facing): we want cross_z < 0 → penalise cross_z > 0
    # obs_winding > 0 (back-facing):  we want cross_z > 0 → penalise cross_z < 0
    _wind_sign = float(np.sign(obs_winding)) if _torso_vis else 0.0

    def chirality_loss(j2d):
        if _wind_sign == 0.0 or not _torso_vis:
            return torch.zeros(1, device=device)
        v1 = j2d[_I_RSHO] - j2d[_I_LSHO]
        v2 = j2d[_I_LHIP] - j2d[_I_LSHO]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        # penalise when cross_z has the WRONG sign
        return torch.relu(_wind_sign * cross_z)

    # Phase 1: orient + transl + coarse pose.
    # PnP already found a good glob_orient — keep its LR LOW so joints (hips, spine)
    # are forced to express the anatomical bend rather than the optimizer just tilting
    # the entire rigid body.  Hip/spine get high LR to reach large angles (≥ π/2).
    # ── Phase 1: orient + transl ──────────────────────────────────────────────
    _phase1_params = [
        {'params': [transl],    'lr': 5e-3},
        {'params': [body_pose], 'lr': 5e-3},
    ]
    if romp_init is None:
        _phase1_params.append({'params': [glob_orient], 'lr': 1e-3})
    opt1 = torch.optim.Adam(_phase1_params)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=n_orient_epochs, eta_min=1e-4)
    loss = torch.zeros(1, device=device)
    for ep in range(n_orient_epochs):
        opt1.zero_grad()
        out = smplx_model(betas=beta, global_orient=glob_orient,
                          transl=transl, body_pose=body_pose)
        j2d = project(out.joints[0, _SMPLX_IDX])
        loss = (kp_loss(j2d, kp)
                + 0.5  * scale_loss(j2d)
                + 1.0  * lr_order_loss(j2d)
                + 10.0 * chirality_loss(j2d)
                + 0.001 * body_pose.pow(2).mean())
        loss.backward()
        opt1.step()
        sched1.step()
        _report('Phase 1', ep, n_orient_epochs, ep, loss)

    log.info("Phase 1 done: loss=%.4f transl_z=%.3f", loss.item(), transl.detach().cpu()[0, 2])

    # ── Phase 2: full body pose ────────────────────────────────────────────────
    opt2 = torch.optim.Adam([
        {'params': [body_pose],   'lr': 1e-2},
        {'params': [glob_orient], 'lr': 5e-4},
        {'params': [transl],      'lr': 2e-3},
    ])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_pose_epochs, eta_min=1e-5)
    for ep in range(n_pose_epochs):
        opt2.zero_grad()
        out = smplx_model(betas=beta, global_orient=glob_orient,
                          transl=transl, body_pose=body_pose)
        j2d = project(out.joints[0, _SMPLX_IDX])
        loss = (kp_loss(j2d, kp)
                + 0.5   * scale_loss(j2d)
                + 1.0   * lr_order_loss(j2d)
                + 0.001 * body_pose.pow(2).mean()
                + (100.0 * _sil_loss_cpu(out) if _sil_gt_cpu is not None else 0.0))
        loss.backward()
        opt2.step()
        sched2.step()
        _report('Phase 2', ep, n_pose_epochs, n_orient_epochs + ep, loss)

    final_loss = float(loss.item()) if loss.numel() == 1 else 0.0
    log.info("single_frame_fit done: loss=%.5f depth=%.3fm", final_loss,
             transl.detach().cpu()[0, 2])

    if progress_cb:
        progress_cb({
            'phase':       'Rendering',
            'epoch':       n_orient_epochs + n_pose_epochs,
            'total_phase': n_orient_epochs + n_pose_epochs,
            'epoch_all':   n_orient_epochs + n_pose_epochs,
            'total_all':   n_orient_epochs + n_pose_epochs,
            'loss':        round(final_loss, 5),
        })

    with torch.no_grad():
        out_f = smplx_model(betas=beta, global_orient=glob_orient,
                            transl=transl, body_pose=body_pose)
        verts  = out_f.vertices[0].cpu().numpy()
        joints = out_f.joints[0, _SMPLX_IDX].cpu().numpy()

    faces = smplx_model.faces

    rendered = _render_overlay(frame_bgr, verts, faces, joints, kp_px,
                               fx, fy, cx, cy)
    depth_m = round(float(transl.detach().cpu()[0, 2]), 3)

    pose_params = {
        'source':        'smplx',
        'body_pose':     body_pose.detach().cpu().numpy()[0].tolist(),
        'global_orient': glob_orient.detach().cpu().numpy()[0].tolist(),
        'transl':        transl.detach().cpu().numpy()[0].tolist(),
        'betas':         beta.detach().cpu().numpy()[0].tolist(),
        'kp_px':         kp_px.tolist(),
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
    }

    del glob_orient, transl, body_pose, opt2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rendered, {
        'kp_loss':     round(final_loss, 5),
        'depth_m':     depth_m,
        'n_visible':   n_visible,
        'epochs':      n_orient_epochs + n_pose_epochs,
        'pose_params': pose_params,
    }


def _estimate_z(kp_px: np.ndarray, fx, fy, W, H) -> float:
    """
    Estimate depth from all visible keypoint pairs with known 3D distances.

    Uses Euclidean pixel distance (not just X or Y component) so it works
    for any body orientation — standing, bent over, lying, side-on.
    Takes the median over all visible pairs for robustness.
    """
    # (coco_a, coco_b, approx_3d_distance_meters) for an average adult
    _PAIRS = [
        ( 5,  6, 0.38),  # shoulder span
        ( 5, 11, 0.53),  # L shoulder → L hip
        ( 6, 12, 0.53),  # R shoulder → R hip
        (11, 12, 0.28),  # hip span
        ( 5,  7, 0.33),  # L shoulder → L elbow
        ( 6,  8, 0.33),  # R shoulder → R elbow
        ( 7,  9, 0.27),  # L elbow → L wrist
        ( 8, 10, 0.27),  # R elbow → R wrist
        (11, 13, 0.42),  # L hip → L knee
        (12, 14, 0.42),  # R hip → R knee
        (13, 15, 0.40),  # L knee → L ankle
        (14, 16, 0.40),  # R knee → R ankle
    ]
    f_avg = (fx + fy) / 2.0
    estimates = []
    for ca, cb, dist_3d in _PAIRS:
        if ca not in _COCO_IDX or cb not in _COCO_IDX:
            continue
        ia, ib = _COCO_IDX.index(ca), _COCO_IDX.index(cb)
        if kp_px[ia, 2] > 0.3 and kp_px[ib, 2] > 0.3:
            px_dist = float(np.hypot(kp_px[ia, 0] - kp_px[ib, 0],
                                     kp_px[ia, 1] - kp_px[ib, 1]))
            if px_dist > 8.0:   # ignore sub-pixel noise
                estimates.append(f_avg * dist_3d / px_dist)

    if estimates:
        arr = np.array(estimates)
        med = float(np.median(arr))
        # Filter outliers: for bent-over poses, shoulder→hip pairs produce
        # artificially small pixel distances → too-close depth. Remove any
        # estimate more than 2× away from the median before final clamp.
        robust = arr[(arr > 0.4 * med) & (arr < 2.5 * med)]
        z = float(np.clip(np.median(robust) if len(robust) else med, 0.3, 8.0))
        log.info("Z_init: median of %d/%d pair estimates = %.2f m  (range %.2f–%.2f)",
                 len(robust), len(arr), z, float(arr.min()), float(arr.max()))
        return z

    log.warning("Z_init: no visible pairs – fallback 3.0 m")
    return 3.0


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_overlay(frame_bgr, verts, faces, joints_3d, kp_px,
                    fx, fy, cx, cy) -> np.ndarray:
    """Render SMPL-X mesh via CPU painter's algorithm, then draw joints."""
    result = _cpu_mesh_render(frame_bgr, verts, faces, fx, fy, cx, cy)
    _draw_joints(result, joints_3d, kp_px, fx, fy, cx, cy)
    return result


def _cpu_mesh_render(frame_bgr: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Render mesh via painter's algorithm using OpenCV.
    Fast: one cv2.fillPoly call for all visible, depth-sorted triangles.
    """
    H, W = frame_bgr.shape[:2]

    # Project all vertices to pixel space (negate Y: SMPL-X is Y-up, image is Y-down)
    z_v = np.maximum(verts[:, 2], 0.01)
    x2d =  verts[:, 0] * fx / z_v + cx
    y2d = -verts[:, 1] * fy / z_v + cy

    # Per-face values
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    px0, py0 = x2d[f0], y2d[f0]
    px1, py1 = x2d[f1], y2d[f1]
    px2, py2 = x2d[f2], y2d[f2]

    # Back-face culling: 2D cross product (CCW = facing camera in image coords)
    cross_z = (px1 - px0) * (py2 - py0) - (py1 - py0) * (px2 - px0)

    # All vertices in front of camera
    in_front = (z_v[f0] > 0.1) & (z_v[f1] > 0.1) & (z_v[f2] > 0.1)

    # Overlaps with screen (with small margin)
    M = 20
    in_screen = (
        (np.minimum(px0, np.minimum(px1, px2)) < W + M) &
        (np.maximum(px0, np.maximum(px1, px2)) > -M) &
        (np.minimum(py0, np.minimum(py1, py2)) < H + M) &
        (np.maximum(py0, np.maximum(py1, py2)) > -M)
    )

    # SMPL-X meshes have consistent CCW winding when viewed from front.
    # Try both signs; choose the one that gives more visible faces.
    mask_ccw = in_front & in_screen & (cross_z < 0)
    mask_cw  = in_front & in_screen & (cross_z > 0)
    vis_mask = mask_ccw if mask_ccw.sum() >= mask_cw.sum() else mask_cw
    log.debug("mesh render: %d front-facing faces (ccw=%d cw=%d)",
              vis_mask.sum(), mask_ccw.sum(), mask_cw.sum())

    if vis_mask.sum() == 0:
        log.warning("cpu_mesh_render: no visible faces – returning original frame")
        return frame_bgr.copy()

    vis_faces = faces[vis_mask]

    # Depth sort: far to near (painter's algorithm)
    mean_z = (z_v[vis_faces[:, 0]] + z_v[vis_faces[:, 1]] + z_v[vis_faces[:, 2]]) / 3.0
    order  = np.argsort(-mean_z)
    sf = vis_faces[order]

    # Build (N, 3, 2) int32 pixel coords
    x_pts = np.clip(x2d[sf], -32000, 32000).astype(np.int32)  # (N, 3)
    y_pts = np.clip(y2d[sf], -32000, 32000).astype(np.int32)  # (N, 3)
    pts   = np.stack([x_pts, y_pts], axis=2)                  # (N, 3, 2)

    # Draw all depth-sorted faces in one fillPoly call (flat skin color).
    # Painter's algorithm: far → near, later polys overwrite earlier ones.
    overlay = frame_bgr.copy()
    skin_bgr = (130, 168, 210)  # warm blue-grey skin tone in BGR
    cv2.fillPoly(overlay, list(pts), skin_bgr)

    # Wireframe pass: draw edges of front-facing silhouette for definition
    # (only boundary edges – skip for simplicity; outline via edge detection)
    return cv2.addWeighted(frame_bgr, 0.3, overlay, 0.7, 0)


def _draw_joints(img: np.ndarray, joints_3d: np.ndarray, kp_px: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float) -> None:
    """Draw detected (yellow) and fitted SMPL-X joints (blue) + skeleton."""
    # Detected keypoints (yellow)
    for px, py, vis in kp_px:
        if vis > 0.3:
            cv2.circle(img, (int(px), int(py)), 8, (0, 220, 255), -1)
            cv2.circle(img, (int(px), int(py)), 8, (0, 0, 0), 1)

    # Fitted SMPL-X joints
    z   = np.maximum(joints_3d[:, 2], 0.01)
    x2d = ( joints_3d[:, 0] * fx / z + cx).astype(int)
    y2d = (-joints_3d[:, 1] * fy / z + cy).astype(int)

    for smplx_j1, smplx_j2 in _SKEL_EDGES:
        i1 = _JOINT_ORDER[smplx_j1]
        i2 = _JOINT_ORDER[smplx_j2]
        cv2.line(img, (x2d[i1], y2d[i1]), (x2d[i2], y2d[i2]), (0, 80, 220), 2)

    for x, y in zip(x2d, y2d):
        cv2.circle(img, (int(x), int(y)), 5, (0, 60, 200), -1)
        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)


def render_smplx_from_params(
    frame_bgr:    np.ndarray,
    body_pose:    list,   # 63 floats
    global_orient: list,  # 3 floats
    transl:       list,   # 3 floats
    betas:        list,   # 10 floats
    kp_px:        list,   # (J, 3) [[x, y, vis], ...]
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Re-render SMPL-X mesh from pre-fitted (or smoothed) parameters."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smplx_model = _load_smplx(device)

    def _t(data, shape):
        return torch.tensor(data, dtype=torch.float32, device=device).reshape(shape)

    with torch.no_grad():
        out = smplx_model(
            betas=         _t(betas,        (1, 10)),
            global_orient= _t(global_orient,(1, 3)),
            transl=        _t(transl,       (1, 3)),
            body_pose=     _t(body_pose,    (1, 63)),
        )
        verts  = out.vertices[0].cpu().numpy()
        joints = out.joints[0, _SMPLX_IDX].cpu().numpy()

    faces  = smplx_model.faces
    kp_arr = np.array(kp_px, dtype=np.float32)
    return _render_overlay(frame_bgr, verts, faces, joints, kp_arr, fx, fy, cx, cy)
