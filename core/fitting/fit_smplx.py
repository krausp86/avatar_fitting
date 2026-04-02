"""
SOTA SMPL-X Fitting Pipeline (ersetzt stage1.py + shape_fit.py).

Phase A  — Beta (Body Shape):
  Gemeinsamer beta-Vektor über alle Clips/Frames einer PersonGroup.
  Body-Keypoints (COCO-17), GMoF-Robust-Loss, VPoser-Prior, Pinhole-Kamera.
  Initialisierung optional via SMPLer-X Regression.

Phase B  — Pro-Frame Full SMPL-X:
  body_pose (VPoser 32-dim) + global_orient + transl
  + expression (10 FLAME) + jaw_pose (3) + left/right_hand_pose (12 PCA)
  Body- + Face- (68 FLAME lmk) + Hand-Landmarks (21+21 RTMPose).
  Weak-Perspective Kamera pro Frame.

Referenzen:
  GMoF: SMPLify-X (Pavlakos et al., CVPR 2019)
  VPoser: ExPose (Choutas et al., ECCV 2020)
  FLAME landmarks: Li et al., SGA 2017
  MANO PCA hands: Romero et al., SGA 2017
"""
from __future__ import annotations

import base64
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from .keypoint_maps import (
    BODY_SMPLX_IDX, BODY_COCO17_IDX, BODY_VIS_THRESHOLD,
    FACE_RTM_OFFSET, N_FACE_LANDMARKS, FACE_VIS_THRESHOLD,
    LHAND_SMPLX_IDX, LHAND_MP_IDX, LHAND_RTM_IDX,
    RHAND_SMPLX_IDX, RHAND_MP_IDX, RHAND_RTM_IDX,
    HAND_VIS_THRESHOLD,
)

log = logging.getLogger(__name__)

# ── GMoF (Geman-McClure) robust loss ──────────────────────────────────────────

def gmof(squared_error: torch.Tensor, sigma_sq: float) -> torch.Tensor:
    """Geman-McClure robust error function. Down-weights large residuals."""
    return sigma_sq * squared_error / (sigma_sq + squared_error)


# ── Caches ─────────────────────────────────────────────────────────────────────
_smplx_cache:       dict = {}  # (model_dir, batch_size, use_pca) → model
_smplx_pca_cache:   dict = {}  # (model_dir, batch_size) → model with use_pca=True
_vposer_cache:      dict = {}  # device str → vposer model


# ── VPoser loading ─────────────────────────────────────────────────────────────

def _get_vposer_class():
    try:
        from human_body_prior.models.vposer_model import VPoser
        return VPoser
    except ImportError:
        pass
    try:
        from human_body_prior.train.vposer_smpl import VPoser
        return VPoser
    except ImportError:
        pass
    raise ImportError("VPoser not found – install human_body_prior")


def _load_vposer_weights(vposer_dir: str, device: torch.device):
    VPoserClass = _get_vposer_class()
    try:
        from human_body_prior.tools.model_loader import load_model
        vp, _ = load_model(vposer_dir, model_code=VPoserClass,
                           remove_words_in_model_weights='vp_model.')
        return vp.eval().to(device)
    except Exception:
        pass
    try:
        from human_body_prior.tools.model_loader import load_vposer
        vp, _ = load_vposer(vposer_dir)
        return vp.eval().to(device)
    except Exception:
        pass
    import glob as _glob
    ckpts = (_glob.glob(os.path.join(vposer_dir, 'snapshots', '**', '*.pt'), recursive=True)
             + _glob.glob(os.path.join(vposer_dir, '*.pt')))
    if not ckpts:
        raise FileNotFoundError(f"No VPoser checkpoint in {vposer_dir}")
    ckpt = sorted(ckpts)[-1]
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('vp_model.', ''): v for k, v in state.items()}
    vp = VPoserClass(num_neurons=512, latentD=32, data_shape=[1, 21, 3])
    vp.load_state_dict(state, strict=False)
    return vp.eval().to(device)


def _try_load_vposer(device: torch.device) -> Optional[object]:
    from django.conf import settings as s
    vposer_dir = getattr(s, 'VPOSER_MODEL_DIR', None)
    if not vposer_dir or not os.path.isdir(vposer_dir):
        log.info("VPoser weights not found at '%s' – using direct axis-angle", vposer_dir)
        return None
    key = str(device)
    if key not in _vposer_cache:
        try:
            _vposer_cache[key] = _load_vposer_weights(vposer_dir, device)
            log.info("VPoser loaded from %s", vposer_dir)
        except Exception as e:
            log.warning("VPoser load failed: %s – using direct axis-angle", e)
            return None
    return _vposer_cache.get(key)


# ── SMPL-X loading ─────────────────────────────────────────────────────────────

def _load_smplx_phase_a(device: torch.device, batch_size: int):
    """Phase A: no PCA for hands (full axis-angle), batch_size=N."""
    import smplx
    from django.conf import settings as s
    model_dir = getattr(s, 'SMPLX_MODEL_DIR', 'models')
    key = (model_dir, batch_size, False)
    if key not in _smplx_cache:
        stale = [k for k in _smplx_cache if k[0] == model_dir and k[1] != batch_size]
        for k in stale:
            del _smplx_cache[k]
        m = smplx.create(
            model_path=model_dir, model_type='smplx',
            gender='neutral', num_betas=10, batch_size=batch_size,
            use_pca=False, flat_hand_mean=True,
            use_face_contour=False,
        ).eval()
        _smplx_cache[key] = m
    return _smplx_cache[key].to(device)


def _load_smplx_phase_b(device: torch.device):
    """Phase B: PCA hands (12 components), batch_size=1."""
    import smplx
    from django.conf import settings as s
    model_dir = getattr(s, 'SMPLX_MODEL_DIR', 'models')
    key = (model_dir, 1, True)
    if key not in _smplx_cache:
        m = smplx.create(
            model_path=model_dir, model_type='smplx',
            gender='neutral', num_betas=10, batch_size=1,
            use_pca=True, num_pca_comps=12, flat_hand_mean=True,
            use_face_contour=False,
        ).eval()
        _smplx_cache[key] = m
    return _smplx_cache[key].to(device)


# ── FLAME landmark computation ──────────────────────────────────────────────────

def compute_flame_landmarks(vertices: torch.Tensor,
                             smplx_model) -> Optional[torch.Tensor]:
    """
    Projiziert 68 FLAME/dlib Landmarks via barycentric Interpolation.

    vertices: (B, V, 3)
    smplx_model: muss lmk_faces_idx + lmk_bary_coords haben

    Returns: (B, 68, 3) oder None wenn nicht verfügbar.
    """
    if not hasattr(smplx_model, 'lmk_faces_idx'):
        return None
    try:
        lmk_faces_idx   = smplx_model.lmk_faces_idx    # (68,) int
        lmk_bary_coords = smplx_model.lmk_bary_coords  # (68, 3) float

        # Vertex-Indizes der Dreieck-Ecken pro Landmark
        faces_t = smplx_model.faces_tensor              # (F, 3) int
        tri_idx = faces_t[lmk_faces_idx]                # (68, 3) vertex indices

        # Eckpunkte der Dreiecke: (B, 68, 3, 3)
        tri_verts = vertices[:, tri_idx]

        # Barycentric Interpolation: (B, 68, 3)
        bary = lmk_bary_coords.to(vertices.device)      # (68, 3)
        lmk_3d = (tri_verts * bary[None, :, :, None]).sum(dim=2)
        return lmk_3d
    except Exception as e:
        log.debug("compute_flame_landmarks failed: %s", e)
        return None


# ── Keypoint extraction helpers ────────────────────────────────────────────────

def _extract_body_kps(landmarks: list, W: int, H: int) -> np.ndarray:
    """
    body_landmarks (ViTPose COCO-17) → (K, 3) [x_px, y_px, vis]
    K = len(BODY_COCO17_IDX)
    """
    K = len(BODY_COCO17_IDX)
    out = np.zeros((K, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in landmarks}
    for k, coco_idx in enumerate(BODY_COCO17_IDX):
        d = lm_map.get(coco_idx)
        if d and d.get('visibility', 0) > 0:
            out[k] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


def _extract_face_kps(rtm_landmarks: list, W: int, H: int) -> np.ndarray:
    """
    RTMPose rtm_landmarks (offset 23..90) → (68, 3) [x_px, y_px, vis]
    """
    out = np.zeros((N_FACE_LANDMARKS, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in rtm_landmarks}
    for dlib_i in range(N_FACE_LANDMARKS):
        d = lm_map.get(FACE_RTM_OFFSET + dlib_i)
        if d and d.get('visibility', 0) > FACE_VIS_THRESHOLD:
            out[dlib_i] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


def _extract_hand_kps(rtm_landmarks: list, W: int, H: int,
                       rtm_offset: int) -> np.ndarray:
    """
    RTMPose hand landmarks → (21, 3) [x_px, y_px, vis]
    rtm_offset: 91 for left, 112 for right
    """
    out = np.zeros((21, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in rtm_landmarks}
    for mp_i in range(21):
        d = lm_map.get(rtm_offset + mp_i)
        if d and d.get('visibility', 0) > HAND_VIS_THRESHOLD:
            out[mp_i] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


# ── SMPLer-X Regressions-Init (optional) ──────────────────────────────────────

def _fetch_smplx_init_for_frame(frame_bgr: np.ndarray) -> Optional[dict]:
    """Ruft SMPLer-X Regression vom pose-worker ab. Gibt None zurück falls nicht verfügbar."""
    try:
        from ..detection.backends import smplx_regress
        return smplx_regress(frame_bgr)
    except Exception as e:
        log.debug("SMPLer-X init fetch failed: %s", e)
        return None


# ── Frame data collection ──────────────────────────────────────────────────────

def _collect_frames(group, max_frames: int = 150, stride: int = 3,
                    max_total: Optional[int] = None) -> list[dict]:
    """
    Sammelt Frame-Daten für alle DetectedPersons einer PersonGroup.

    Returns list of dicts mit:
        W, H, frame_idx, person_id, clip_id,
        body_landmarks, rtm_landmarks (falls vorhanden)
    """
    from ..models import PersonFrameKeypoints

    all_frames = []

    for person in group.persons.all():
        video_path = person.video.path
        fps        = person.video.fps or 25.0
        W_str, H_str = (person.video.resolution or '1920x1080').split('x')
        W, H = int(W_str), int(H_str)

        # Alle gecachten Keypoints für diese Person
        kps_qs = (PersonFrameKeypoints.objects
                  .filter(person=person)
                  .order_by('frame_idx'))

        # Stride-Sampling
        kps_list = list(kps_qs)
        if len(kps_list) > max_frames:
            step = max(1, len(kps_list) // max_frames)
            kps_list = kps_list[::step][:max_frames]

        for i, kp in enumerate(kps_list):
            all_frames.append({
                'person_id':       str(person.id),
                'frame_idx':       kp.frame_idx,
                'clip_id':         str(person.id),   # jede Person = eigener Clip
                'sample_order':    i,
                'W':               W,
                'H':               H,
                'body_landmarks':  kp.body_landmarks or [],
                'rtm_landmarks':   kp.rtm_landmarks or [],
                'kp_obj':          kp,
            })

    # Optionales Gesamt-Limit (wichtig für Phase A: Batch läuft auf GPU)
    if max_total is not None and len(all_frames) > max_total:
        step = max(1, len(all_frames) // max_total)
        all_frames = all_frames[::step][:max_total]

    log.info("_collect_frames: %d frames aus %d persons (max_per_person=%d, max_total=%s)",
             len(all_frames), group.persons.count(), max_frames, max_total)
    return all_frames


# ── Phase A: Beta Estimation ────────────────────────────────────────────────────

def run_phase_a(
    frames_data:     list[dict],
    smplx_inits:     list[Optional[dict]],  # SMPLer-X init pro Frame (kann None sein)
    n_phase1_epochs: int = 300,
    n_phase2_epochs: int = 200,
    progress_cb=None,
) -> dict:
    """
    Schätzt shared beta (10) über alle Frames.

    Returns: {'betas': list[10], 'focal_scale': float, 'kp_loss': float,
              'n_frames': int, 'per_frame_pose': list[dict]}
    """
    N = len(frames_data)
    if N == 0:
        raise ValueError("Phase A: keine Frames")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Phase A: N=%d frames, device=%s", N, device)

    smplx_model = _load_smplx_phase_a(device, N)
    vposer      = _try_load_vposer(device)
    use_vposer  = vposer is not None
    log.info("Phase A: use_vposer=%s", use_vposer)

    K = len(BODY_COCO17_IDX)

    # ── Beobachtungs-Tensoren aufbauen ─────────────────────────────────────────
    kp_px   = np.zeros((N, K, 3), dtype=np.float32)
    pelv_px = np.zeros((N, 3),    dtype=np.float32)
    Ws      = np.zeros(N, dtype=np.float32)
    Hs      = np.zeros(N, dtype=np.float32)

    for n, fd in enumerate(frames_data):
        W, H = int(fd['W']), int(fd['H'])
        Ws[n], Hs[n] = W, H
        kp_px[n] = _extract_body_kps(fd['body_landmarks'], W, H)
        # Becken = Mittelwert Hüften (BODY_COCO17_IDX-Positionen für Hip L/R)
        hip_l_k = BODY_COCO17_IDX.index(11)
        hip_r_k = BODY_COCO17_IDX.index(12)
        hl, hr  = kp_px[n, hip_l_k], kp_px[n, hip_r_k]
        pelv_px[n] = [(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2, min(hl[2], hr[2])]

    kp_t    = torch.tensor(kp_px,  dtype=torch.float32, device=device)   # (N, K, 3)
    pelv_t  = torch.tensor(pelv_px, dtype=torch.float32, device=device)  # (N, 3)

    # Kamerabasis
    fx_base = np.maximum(Ws, Hs) * 1.2
    fx_base_t = torch.tensor(fx_base, dtype=torch.float32, device=device)
    cx_t      = torch.tensor(Ws / 2.0, dtype=torch.float32, device=device)
    cy_t      = torch.tensor(Hs / 2.0, dtype=torch.float32, device=device)

    # Tiefenschätzung pro Frame (Schulter-Knöchel Abstand)
    sho_l_k = BODY_COCO17_IDX.index(5)
    ank_l_k = BODY_COCO17_IDX.index(15)
    ank_r_k = BODY_COCO17_IDX.index(16)
    z_init  = np.full(N, 3.0, dtype=np.float32)
    for n in range(N):
        sl = kp_px[n, sho_l_k]
        al = kp_px[n, ank_l_k]
        ar = kp_px[n, ank_r_k]
        if sl[2] > 0.3 and (al[2] > 0.3 or ar[2] > 0.3):
            ay = al[1] if al[2] > 0.3 else ar[1]
            dy = abs(sl[1] - ay)
            if dy > 1:
                z_init[n] = float(np.clip(fx_base[n] * 1.311 / dy, 1.0, 15.0))

    # ── Clips für Temporal-Loss ────────────────────────────────────────────────
    adj_pairs: list[tuple[int, int]] = []
    clip_frames: dict[str, list] = defaultdict(list)
    for n, fd in enumerate(frames_data):
        clip_frames[fd.get('clip_id', 'default')].append((fd.get('sample_order', n), n))
    for items in clip_frames.values():
        items.sort()
        for i in range(len(items) - 1):
            adj_pairs.append((items[i][1], items[i + 1][1]))

    # ── Initialisierung ────────────────────────────────────────────────────────
    valid_inits = [x for x in smplx_inits if x is not None]
    if valid_inits:
        beta_init = np.mean([x['beta'] for x in valid_inits], axis=0)
        log.info("Phase A: beta0 aus SMPLer-X (%d frames)", len(valid_inits))
    else:
        beta_init = np.zeros(10, dtype=np.float32)
        log.info("Phase A: beta0 = zeros (kein SMPLer-X)")

    # ── Parameter ─────────────────────────────────────────────────────────────
    beta        = nn.Parameter(torch.tensor(beta_init[None], dtype=torch.float32, device=device))
    glob_orient = nn.Parameter(torch.zeros(N, 3, device=device))
    transl      = nn.Parameter(torch.tensor(
        np.column_stack([np.zeros(N), np.zeros(N), z_init]),
        dtype=torch.float32, device=device,
    ))
    log_focal   = nn.Parameter(torch.tensor(0.0, device=device))
    pose_z      = nn.Parameter(torch.zeros(N, 32 if use_vposer else 63, device=device))

    # Init from SMPLer-X wenn vorhanden
    if valid_inits and use_vposer:
        # Encode body_pose via VPoser → pose_z_init
        try:
            bp_arr = np.array([x['body_pose'] for x in smplx_inits
                               if x is not None], dtype=np.float32)  # (M, 63)
            m_idx  = [i for i, x in enumerate(smplx_inits) if x is not None]
            bp_t   = torch.tensor(bp_arr, dtype=torch.float32, device=device)
            bp_t   = bp_t.view(-1, 21, 3)
            with torch.no_grad():
                enc = vposer.encode(bp_t)
                z_enc = enc.mean if hasattr(enc, 'mean') else enc
            for j, n in enumerate(m_idx):
                if j < len(z_enc):
                    pose_z.data[n] = z_enc[j]
            log.info("Phase A: pose_z init aus VPoser-Encoding")
        except Exception as e:
            log.debug("pose_z VPoser init failed: %s", e)

    # ── Projektions-Helfer ─────────────────────────────────────────────────────
    def _fx():
        return fx_base_t * torch.exp(log_focal)

    def project(j3d):
        """(N, J, 3) → (N, J, 2) pixel, mit Y-Flip."""
        z   = j3d[:, :, 2:3].clamp(min=0.1)
        x2d =  j3d[:, :, 0:1] * _fx().view(N,1,1) / z + cx_t.view(N,1,1)
        y2d = -j3d[:, :, 1:2] * _fx().view(N,1,1) / z + cy_t.view(N,1,1)
        return torch.cat([x2d, y2d], dim=2)

    def get_body_pose():
        if use_vposer:
            dec = vposer.decode(pose_z)
            bp  = dec.get('pose_body', dec.get('pose_body_matrot'))
            return bp.reshape(N, 63)
        return pose_z

    # ── Optimierungs-Schleife ──────────────────────────────────────────────────
    params_p1 = [beta, glob_orient, transl, log_focal]
    params_p2 = params_p1 + [pose_z]

    SIGMA_BODY_SQ = 100.0 ** 2
    W_KP, W_PELV, W_BETA, W_POSE, W_TEMP, W_FOC = 1.0, 0.5, 0.5, 0.003, 0.2, 0.05

    total_epochs = n_phase1_epochs + n_phase2_epochs

    for phase, (params, n_ep) in enumerate(
            [(params_p1, n_phase1_epochs), (params_p2, n_phase2_epochs)]):
        opt = torch.optim.Adam(params, lr=5e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)

        for ep in range(n_ep):
            opt.zero_grad()
            body_pose = get_body_pose()

            with torch.no_grad():
                smplx_model.betas.fill_(0)  # ensure model betas don't interfere

            out = smplx_model(
                betas        = beta.expand(N, -1),
                global_orient= glob_orient,
                body_pose    = body_pose,
                transl       = transl,
            )
            j3d  = out.joints     # (N, 55, 3)
            j2d  = project(j3d)   # (N, 55, 2)

            # SMPL-X joints → COCO-17 Auswahl
            j2d_body = j2d[:, BODY_SMPLX_IDX, :]  # (N, K, 2)
            vis  = kp_t[:, :, 2:3]                 # (N, K, 1)
            r_sq = ((j2d_body - kp_t[:, :, :2]) ** 2).sum(-1)  # (N, K)
            loss_kp = (gmof(r_sq, SIGMA_BODY_SQ) * vis[:, :, 0]).sum() / (vis.sum() + 1e-6)

            # Becken-Anker
            j2d_pelv = j2d[:, 0, :]  # SMPL-X joint 0 = Becken
            pelv_vis  = pelv_t[:, 2:3]
            r_sq_pelv = ((j2d_pelv - pelv_t[:, :2]) ** 2).sum(-1)  # (N,)
            loss_pelv = (gmof(r_sq_pelv, SIGMA_BODY_SQ) * pelv_vis[:, 0]).sum() / (pelv_vis.sum() + 1e-6)

            # Priors
            loss_beta = beta.pow(2).mean()
            loss_pose = pose_z.pow(2).mean()
            loss_foc  = log_focal.pow(2)

            # Temporal Smoothness
            if adj_pairs:
                diffs = [(pose_z[i] - pose_z[j]).pow(2).mean() +
                         (glob_orient[i] - glob_orient[j]).pow(2).mean()
                         for i, j in adj_pairs]
                loss_temp = torch.stack(diffs).mean()
            else:
                loss_temp = torch.zeros(1, device=device)

            loss = (W_KP   * loss_kp   +
                    W_PELV * loss_pelv  +
                    W_BETA * loss_beta  +
                    W_POSE * loss_pose  +
                    W_TEMP * loss_temp  +
                    W_FOC  * loss_foc)

            loss.backward()
            opt.step()
            sched.step()

            ep_all = phase * n_phase1_epochs + ep
            if progress_cb and ep % 25 == 0:
                progress_cb({
                    'phase':       f'A{phase+1}',
                    'epoch':       ep,
                    'epoch_all':   ep_all,
                    'total':       total_epochs,
                    'loss':        float(loss),
                    'loss_terms':  {
                        'kp':   float(loss_kp),
                        'pelv': float(loss_pelv),
                        'beta': float(loss_beta),
                        'pose': float(loss_pose),
                    },
                })
            if ep % 100 == 0:
                log.info("Phase A%d ep %3d/%d  loss=%.4f  kp=%.4f  beta_rms=%.3f",
                         phase+1, ep, n_ep, float(loss), float(loss_kp),
                         float(beta.pow(2).mean().sqrt()))

    # ── Ergebnis sammeln ────────────────────────────────────────────────────────
    with torch.no_grad():
        beta_np  = beta.squeeze(0).cpu().numpy().tolist()
        focal_sc = float(torch.exp(log_focal).item())

        # Per-Frame Pose als Initialisierung für Phase B
        body_pose_np   = get_body_pose().cpu().numpy()         # (N, 63)
        glob_orient_np = glob_orient.cpu().numpy()             # (N, 3)
        transl_np      = transl.cpu().numpy()                  # (N, 3)

    per_frame = []
    for n, fd in enumerate(frames_data):
        per_frame.append({
            'person_id':   fd['person_id'],
            'frame_idx':   fd['frame_idx'],
            'body_pose':   body_pose_np[n].tolist(),
            'global_orient': glob_orient_np[n].tolist(),
            'transl':      transl_np[n].tolist(),
            'W':           fd['W'],
            'H':           fd['H'],
        })

    log.info("Phase A done: beta_rms=%.3f  focal_scale=%.3f",
             float(np.array(beta_np).mean()), focal_sc)

    return {
        'betas':        beta_np,
        'focal_scale':  focal_sc,
        'kp_loss':      float(loss_kp),
        'n_frames':     N,
        'per_frame':    per_frame,
    }


# ── Phase B: Pro-Frame Full SMPL-X ─────────────────────────────────────────────

@dataclass
class FrameFitResult:
    person_id:       str
    frame_idx:       int
    body_pose:       List[float]        # 63
    global_orient:   List[float]        # 3
    transl:          List[float]        # 3
    expression:      List[float]        # 10
    jaw_pose:        List[float]        # 3
    left_hand_pose:  List[float]        # 12 PCA
    right_hand_pose: List[float]        # 12 PCA
    cam_scale:       float
    cam_tx:          float
    cam_ty:          float
    loss_body:       float = 0.0
    loss_face:       float = 0.0
    loss_lhand:      float = 0.0
    loss_rhand:      float = 0.0


def run_phase_b_frame(
    fd:            dict,
    betas:         torch.Tensor,        # (1, 10), requires_grad=False
    smplx_model,                        # Phase-B Modell (use_pca=True)
    vposer,                             # Optional
    phase_a_pose:  dict,                # {'body_pose', 'global_orient', 'transl'}
    smplx_init:    Optional[dict],      # SMPLer-X Regression für diesen Frame
    n_b1_epochs:   int = 100,
    n_b2_epochs:   int = 150,
    device:        torch.device = torch.device('cpu'),
) -> FrameFitResult:
    """
    Optimiert SMPL-X für einen einzelnen Frame.
    B1: Body-only  B2: Body + Face + Hands
    """
    use_vposer = vposer is not None
    W, H = int(fd['W']), int(fd['H'])

    # ── Keypoints laden ───────────────────────────────────────────────────────
    kp_body  = _extract_body_kps(fd['body_landmarks'], W, H)   # (K, 3)
    kp_face  = _extract_face_kps(fd['rtm_landmarks'],  W, H)   # (68, 3)
    kp_lhand = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 91)   # (21, 3)
    kp_rhand = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 112)  # (21, 3)

    kp_body_t  = torch.tensor(kp_body,  dtype=torch.float32, device=device)   # (K, 3)
    kp_face_t  = torch.tensor(kp_face,  dtype=torch.float32, device=device)   # (68, 3)
    kp_lhand_t = torch.tensor(kp_lhand, dtype=torch.float32, device=device)   # (21, 3)
    kp_rhand_t = torch.tensor(kp_rhand, dtype=torch.float32, device=device)   # (21, 3)

    # Sichtbare Punkte checken
    has_face  = kp_face_t[:, 2].sum() > 5
    has_lhand = kp_lhand_t[:, 2].sum() > 3
    has_rhand = kp_rhand_t[:, 2].sum() > 3

    # ── Initialisierung ────────────────────────────────────────────────────────
    init = smplx_init or {}

    def _init_tensor(key, fallback_np, n):
        v = init.get(key)
        if v and len(v) >= n:
            return torch.tensor(v[:n], dtype=torch.float32, device=device)
        return torch.tensor(np.array(fallback_np[:n], dtype=np.float32), device=device)

    bp_init   = phase_a_pose.get('body_pose', [0.0]*63)
    go_init   = phase_a_pose.get('global_orient', [0.0]*3)
    tr_init   = phase_a_pose.get('transl', [0.0, 0.0, 3.0])

    glob_orient  = nn.Parameter(_init_tensor('global_orient', go_init, 3).unsqueeze(0))
    transl_param = nn.Parameter(_init_tensor('transl', tr_init, 3).unsqueeze(0))
    expression   = nn.Parameter(torch.zeros(1, 10, device=device))
    jaw_pose     = nn.Parameter(torch.zeros(1, 3, device=device))
    lhand_pose   = nn.Parameter(torch.zeros(1, 12, device=device))
    rhand_pose   = nn.Parameter(torch.zeros(1, 12, device=device))

    if use_vposer:
        if smplx_init and smplx_init.get('body_pose'):
            try:
                bp_t = torch.tensor(smplx_init['body_pose'][:63], dtype=torch.float32, device=device)
                with torch.no_grad():
                    enc   = vposer.encode(bp_t.view(1, 21, 3))
                    z_enc = enc.mean if hasattr(enc, 'mean') else enc
                pose_z = nn.Parameter(z_enc.detach())
            except Exception:
                pose_z = nn.Parameter(torch.zeros(1, 32, device=device))
        else:
            # Encode Phase-A body_pose
            try:
                bp_t = torch.tensor(bp_init[:63], dtype=torch.float32, device=device)
                with torch.no_grad():
                    enc   = vposer.encode(bp_t.view(1, 21, 3))
                    z_enc = enc.mean if hasattr(enc, 'mean') else enc
                pose_z = nn.Parameter(z_enc.detach())
            except Exception:
                pose_z = nn.Parameter(torch.zeros(1, 32, device=device))
    else:
        pose_z = nn.Parameter(_init_tensor('body_pose', bp_init, 63).unsqueeze(0))

    # Kamera: Weak-Perspective
    # scale ≈ fx / z_body, tx ≈ cx, ty ≈ cy
    z_body = float(transl_param.data[0, 2].item())
    if z_body < 0.5:
        z_body = 3.0
    fx_init   = max(W, H) * 1.2
    scale_init = fx_init / z_body
    cam_scale  = nn.Parameter(torch.tensor([[scale_init]], dtype=torch.float32, device=device))
    cam_tx     = nn.Parameter(torch.tensor([[W / 2.0]],    dtype=torch.float32, device=device))
    cam_ty     = nn.Parameter(torch.tensor([[H / 2.0]],    dtype=torch.float32, device=device))

    def get_body_pose():
        if use_vposer:
            dec = vposer.decode(pose_z)
            bp  = dec.get('pose_body', dec.get('pose_body_matrot'))
            return bp.reshape(1, 63)
        return pose_z

    def project_wp(j3d_1: torch.Tensor) -> torch.Tensor:
        """(1, J, 3) → (J, 2) pixel, Weak-Perspective."""
        j = j3d_1[0]    # (J, 3)
        x_px = cam_scale[0, 0] * j[:, 0] + cam_tx[0, 0]
        y_px = cam_scale[0, 0] * (-j[:, 1]) + cam_ty[0, 0]
        return torch.stack([x_px, y_px], dim=-1)   # (J, 2)

    SIGMA_BODY_SQ  = 100.0 ** 2
    SIGMA_FACE_SQ  = 5.0   ** 2
    SIGMA_HAND_SQ  = 10.0  ** 2

    W_BODY, W_FACE, W_HAND = 1.0, 0.5, 0.3
    W_POSE, W_EXPR, W_JAW, W_HAND_PRIOR = 0.005, 0.01, 0.05, 0.01

    def compute_losses(include_face_hands: bool):
        body_pose = get_body_pose()

        out = smplx_model(
            betas         = betas,
            global_orient = glob_orient,
            body_pose     = body_pose,
            transl        = transl_param,
            expression    = expression,
            jaw_pose      = jaw_pose,
            left_hand_pose = lhand_pose,
            right_hand_pose= rhand_pose,
        )

        j3d = out.joints    # (1, 55, 3)
        j2d = project_wp(j3d)  # (55, 2)

        # Body
        j2d_body = j2d[BODY_SMPLX_IDX]  # (K, 2)
        vis_body = kp_body_t[:, 2]       # (K,)
        r_sq_body = ((j2d_body - kp_body_t[:, :2]) ** 2).sum(-1)  # (K,)
        loss_body = (gmof(r_sq_body, SIGMA_BODY_SQ) * vis_body).sum() / (vis_body.sum() + 1e-6)

        loss_face = loss_lhand = loss_rhand = torch.zeros(1, device=device)[0]

        if include_face_hands:
            # Face via FLAME landmarks
            if has_face:
                verts = out.vertices   # (1, V, 3)
                lmk3d = compute_flame_landmarks(verts, smplx_model)
                if lmk3d is not None:
                    lmk2d = project_wp(lmk3d)     # (N_lmk, 2), N_lmk=51 static FLAME lmks
                    n_lmk = lmk2d.shape[0]
                    # lmk_faces_idx holds only the 51 static FLAME landmarks
                    # (dlib 17-67); dlib 0-16 are dynamic jaw contour → skip them
                    kp_face_sel = kp_face_t[-n_lmk:, :]  # (N_lmk, 3)
                    vis_face = kp_face_sel[:, 2]
                    r_sq_face = ((lmk2d - kp_face_sel[:, :2]) ** 2).sum(-1)
                    loss_face = (gmof(r_sq_face, SIGMA_FACE_SQ) * vis_face).sum() / (vis_face.sum() + 1e-6)

            # Linke Hand
            if has_lhand:
                j2d_lhand = j2d[LHAND_SMPLX_IDX]  # (16, 2)  — incl. wrist
                lhand_mp  = [LHAND_MP_IDX.index(mp) for mp in LHAND_MP_IDX
                             if mp < 21]           # index into 21-pt array
                kp_lhand_sel = kp_lhand_t[LHAND_MP_IDX]  # (16, 3)
                vis_lh = kp_lhand_sel[:, 2]
                r_sq_lh = ((j2d_lhand - kp_lhand_sel[:, :2]) ** 2).sum(-1)
                loss_lhand = (gmof(r_sq_lh, SIGMA_HAND_SQ) * vis_lh).sum() / (vis_lh.sum() + 1e-6)

            # Rechte Hand
            if has_rhand:
                j2d_rhand = j2d[RHAND_SMPLX_IDX]
                kp_rhand_sel = kp_rhand_t[RHAND_MP_IDX]
                vis_rh = kp_rhand_sel[:, 2]
                r_sq_rh = ((j2d_rhand - kp_rhand_sel[:, :2]) ** 2).sum(-1)
                loss_rhand = (gmof(r_sq_rh, SIGMA_HAND_SQ) * vis_rh).sum() / (vis_rh.sum() + 1e-6)

        # Priors
        loss_pose = pose_z.pow(2).mean()
        loss_expr = expression.pow(2).mean()
        loss_jaw  = jaw_pose.pow(2).mean()
        loss_hp   = (lhand_pose.pow(2).mean() + rhand_pose.pow(2).mean()) * 0.5

        total = (W_BODY  * loss_body  +
                 W_FACE  * loss_face  +
                 W_HAND  * (loss_lhand + loss_rhand) +
                 W_POSE  * loss_pose  +
                 W_EXPR  * loss_expr  +
                 W_JAW   * loss_jaw   +
                 W_HAND_PRIOR * loss_hp)

        return total, loss_body, loss_face, loss_lhand, loss_rhand

    # ── B1: Body-only ──────────────────────────────────────────────────────────
    params_b1 = [pose_z, glob_orient, transl_param, cam_scale, cam_tx, cam_ty]
    opt = torch.optim.Adam(params_b1, lr=3e-3)
    for ep in range(n_b1_epochs):
        opt.zero_grad()
        loss, *_ = compute_losses(include_face_hands=False)
        loss.backward()
        opt.step()

    # ── B2: Body + Face + Hands ────────────────────────────────────────────────
    params_b2 = params_b1 + [expression, jaw_pose, lhand_pose, rhand_pose]
    opt2 = torch.optim.Adam(params_b2, lr=2e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=75, gamma=0.3)
    for ep in range(n_b2_epochs):
        opt2.zero_grad()
        loss, l_body, l_face, l_lh, l_rh = compute_losses(include_face_hands=True)
        loss.backward()
        opt2.step()
        sched2.step()

    # ── Ergebnis ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        body_pose_np = get_body_pose().squeeze(0).cpu().numpy().tolist()
        return FrameFitResult(
            person_id      = fd['person_id'],
            frame_idx      = fd['frame_idx'],
            body_pose      = body_pose_np,
            global_orient  = glob_orient.squeeze(0).cpu().numpy().tolist(),
            transl         = transl_param.squeeze(0).cpu().numpy().tolist(),
            expression     = expression.squeeze(0).cpu().numpy().tolist(),
            jaw_pose       = jaw_pose.squeeze(0).cpu().numpy().tolist(),
            left_hand_pose = lhand_pose.squeeze(0).cpu().numpy().tolist(),
            right_hand_pose= rhand_pose.squeeze(0).cpu().numpy().tolist(),
            cam_scale      = float(cam_scale.item()),
            cam_tx         = float(cam_tx.item()),
            cam_ty         = float(cam_ty.item()),
            loss_body      = float(l_body),
            loss_face      = float(l_face),
            loss_lhand     = float(l_lh),
            loss_rhand     = float(l_rh),
        )


# ── Ergebnisse in DB speichern ─────────────────────────────────────────────────

def save_phase_a_result(phase_a: dict, group) -> None:
    """Speichert beta + focal_scale in PersonShape."""
    from ..models import PersonShape
    shape, _ = PersonShape.objects.get_or_create(group=group)
    shape.betas       = phase_a['betas']
    shape.focal_scale = phase_a['focal_scale']
    shape.fit_quality = {
        'kp_loss':    phase_a['kp_loss'],
        'n_frames':   phase_a['n_frames'],
    }
    shape.status = PersonShape.Status.DONE
    from django.utils import timezone
    shape.fitted_at = timezone.now()
    shape.save()
    log.info("Phase A gespeichert: betas_rms=%.3f  focal_scale=%.3f",
             float(np.array(phase_a['betas']).mean()), phase_a['focal_scale'])


def save_phase_b_results(frame_results: list[FrameFitResult]) -> None:
    """Speichert per-Frame SMPL-X Params in PersonFramePose (bulk upsert)."""
    from ..models import PersonFramePose, DetectedPerson

    to_create = []
    to_update = []

    # Existierende Einträge suchen
    person_frame_pairs = {(r.person_id, r.frame_idx) for r in frame_results}
    existing = {
        (str(p.person_id), p.frame_idx): p
        for p in PersonFramePose.objects.filter(
            person_id__in={pid for pid, _ in person_frame_pairs},
        )
    }

    for r in frame_results:
        key = (r.person_id, r.frame_idx)
        pfp = existing.get(key)
        if pfp is None:
            pfp = PersonFramePose(
                person_id=r.person_id,
                frame_idx=r.frame_idx,
            )
            to_create.append(pfp)
        else:
            to_update.append(pfp)

        pfp.body_pose       = r.body_pose
        pfp.global_orient   = r.global_orient
        pfp.transl          = r.transl
        pfp.expression      = r.expression
        pfp.jaw_pose        = r.jaw_pose
        pfp.left_hand_pose  = r.left_hand_pose
        pfp.right_hand_pose = r.right_hand_pose
        pfp.cam_scale       = r.cam_scale
        pfp.cam_tx          = r.cam_tx
        pfp.cam_ty          = r.cam_ty

    update_fields = ['body_pose', 'global_orient', 'transl',
                     'expression', 'jaw_pose', 'left_hand_pose', 'right_hand_pose',
                     'cam_scale', 'cam_tx', 'cam_ty']

    if to_create:
        PersonFramePose.objects.bulk_create(to_create, ignore_conflicts=False)
    if to_update:
        PersonFramePose.objects.bulk_update(to_update, update_fields)

    log.info("Phase B gespeichert: %d create, %d update", len(to_create), len(to_update))


# ── Haupt-Entry-Point ──────────────────────────────────────────────────────────

def run_smplx_fit(avatar, config: dict, progress_cb=None) -> dict:
    """
    Hauptfunktion für die neue SMPL-X Fitting-Pipeline.

    config keys (alle optional):
        max_frames_a:     int (150)  — max. Frames für Phase A
        n_a1_epochs:      int (300)  — Phase A1 Epochen
        n_a2_epochs:      int (200)  — Phase A2 Epochen
        n_b1_epochs:      int (100)  — Phase B1 (body-only)
        n_b2_epochs:      int (150)  — Phase B2 (body+face+hands)
        use_smplx_init:   bool (True) — SMPLer-X Regression als Init
        max_frames_b:     int (200)  — max. Frames für Phase B
    """
    group = avatar.group
    if group is None:
        raise ValueError("Avatar hat keine PersonGroup")

    max_frames_a   = int(config.get('max_frames_a',   150))
    n_a1_epochs    = int(config.get('n_a1_epochs',    300))
    n_a2_epochs    = int(config.get('n_a2_epochs',    200))
    n_b1_epochs    = int(config.get('n_b1_epochs',    100))
    n_b2_epochs    = int(config.get('n_b2_epochs',    150))
    use_smplx_init = bool(config.get('use_smplx_init', True))
    max_frames_b   = int(config.get('max_frames_b',   200))

    def _cb(data):
        if progress_cb:
            progress_cb(
                epoch    = data.get('epoch_all', data.get('epoch', 0)),
                loss_val = data.get('loss', 0.0),
                loss_dict= data.get('loss_terms', {}),
                notes    = data.get('phase', ''),
            )

    # ── Frame-Daten sammeln ────────────────────────────────────────────────────
    log.info("run_smplx_fit: sammle Frames für Gruppe %s", group.id)
    frames = _collect_frames(group, max_frames=max_frames_a)
    if not frames:
        raise ValueError("Keine Frames mit Keypoints in der PersonGroup")

    # ── SMPLer-X Regressions-Init (optional) ──────────────────────────────────
    smplx_inits: list[Optional[dict]] = [None] * len(frames)
    if use_smplx_init:
        import cv2

        n_attempted = 0
        for i, fd in enumerate(frames):
            try:
                frame_bgr = _read_frame_for_fd(fd)
                if frame_bgr is not None:
                    smplx_inits[i] = _fetch_smplx_init_for_frame(frame_bgr)
                    n_attempted += 1
            except Exception as e:
                log.debug("SMPLer-X init frame %d: %s", i, e)
        n_success = sum(1 for x in smplx_inits if x is not None)
        log.info("SMPLer-X init: %d/%d Frames erfolgreich", n_success, n_attempted)

    # ── Phase A: Beta ──────────────────────────────────────────────────────────
    if progress_cb:
        progress_cb(epoch=0, loss_val=0.0, loss_dict={},
                    notes='Phase A: Beta-Schätzung läuft...')

    phase_a = run_phase_a(
        frames_data     = frames,
        smplx_inits     = smplx_inits,
        n_phase1_epochs = n_a1_epochs,
        n_phase2_epochs = n_a2_epochs,
        progress_cb     = _cb,
    )
    save_phase_a_result(phase_a, group)

    # ── Phase B: Pro-Frame Full Fit ────────────────────────────────────────────
    if progress_cb:
        progress_cb(epoch=n_a1_epochs + n_a2_epochs, loss_val=phase_a['kp_loss'],
                    loss_dict={'kp': phase_a['kp_loss']},
                    notes='Phase B: Per-Frame-Fitting startet...')

    # Alle Frames mit RTMPose-Daten sammeln (stride größer für Phase B OK)
    frames_b = _collect_frames(group, max_frames=max_frames_b)
    # Phase-A Pose per (person_id, frame_idx)
    phase_a_by_key = {(p['person_id'], p['frame_idx']): p
                      for p in phase_a['per_frame']}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smplx_b = _load_smplx_phase_b(device)
    vposer  = _try_load_vposer(device)

    betas_t = torch.tensor([phase_a['betas']], dtype=torch.float32, device=device)

    frame_results: list[FrameFitResult] = []
    n_b = len(frames_b)

    for i, fd in enumerate(frames_b):
        key = (fd['person_id'], fd['frame_idx'])
        phase_a_pose = phase_a_by_key.get(key, {
            'body_pose':     [0.0] * 63,
            'global_orient': [0.0] * 3,
            'transl':        [0.0, 0.0, 3.0],
        })

        # SMPLer-X init für diesen Frame (evtl. nochmal fetchen wenn nicht in Phase-A dabei)
        smplx_init = None
        if use_smplx_init:
            try:
                frame_bgr = _read_frame_for_fd(fd)
                if frame_bgr is not None:
                    smplx_init = _fetch_smplx_init_for_frame(frame_bgr)
            except Exception:
                pass

        result = run_phase_b_frame(
            fd          = fd,
            betas       = betas_t,
            smplx_model = smplx_b,
            vposer      = vposer,
            phase_a_pose= phase_a_pose,
            smplx_init  = smplx_init,
            n_b1_epochs = n_b1_epochs,
            n_b2_epochs = n_b2_epochs,
            device      = device,
        )
        frame_results.append(result)

        if progress_cb and i % 10 == 0:
            ep_b  = n_a1_epochs + n_a2_epochs + i * (n_b1_epochs + n_b2_epochs)
            total_ep = ep_b + n_b * (n_b1_epochs + n_b2_epochs)
            progress_cb(
                epoch    = ep_b,
                loss_val = result.loss_body,
                loss_dict= {'body': result.loss_body, 'face': result.loss_face,
                            'lhand': result.loss_lhand, 'rhand': result.loss_rhand},
                notes    = f'Phase B: Frame {i+1}/{n_b}',
            )
        if i % 20 == 0:
            log.info("Phase B: %d/%d  loss_body=%.4f  loss_face=%.4f",
                     i + 1, n_b, result.loss_body, result.loss_face)

    save_phase_b_results(frame_results)

    # Temporal Smoothing der neuen Felder
    try:
        from .pose_smoothing import smooth_new_fields
        smooth_new_fields([r.person_id for r in frame_results])
    except Exception as e:
        log.warning("Temporal smoothing der Phase-B Felder fehlgeschlagen: %s", e)

    if progress_cb:
        progress_cb(epoch=n_a1_epochs + n_a2_epochs + n_b * (n_b1_epochs + n_b2_epochs),
                    loss_val=0.0, loss_dict={},
                    notes='SMPL-X Fitting abgeschlossen')

    return {
        'betas':     phase_a['betas'],
        'n_frames_a': len(frames),
        'n_frames_b': len(frames_b),
        'kp_loss_a':  phase_a['kp_loss'],
    }


# ── Hilfsfunktion: Frame aus Video lesen ──────────────────────────────────────

def _read_frame_for_fd(fd: dict) -> Optional[np.ndarray]:
    """Liest das Videobild für einen Frame-Data-Dict."""
    try:
        from ..models import DetectedPerson
        person = DetectedPerson.objects.get(id=fd['person_id'])
        cap = cv2.VideoCapture(person.video.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fd['frame_idx'])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        # Auf max 640px skalieren für CPU-Effizienz
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame
    except Exception as e:
        log.debug("_read_frame_for_fd failed: %s", e)
        return None
