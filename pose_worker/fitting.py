"""
SMPL-X Fitting – Django-freie Version für den pose-worker.

Enthält run_phase_a() und run_phase_b_frame() aus core/fitting/fit_smplx.py,
mit folgenden Anpassungen:
  - Django-Settings → os.environ.get()
  - _read_frame_for_fd: liest video_path aus dem Frame-Dict (kein DB-Zugriff)
  - _fetch_smplx_init_for_frame: ruft SMPLerX/HMR2 direkt lokal auf
  - _collect_frames entfernt (Frames kommen via HTTP-Payload)
  - save_* entfernt (werden im web-Container ausgeführt)
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from fitting_keypoint_maps import (
    BODY_SMPLX_IDX, BODY_COCO17_IDX, BODY_VIS_THRESHOLD,
    FACE_RTM_OFFSET, N_FACE_LANDMARKS, FACE_VIS_THRESHOLD,
    LHAND_SMPLX_IDX, LHAND_MP_IDX, LHAND_RTM_IDX,
    RHAND_SMPLX_IDX, RHAND_MP_IDX, RHAND_RTM_IDX,
    HAND_VIS_THRESHOLD,
)

log = logging.getLogger(__name__)


def gmof(squared_error: torch.Tensor, sigma_sq: float) -> torch.Tensor:
    return sigma_sq * squared_error / (sigma_sq + squared_error)


_smplx_cache:  dict = {}
_vposer_cache: dict = {}


# ── VPoser ─────────────────────────────────────────────────────────────────────

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
    vposer_dir = os.environ.get('VPOSER_MODEL_DIR', '')
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
    import smplx
    model_dir = os.environ.get('SMPLX_MODEL_DIR', '/data/models')
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


def _load_smplx_phase_b(device: torch.device, batch_size: int = 1):
    import smplx
    model_dir = os.environ.get('SMPLX_MODEL_DIR', '/data/models')
    key = (model_dir, batch_size, True)
    if key not in _smplx_cache:
        m = smplx.create(
            model_path=model_dir, model_type='smplx',
            gender='neutral', num_betas=10, batch_size=batch_size,
            use_pca=True, num_pca_comps=12, flat_hand_mean=True,
            use_face_contour=False,
        ).eval()
        _smplx_cache[key] = m
    return _smplx_cache[key].to(device)


# ── FLAME landmark computation ──────────────────────────────────────────────────

def compute_flame_landmarks(vertices: torch.Tensor, smplx_model) -> Optional[torch.Tensor]:
    if not hasattr(smplx_model, 'lmk_faces_idx'):
        return None
    try:
        lmk_faces_idx   = smplx_model.lmk_faces_idx
        lmk_bary_coords = smplx_model.lmk_bary_coords
        faces_t = smplx_model.faces_tensor
        tri_idx = faces_t[lmk_faces_idx]
        tri_verts = vertices[:, tri_idx]
        bary = lmk_bary_coords.to(vertices.device)
        lmk_3d = (tri_verts * bary[None, :, :, None]).sum(dim=2)
        return lmk_3d
    except Exception as e:
        log.debug("compute_flame_landmarks failed: %s", e)
        return None


# ── Keypoint extraction helpers ────────────────────────────────────────────────

def _extract_body_kps(landmarks: list, W: int, H: int) -> np.ndarray:
    K = len(BODY_COCO17_IDX)
    out = np.zeros((K, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in landmarks}
    for k, coco_idx in enumerate(BODY_COCO17_IDX):
        d = lm_map.get(coco_idx)
        if d and d.get('visibility', 0) > 0:
            out[k] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


def _extract_face_kps(rtm_landmarks: list, W: int, H: int) -> np.ndarray:
    out = np.zeros((N_FACE_LANDMARKS, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in rtm_landmarks}
    for dlib_i in range(N_FACE_LANDMARKS):
        d = lm_map.get(FACE_RTM_OFFSET + dlib_i)
        if d and d.get('visibility', 0) > FACE_VIS_THRESHOLD:
            out[dlib_i] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


def _extract_hand_kps(rtm_landmarks: list, W: int, H: int, rtm_offset: int) -> np.ndarray:
    out = np.zeros((21, 3), dtype=np.float32)
    lm_map = {d['idx']: d for d in rtm_landmarks}
    for mp_i in range(21):
        d = lm_map.get(rtm_offset + mp_i)
        if d and d.get('visibility', 0) > HAND_VIS_THRESHOLD:
            out[mp_i] = [d['x'] * W, d['y'] * H, d['visibility']]
    return out


# ── Frame lesen (video_path aus Frame-Dict, kein DB-Zugriff) ───────────────────

def _read_frame_for_fd(fd: dict) -> Optional[np.ndarray]:
    try:
        video_path = fd.get('video_path')
        if not video_path:
            return None
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fd['frame_idx'])
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame
    except Exception as e:
        log.debug("_read_frame_for_fd failed: %s", e)
        return None


# ── SMPLer-X / HMR2 Init (lokaler Aufruf, kein HTTP-Hop) ──────────────────────

def _fetch_smplx_init_for_frame(frame_bgr: np.ndarray) -> Optional[dict]:
    try:
        from backends import SMPLerXBackend, HMR2Backend
        if SMPLerXBackend.available:
            return SMPLerXBackend.regress(frame_bgr)
        if HMR2Backend.available:
            return HMR2Backend.regress(frame_bgr)
        return None
    except Exception as e:
        log.warning("SMPLer-X/HMR2 init failed: %s", e, exc_info=True)
        return None


# ── Phase A: Beta Estimation ────────────────────────────────────────────────────

def run_phase_a(
    frames_data:     list,
    smplx_inits:     list,
    n_phase1_epochs: int = 300,
    n_phase2_epochs: int = 200,
    progress_cb=None,
) -> dict:
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

    kp_px   = np.zeros((N, K, 3), dtype=np.float32)
    pelv_px = np.zeros((N, 3),    dtype=np.float32)
    Ws      = np.zeros(N, dtype=np.float32)
    Hs      = np.zeros(N, dtype=np.float32)

    for n, fd in enumerate(frames_data):
        W, H = int(fd['W']), int(fd['H'])
        Ws[n], Hs[n] = W, H
        kp_px[n] = _extract_body_kps(fd['body_landmarks'], W, H)
        hip_l_k = BODY_COCO17_IDX.index(11)
        hip_r_k = BODY_COCO17_IDX.index(12)
        hl, hr  = kp_px[n, hip_l_k], kp_px[n, hip_r_k]
        pelv_px[n] = [(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2, min(hl[2], hr[2])]

    kp_t   = torch.tensor(kp_px,   dtype=torch.float32, device=device)
    pelv_t = torch.tensor(pelv_px, dtype=torch.float32, device=device)

    # Diagnose: Keypoint-Visibility-Statistik
    vis_all = kp_px[:, :, 2]
    n_vis   = int((vis_all > 0.1).sum())
    log.info("Phase A keypoints: N=%d frames, K=%d joints, "
             "vis>0.1: %d/%d  vis_mean=%.3f  vis_max=%.3f",
             N, K, n_vis, N * K, float(vis_all.mean()), float(vis_all.max()))
    if n_vis == 0:
        log.warning("Phase A: ALLE Keypoints haben visibility=0 – kp_loss wird 0 sein! "
                    "PersonFrameKeypoints ggf. neu berechnen (Datenbank leeren).")
    # Diagnose: Stichprobe der ersten gespeicherten Landmarks (Rohformat)
    if N > 0 and frames_data[0].get('body_landmarks'):
        sample = frames_data[0]['body_landmarks'][:3]
        log.info("Phase A landmark sample (frame 0): %s", sample)

    fx_base    = np.maximum(Ws, Hs) * 1.2
    fx_base_t  = torch.tensor(fx_base,   dtype=torch.float32, device=device)
    cx_t       = torch.tensor(Ws / 2.0,  dtype=torch.float32, device=device)
    cy_t       = torch.tensor(Hs / 2.0,  dtype=torch.float32, device=device)

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

    adj_pairs: list = []
    clip_frames: dict = defaultdict(list)
    for n, fd in enumerate(frames_data):
        clip_frames[fd.get('clip_id', 'default')].append((fd.get('sample_order', n), n))
    for items in clip_frames.values():
        items.sort()
        for i in range(len(items) - 1):
            adj_pairs.append((items[i][1], items[i + 1][1]))

    valid_inits = [x for x in smplx_inits if x is not None]
    if valid_inits:
        beta_init = np.mean([x['beta'] for x in valid_inits], axis=0)
        log.info("Phase A: beta0 aus SMPLer-X (%d frames)", len(valid_inits))
    else:
        beta_init = np.zeros(10, dtype=np.float32)
        log.info("Phase A: beta0 = zeros (kein SMPLer-X)")

    beta        = nn.Parameter(torch.tensor(beta_init[None], dtype=torch.float32, device=device))
    glob_orient = nn.Parameter(torch.zeros(N, 3, device=device))
    transl      = nn.Parameter(torch.tensor(
        np.column_stack([np.zeros(N), np.zeros(N), z_init]),
        dtype=torch.float32, device=device,
    ))
    log_focal   = nn.Parameter(torch.tensor(0.0, device=device))
    pose_z      = nn.Parameter(torch.zeros(N, 32 if use_vposer else 63, device=device))

    if valid_inits and use_vposer:
        try:
            bp_arr = np.array([x['body_pose'] for x in smplx_inits if x is not None], dtype=np.float32)
            m_idx  = [i for i, x in enumerate(smplx_inits) if x is not None]
            bp_t   = torch.tensor(bp_arr, dtype=torch.float32, device=device)
            bp_t   = bp_t.view(-1, 21, 3)
            with torch.no_grad():
                enc   = vposer.encode(bp_t)
                z_enc = enc.mean if hasattr(enc, 'mean') else enc
            for j, n in enumerate(m_idx):
                if j < len(z_enc):
                    pose_z.data[n] = z_enc[j]
            log.info("Phase A: pose_z init aus VPoser-Encoding")
        except Exception as e:
            log.debug("pose_z VPoser init failed: %s", e)

    def _fx():
        return fx_base_t * torch.exp(log_focal)

    def project(j3d):
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

    params_p1 = [beta, glob_orient, transl, log_focal, pose_z]
    params_p2 = [beta, glob_orient, transl, log_focal, pose_z]

    SIGMA_BODY_SQ = 100.0 ** 2
    # Warm-up sigma: start near-L2 (large sigma) and anneal to SIGMA_BODY_SQ.
    # Prevents near-zero gradients when initial projection errors are large.
    SIGMA_WARMUP  = 1000.0 ** 2   # ≈ L2 at startup
    W_KP, W_PELV, W_BETA, W_POSE, W_TEMP, W_FOC = 1.0, 0.5, 0.5, 0.003, 0.2, 0.05

    total_epochs = n_phase1_epochs + n_phase2_epochs

    for phase, (params, n_ep) in enumerate(
            [(params_p1, n_phase1_epochs), (params_p2, n_phase2_epochs)]):
        opt   = torch.optim.Adam(params, lr=5e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)

        for ep in range(n_ep):
            opt.zero_grad()
            body_pose = get_body_pose()

            # Anneal sigma: first 30% of epochs use warm-up sigma, rest use target
            t = ep / max(n_ep - 1, 1)
            warmup_frac = min(1.0, t / 0.3)   # 0→1 over first 30%
            sigma_sq = SIGMA_WARMUP * (1 - warmup_frac) + SIGMA_BODY_SQ * warmup_frac

            out  = smplx_model(
                betas         = beta.expand(N, -1),
                global_orient = glob_orient,
                body_pose     = body_pose,
                transl        = transl,
            )
            j3d  = out.joints
            j2d  = project(j3d)

            j2d_body = j2d[:, BODY_SMPLX_IDX, :]
            vis      = kp_t[:, :, 2:3]
            r_sq     = ((j2d_body - kp_t[:, :, :2]) ** 2).sum(-1)
            loss_kp  = (gmof(r_sq, sigma_sq) * vis[:, :, 0]).sum() / (vis.sum() + 1e-6)

            j2d_pelv   = j2d[:, 0, :]
            pelv_vis   = pelv_t[:, 2:3]
            r_sq_pelv  = ((j2d_pelv - pelv_t[:, :2]) ** 2).sum(-1)
            loss_pelv  = (gmof(r_sq_pelv, sigma_sq) * pelv_vis[:, 0]).sum() / (pelv_vis.sum() + 1e-6)

            loss_beta = beta.pow(2).mean()
            loss_pose = pose_z.pow(2).mean()
            loss_foc  = log_focal.pow(2)

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
                    'phase':      f'A{phase+1}',
                    'epoch':      ep,
                    'epoch_all':  ep_all,
                    'total':      total_epochs,
                    'loss':       float(loss),
                    'loss_terms': {
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

    with torch.no_grad():
        beta_np        = beta.squeeze(0).cpu().numpy().tolist()
        focal_sc       = float(torch.exp(log_focal).item())
        body_pose_np   = get_body_pose().cpu().numpy()
        glob_orient_np = glob_orient.cpu().numpy()
        transl_np      = transl.cpu().numpy()

    per_frame = []
    for n, fd in enumerate(frames_data):
        per_frame.append({
            'person_id':     fd['person_id'],
            'frame_idx':     fd['frame_idx'],
            'body_pose':     body_pose_np[n].tolist(),
            'global_orient': glob_orient_np[n].tolist(),
            'transl':        transl_np[n].tolist(),
            'W':             fd['W'],
            'H':             fd['H'],
        })

    log.info("Phase A done: beta_rms=%.3f  focal_scale=%.3f",
             float(np.array(beta_np).mean()), focal_sc)

    return {
        'betas':       beta_np,
        'focal_scale': focal_sc,
        'kp_loss':     float(loss_kp),
        'n_frames':    N,
        'per_frame':   per_frame,
    }


# ── Phase B: Pro-Frame Full SMPL-X ─────────────────────────────────────────────

@dataclass
class FrameFitResult:
    person_id:       str
    frame_idx:       int
    body_pose:       List[float]
    global_orient:   List[float]
    transl:          List[float]
    expression:      List[float]
    jaw_pose:        List[float]
    left_hand_pose:  List[float]
    right_hand_pose: List[float]
    cam_scale:       float
    cam_tx:          float
    cam_ty:          float
    loss_body:       float = 0.0
    loss_face:       float = 0.0
    loss_lhand:      float = 0.0
    loss_rhand:      float = 0.0


def run_phase_b_frame(
    fd:           dict,
    betas:        torch.Tensor,
    smplx_model,
    vposer,
    phase_a_pose: dict,
    smplx_init:   Optional[dict],
    n_b1_epochs:  int = 100,
    n_b2_epochs:  int = 150,
    device:       torch.device = torch.device('cpu'),
) -> FrameFitResult:
    use_vposer = vposer is not None
    W, H = int(fd['W']), int(fd['H'])

    kp_body  = _extract_body_kps(fd['body_landmarks'], W, H)
    kp_face  = _extract_face_kps(fd['rtm_landmarks'],  W, H)
    kp_lhand = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 91)
    kp_rhand = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 112)

    n_body_lm = len(fd.get('body_landmarks', []))
    n_vis_body = int((kp_body[:, 2] > 0).sum())
    log.info("Phase B frame person=%s idx=%s: body_lm=%d  vis_body=%d/%d  vis_sum=%.3f",
             fd.get('person_id'), fd.get('frame_idx'), n_body_lm, n_vis_body, len(BODY_COCO17_IDX),
             float(kp_body[:, 2].sum()))

    kp_body_t  = torch.tensor(kp_body,  dtype=torch.float32, device=device)
    kp_face_t  = torch.tensor(kp_face,  dtype=torch.float32, device=device)
    kp_lhand_t = torch.tensor(kp_lhand, dtype=torch.float32, device=device)
    kp_rhand_t = torch.tensor(kp_rhand, dtype=torch.float32, device=device)

    has_face  = kp_face_t[:, 2].sum() > 5
    has_lhand = kp_lhand_t[:, 2].sum() > 3
    has_rhand = kp_rhand_t[:, 2].sum() > 3

    init = smplx_init or {}

    def _init_tensor(key, fallback_np, n):
        v = init.get(key)
        if v and len(v) >= n:
            return torch.tensor(v[:n], dtype=torch.float32, device=device)
        return torch.tensor(np.array(fallback_np[:n], dtype=np.float32), device=device)

    bp_init = phase_a_pose.get('body_pose',     [0.0] * 63)
    go_init = phase_a_pose.get('global_orient', [0.0] * 3)
    tr_init = phase_a_pose.get('transl',        [0.0, 0.0, 3.0])

    glob_orient  = nn.Parameter(_init_tensor('global_orient', go_init, 3).unsqueeze(0))
    transl_param = nn.Parameter(_init_tensor('transl', tr_init, 3).unsqueeze(0))
    expression   = nn.Parameter(torch.zeros(1, 10,  device=device))
    jaw_pose     = nn.Parameter(torch.zeros(1,  3,  device=device))
    lhand_pose   = nn.Parameter(torch.zeros(1, 12,  device=device))
    rhand_pose   = nn.Parameter(torch.zeros(1, 12,  device=device))

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

    z_body     = float(transl_param.data[0, 2].item())
    if z_body < 0.5:
        z_body = 3.0
    fx_init    = max(W, H) * 1.2
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
        j = j3d_1[0]
        x_px = cam_scale[0, 0] *   j[:, 0] + cam_tx[0, 0]
        y_px = cam_scale[0, 0] * (-j[:, 1]) + cam_ty[0, 0]
        return torch.stack([x_px, y_px], dim=-1)

    SIGMA_BODY_SQ = 100.0 ** 2
    SIGMA_BODY_WARMUP = 1000.0 ** 2   # near-L2 warm-up sigma for large initial errors
    SIGMA_FACE_SQ = 5.0   ** 2
    SIGMA_HAND_SQ = 10.0  ** 2
    W_BODY, W_FACE, W_HAND = 1.0, 0.5, 0.3
    W_POSE, W_EXPR, W_JAW, W_HAND_PRIOR = 0.005, 0.01, 0.05, 0.01

    def compute_losses(include_face_hands: bool, sigma_body_sq: float = SIGMA_BODY_SQ):
        body_pose = get_body_pose()
        out = smplx_model(
            betas           = betas,
            global_orient   = glob_orient,
            body_pose       = body_pose,
            transl          = transl_param,
            expression      = expression,
            jaw_pose        = jaw_pose,
            left_hand_pose  = lhand_pose,
            right_hand_pose = rhand_pose,
        )
        j3d = out.joints
        j2d = project_wp(j3d)

        j2d_body  = j2d[BODY_SMPLX_IDX]
        vis_body  = kp_body_t[:, 2]
        r_sq_body = ((j2d_body - kp_body_t[:, :2]) ** 2).sum(-1)
        loss_body = (gmof(r_sq_body, sigma_body_sq) * vis_body).sum() / (vis_body.sum() + 1e-6)

        loss_face = loss_lhand = loss_rhand = torch.zeros(1, device=device)[0]

        if include_face_hands:
            if has_face:
                verts = out.vertices
                lmk3d = compute_flame_landmarks(verts, smplx_model)
                if lmk3d is not None:
                    lmk2d    = project_wp(lmk3d)
                    n_lmk    = lmk2d.shape[0]
                    kp_face_sel = kp_face_t[-n_lmk:, :]
                    vis_face = kp_face_sel[:, 2]
                    r_sq_face = ((lmk2d - kp_face_sel[:, :2]) ** 2).sum(-1)
                    loss_face = (gmof(r_sq_face, SIGMA_FACE_SQ) * vis_face).sum() / (vis_face.sum() + 1e-6)

            if has_lhand:
                j2d_lhand    = j2d[LHAND_SMPLX_IDX]
                kp_lhand_sel = kp_lhand_t[LHAND_MP_IDX]
                vis_lh       = kp_lhand_sel[:, 2]
                r_sq_lh      = ((j2d_lhand - kp_lhand_sel[:, :2]) ** 2).sum(-1)
                loss_lhand   = (gmof(r_sq_lh, SIGMA_HAND_SQ) * vis_lh).sum() / (vis_lh.sum() + 1e-6)

            if has_rhand:
                j2d_rhand    = j2d[RHAND_SMPLX_IDX]
                kp_rhand_sel = kp_rhand_t[RHAND_MP_IDX]
                vis_rh       = kp_rhand_sel[:, 2]
                r_sq_rh      = ((j2d_rhand - kp_rhand_sel[:, :2]) ** 2).sum(-1)
                loss_rhand   = (gmof(r_sq_rh, SIGMA_HAND_SQ) * vis_rh).sum() / (vis_rh.sum() + 1e-6)

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

    params_b1 = [pose_z, glob_orient, transl_param, cam_scale, cam_tx, cam_ty]
    opt = torch.optim.Adam(params_b1, lr=3e-3)
    for ep in range(n_b1_epochs):
        opt.zero_grad()
        loss, *_ = compute_losses(include_face_hands=False)
        loss.backward()
        opt.step()

    params_b2 = params_b1 + [expression, jaw_pose, lhand_pose, rhand_pose]
    opt2  = torch.optim.Adam(params_b2, lr=2e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=75, gamma=0.3)
    for ep in range(n_b2_epochs):
        opt2.zero_grad()
        loss, l_body, l_face, l_lh, l_rh = compute_losses(include_face_hands=True)
        loss.backward()
        opt2.step()
        sched2.step()

    log.info("Phase B frame done: loss_body=%.4f  loss_face=%.4f  vis_body_sum=%.3f",
             float(l_body), float(l_face), float(kp_body_t[:, 2].sum()))

    with torch.no_grad():
        body_pose_np = get_body_pose().squeeze(0).cpu().numpy().tolist()
        return FrameFitResult(
            person_id       = fd['person_id'],
            frame_idx       = fd['frame_idx'],
            body_pose       = body_pose_np,
            global_orient   = glob_orient.squeeze(0).cpu().numpy().tolist(),
            transl          = transl_param.squeeze(0).cpu().numpy().tolist(),
            expression      = expression.squeeze(0).cpu().numpy().tolist(),
            jaw_pose        = jaw_pose.squeeze(0).cpu().numpy().tolist(),
            left_hand_pose  = lhand_pose.squeeze(0).cpu().numpy().tolist(),
            right_hand_pose = rhand_pose.squeeze(0).cpu().numpy().tolist(),
            cam_scale       = float(cam_scale.item()),
            cam_tx          = float(cam_tx.item()),
            cam_ty          = float(cam_ty.item()),
            loss_body       = float(l_body),
            loss_face       = float(l_face),
            loss_lhand      = float(l_lh),
            loss_rhand      = float(l_rh),
        )


# ── Phase B: Batched (alle Frames gleichzeitig) ────────────────────────────────

def run_phase_b_batch(
    frames_data:   list,
    betas:         torch.Tensor,   # (1, 10)
    smplx_model,                   # batch_size=N
    vposer,
    phase_a_poses: list,           # len=N, je {body_pose, global_orient, transl}
    smplx_inits:   list,           # len=N, je dict oder None
    n_b1_epochs:   int = 100,
    n_b2_epochs:   int = 150,
    device:        torch.device = torch.device('cpu'),
    progress_cb    = None,
) -> list:
    """
    Verarbeitet alle N Frames in einem einzigen SMPL-X-Forward-Pass.
    Ersetzt die sequentielle Schleife über run_phase_b_frame.
    """
    use_vposer = vposer is not None
    N = len(frames_data)
    log.info("Phase B batch: N=%d frames, device=%s, use_vposer=%s", N, device, use_vposer)

    K = len(BODY_COCO17_IDX)
    kp_body_all  = np.zeros((N, K,                  3), dtype=np.float32)
    kp_face_all  = np.zeros((N, N_FACE_LANDMARKS,   3), dtype=np.float32)
    kp_lhand_all = np.zeros((N, 21,                 3), dtype=np.float32)
    kp_rhand_all = np.zeros((N, 21,                 3), dtype=np.float32)
    Ws = np.zeros(N, dtype=np.float32)
    Hs = np.zeros(N, dtype=np.float32)

    for n, fd in enumerate(frames_data):
        W, H = int(fd['W']), int(fd['H'])
        Ws[n], Hs[n] = W, H
        kp_body_all[n]  = _extract_body_kps(fd['body_landmarks'], W, H)
        kp_face_all[n]  = _extract_face_kps(fd['rtm_landmarks'],  W, H)
        kp_lhand_all[n] = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 91)
        kp_rhand_all[n] = _extract_hand_kps(fd['rtm_landmarks'],  W, H, 112)

    kp_body_t  = torch.tensor(kp_body_all,  dtype=torch.float32, device=device)
    kp_face_t  = torch.tensor(kp_face_all,  dtype=torch.float32, device=device)
    kp_lhand_t = torch.tensor(kp_lhand_all, dtype=torch.float32, device=device)
    kp_rhand_t = torch.tensor(kp_rhand_all, dtype=torch.float32, device=device)

    has_face  = kp_face_t[:, :, 2].sum(dim=1) > 5    # (N,) bool
    has_lhand = kp_lhand_t[:, :, 2].sum(dim=1) > 3
    has_rhand = kp_rhand_t[:, :, 2].sum(dim=1) > 3

    # ── Parameter-Initialisierung ──
    glob_orient_np = np.array(
        [p.get('global_orient', [0., 0., 0.])[:3] for p in phase_a_poses], dtype=np.float32)
    transl_np = np.array(
        [p.get('transl', [0., 0., 3.])[:3] for p in phase_a_poses], dtype=np.float32)

    glob_orient  = nn.Parameter(torch.tensor(glob_orient_np, device=device))   # (N, 3)
    transl_param = nn.Parameter(torch.tensor(transl_np,      device=device))   # (N, 3)
    expression   = nn.Parameter(torch.zeros(N, 10, device=device))
    jaw_pose     = nn.Parameter(torch.zeros(N,  3, device=device))
    lhand_pose   = nn.Parameter(torch.zeros(N, 12, device=device))
    rhand_pose   = nn.Parameter(torch.zeros(N, 12, device=device))

    # ── pose_z Init ──
    bp_inits = []
    for n, p in enumerate(phase_a_poses):
        si = smplx_inits[n] if smplx_inits else None
        if si and si.get('body_pose') and len(si['body_pose']) >= 63:
            bp_inits.append(si['body_pose'][:63])
        else:
            bp_inits.append(p.get('body_pose', [0.] * 63)[:63])
    bp_arr = np.array(bp_inits, dtype=np.float32)   # (N, 63)

    if use_vposer:
        try:
            bp_t = torch.tensor(bp_arr, dtype=torch.float32, device=device).view(N, 21, 3)
            with torch.no_grad():
                enc   = vposer.encode(bp_t)
                z_enc = enc.mean if hasattr(enc, 'mean') else enc
            pose_z = nn.Parameter(z_enc.detach())   # (N, 32)
        except Exception as e:
            log.warning("Phase B batch: VPoser init failed: %s", e)
            pose_z = nn.Parameter(torch.zeros(N, 32, device=device))
    else:
        pose_z = nn.Parameter(torch.tensor(bp_arr, device=device))   # (N, 63)

    # ── Kamera-Parameter pro Frame ──
    z_bodies    = np.clip(transl_np[:, 2], 0.5, None)
    fx_inits    = np.maximum(Ws, Hs) * 1.2
    scale_inits = fx_inits / z_bodies
    cam_scale   = nn.Parameter(torch.tensor(scale_inits[:, None], dtype=torch.float32, device=device))  # (N,1)
    cam_tx      = nn.Parameter(torch.tensor((Ws / 2.0)[:, None],  dtype=torch.float32, device=device))
    cam_ty      = nn.Parameter(torch.tensor((Hs / 2.0)[:, None],  dtype=torch.float32, device=device))

    def get_body_pose():
        if use_vposer:
            dec = vposer.decode(pose_z)
            bp  = dec.get('pose_body', dec.get('pose_body_matrot'))
            return bp.reshape(N, 63)
        return pose_z

    def project_wp(j3d: torch.Tensor) -> torch.Tensor:
        """j3d: (N, J, 3)  →  (N, J, 2)"""
        x_px = cam_scale * j3d[:, :, 0] + cam_tx     # (N, J)
        y_px = cam_scale * (-j3d[:, :, 1]) + cam_ty  # (N, J)
        return torch.stack([x_px, y_px], dim=-1)      # (N, J, 2)

    SIGMA_BODY_SQ = 100.0 ** 2
    SIGMA_FACE_SQ = 5.0   ** 2
    SIGMA_HAND_SQ = 10.0  ** 2
    W_BODY, W_FACE, W_HAND = 1.0, 0.5, 0.3
    W_POSE, W_EXPR, W_JAW, W_HAND_PRIOR = 0.005, 0.01, 0.05, 0.01

    def compute_losses(include_face_hands: bool):
        body_pose = get_body_pose()
        out = smplx_model(
            betas           = betas.expand(N, -1),
            global_orient   = glob_orient,
            body_pose       = body_pose,
            transl          = transl_param,
            expression      = expression,
            jaw_pose        = jaw_pose,
            left_hand_pose  = lhand_pose,
            right_hand_pose = rhand_pose,
        )
        j3d = out.joints       # (N, J, 3)
        j2d = project_wp(j3d)  # (N, J, 2)

        j2d_body  = j2d[:, BODY_SMPLX_IDX, :]              # (N, K, 2)
        vis_body  = kp_body_t[:, :, 2]                      # (N, K)
        r_sq_body = ((j2d_body - kp_body_t[:, :, :2]) ** 2).sum(-1)
        loss_body = (gmof(r_sq_body, SIGMA_BODY_SQ) * vis_body).sum() / (vis_body.sum() + 1e-6)

        loss_face = loss_lhand = loss_rhand = torch.zeros(1, device=device)[0]

        if include_face_hands:
            if has_face.any():
                verts = out.vertices                          # (N, V, 3)
                lmk3d = compute_flame_landmarks(verts, smplx_model)  # (N, n_lmk, 3) or None
                if lmk3d is not None:
                    lmk2d       = project_wp(lmk3d)          # (N, n_lmk, 2)
                    n_lmk       = lmk2d.shape[1]
                    kp_face_sel = kp_face_t[:, -n_lmk:, :]   # (N, n_lmk, 3)
                    mask        = has_face.float().unsqueeze(1)  # (N, 1)
                    vis_face    = kp_face_sel[:, :, 2] * mask
                    r_sq_face   = ((lmk2d - kp_face_sel[:, :, :2]) ** 2).sum(-1)
                    loss_face   = (gmof(r_sq_face, SIGMA_FACE_SQ) * vis_face).sum() / (vis_face.sum() + 1e-6)

            if has_lhand.any():
                j2d_lhand    = j2d[:, LHAND_SMPLX_IDX, :]
                kp_lhand_sel = kp_lhand_t[:, LHAND_MP_IDX, :]
                mask_lh      = has_lhand.float().unsqueeze(1)
                vis_lh       = kp_lhand_sel[:, :, 2] * mask_lh
                r_sq_lh      = ((j2d_lhand - kp_lhand_sel[:, :, :2]) ** 2).sum(-1)
                loss_lhand   = (gmof(r_sq_lh, SIGMA_HAND_SQ) * vis_lh).sum() / (vis_lh.sum() + 1e-6)

            if has_rhand.any():
                j2d_rhand    = j2d[:, RHAND_SMPLX_IDX, :]
                kp_rhand_sel = kp_rhand_t[:, RHAND_MP_IDX, :]
                mask_rh      = has_rhand.float().unsqueeze(1)
                vis_rh       = kp_rhand_sel[:, :, 2] * mask_rh
                r_sq_rh      = ((j2d_rhand - kp_rhand_sel[:, :, :2]) ** 2).sum(-1)
                loss_rhand   = (gmof(r_sq_rh, SIGMA_HAND_SQ) * vis_rh).sum() / (vis_rh.sum() + 1e-6)

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

    # ── B1: Körper ausrichten ──
    params_b1 = [pose_z, glob_orient, transl_param, cam_scale, cam_tx, cam_ty]
    opt = torch.optim.Adam(params_b1, lr=3e-3)
    for ep in range(n_b1_epochs):
        opt.zero_grad()
        loss, *_ = compute_losses(include_face_hands=False)
        loss.backward()
        opt.step()
        if progress_cb and ep % 25 == 0:
            progress_cb({'phase': 'B1', 'epoch': ep, 'loss': float(loss)})
        if ep % 50 == 0:
            log.info("Phase B1 batch ep %3d/%d  loss=%.4f", ep, n_b1_epochs, float(loss))

    # ── B2: Gesicht + Hände ──
    params_b2 = params_b1 + [expression, jaw_pose, lhand_pose, rhand_pose]
    opt2   = torch.optim.Adam(params_b2, lr=2e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=75, gamma=0.3)
    for ep in range(n_b2_epochs):
        opt2.zero_grad()
        loss, l_body, l_face, l_lh, l_rh = compute_losses(include_face_hands=True)
        loss.backward()
        opt2.step()
        sched2.step()
        if progress_cb and ep % 25 == 0:
            progress_cb({'phase': 'B2', 'epoch': ep, 'loss': float(loss)})
        if ep % 50 == 0:
            log.info("Phase B2 batch ep %3d/%d  loss=%.4f  body=%.4f", ep, n_b2_epochs,
                     float(loss), float(l_body))

    log.info("Phase B batch done: N=%d  loss_body=%.4f  loss_face=%.4f",
             N, float(l_body), float(l_face))

    # ── Pro-Frame Verluste berechnen ──
    with torch.no_grad():
        body_pose_final = get_body_pose()
        out_f = smplx_model(
            betas=betas.expand(N, -1), global_orient=glob_orient,
            body_pose=body_pose_final, transl=transl_param,
            expression=expression, jaw_pose=jaw_pose,
            left_hand_pose=lhand_pose, right_hand_pose=rhand_pose,
        )
        j2d_f     = project_wp(out_f.joints)
        j2d_b_f   = j2d_f[:, BODY_SMPLX_IDX, :]
        vis_b_f   = kp_body_t[:, :, 2]
        r_sq_f    = ((j2d_b_f - kp_body_t[:, :, :2]) ** 2).sum(-1)
        loss_body_per_frame = (
            (gmof(r_sq_f, SIGMA_BODY_SQ) * vis_b_f).sum(dim=1) /
            (vis_b_f.sum(dim=1) + 1e-6)
        ).cpu().numpy()   # (N,)

        body_pose_np   = body_pose_final.cpu().numpy()      # (N, 63)
        glob_orient_np = glob_orient.cpu().numpy()          # (N, 3)
        transl_out_np  = transl_param.cpu().numpy()         # (N, 3)
        expression_np  = expression.cpu().numpy()           # (N, 10)
        jaw_pose_np    = jaw_pose.cpu().numpy()             # (N, 3)
        lhand_np       = lhand_pose.cpu().numpy()           # (N, 12)
        rhand_np       = rhand_pose.cpu().numpy()           # (N, 12)
        cam_scale_np   = cam_scale[:, 0].cpu().numpy()      # (N,)
        cam_tx_np      = cam_tx[:, 0].cpu().numpy()
        cam_ty_np      = cam_ty[:, 0].cpu().numpy()

    results = []
    for n, fd in enumerate(frames_data):
        results.append(FrameFitResult(
            person_id       = fd['person_id'],
            frame_idx       = fd['frame_idx'],
            body_pose       = body_pose_np[n].tolist(),
            global_orient   = glob_orient_np[n].tolist(),
            transl          = transl_out_np[n].tolist(),
            expression      = expression_np[n].tolist(),
            jaw_pose        = jaw_pose_np[n].tolist(),
            left_hand_pose  = lhand_np[n].tolist(),
            right_hand_pose = rhand_np[n].tolist(),
            cam_scale       = float(cam_scale_np[n]),
            cam_tx          = float(cam_tx_np[n]),
            cam_ty          = float(cam_ty_np[n]),
            loss_body       = float(loss_body_per_frame[n]),
            loss_face       = float(l_face),
            loss_lhand      = float(l_lh),
            loss_rhand      = float(l_rh),
        ))
    return results
