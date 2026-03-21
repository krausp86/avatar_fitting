"""
Stage 2 – Static Vertex Offsets (ΔV_static)

Estimates per-vertex shape offsets in rest-pose space that capture surface
detail not explained by the SMPL-X shape basis (clothing, body proportions,
hair volume, etc.).

Algorithm:
  1. Load Stage 1 results: β, θ_t, global_orient_t, transl_t, camera.
  2. Cluster frames by pose similarity → select representative frames so
     different body configurations all contribute to the offset estimate.
  3. Optimise ΔV_static  (V, 3)  by minimising:
       L = w_sil  * silhouette_loss(V_smplx(β, θ_t) + ΔV, masks)
         + w_lap  * laplacian_smooth(ΔV)
         + w_l2   * ||ΔV||²
  4. ΔV_static lives in rest-pose / canonical space → applied via
     smplx_model.v_template before each forward pass.

Outputs saved to avatar.data_path:
  geometry.npz  – delta_v_static (V, 3), cluster_ids (N,)
  metadata.json – merged with stage2 quality metrics
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

ProgressCB = Callable[[dict], None]


# ── Config & result types ──────────────────────────────────────────────────────

@dataclass
class Stage2Config:
    n_offset_epochs: int   = 200
    lr_offsets:      float = 1e-3
    w_silhouette:    float = 1.0
    w_laplacian:     float = 10.0
    w_l2:            float = 0.1
    n_pose_clusters: int   = 8
    max_frames:      int   = 100   # cap for GPU memory


@dataclass
class Stage2Result:
    delta_v_static:  np.ndarray   # (V, 3)  vertex offsets in rest-pose space
    cluster_ids:     np.ndarray   # (N,)    cluster assignment per selected frame
    fitting_quality: dict


# ── Entry point ────────────────────────────────────────────────────────────────

def run_stage2(
    avatar,
    config_dict: dict,
    progress_cb: Optional[ProgressCB] = None,
) -> Stage2Result:
    """
    Run Stage 2 static vertex offset optimisation.

    Args:
        avatar:      Avatar Django model instance (.group, .data_path).
        config_dict: Fitting config from the FittingJob.
        progress_cb: Called with progress dict every 10 epochs.

    Returns:
        Stage2Result with ΔV_static and quality metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Stage2Config(
        n_offset_epochs = config_dict.get('n_offset_epochs', 200),
        lr_offsets      = config_dict.get('lr_offsets', 1e-3),
        w_silhouette    = config_dict.get('w_silhouette_s2', 1.0),
        w_laplacian     = config_dict.get('w_laplacian', 10.0),
        w_l2            = config_dict.get('w_l2', 0.1),
        n_pose_clusters = config_dict.get('n_pose_clusters', 8),
    )
    log.info("Stage 2 on device=%s  epochs=%d", device, cfg.n_offset_epochs)

    # ── Load Stage 1 outputs ─────────────────────────────────────────────────
    s1 = _load_stage1(avatar.data_path)

    theta_t   = torch.tensor(s1['theta_t'],        dtype=torch.float32, device=device)
    orient_t  = torch.tensor(s1['global_orient_t'], dtype=torch.float32, device=device)
    transl_t  = torch.tensor(s1['transl_t'],        dtype=torch.float32, device=device)
    beta      = torch.tensor(s1['beta'],            dtype=torch.float32, device=device).unsqueeze(0)
    cam       = s1['camera_intrinsics']
    num_betas = len(s1['beta'])

    # ── Load video frames + silhouette masks ─────────────────────────────────
    from .stage1 import _load_person_frames
    _, masks, _ = _load_person_frames(avatar)
    N = theta_t.shape[0]

    # ── Sub-sample frames if needed ──────────────────────────────────────────
    if N > cfg.max_frames:
        step = N // cfg.max_frames
        sel  = list(range(0, N, step))[:cfg.max_frames]
    else:
        sel  = list(range(N))

    theta_t  = theta_t[sel]
    orient_t = orient_t[sel]
    transl_t = transl_t[sel]
    masks_sel = [masks[i] if i < len(masks) else None for i in sel]
    n_frames  = len(sel)

    # ── Cluster frames by pose for representative coverage ───────────────────
    cluster_ids = _cluster_poses(theta_t.cpu().numpy(), cfg.n_pose_clusters)

    # ── Load SMPL-X ──────────────────────────────────────────────────────────
    smplx_model     = _load_smplx(num_betas, device, batch_size=n_frames)
    V               = smplx_model.v_template.shape[0]
    v_template_orig = smplx_model.v_template.data.clone()

    # ── Precompute sparse Laplacian ──────────────────────────────────────────
    L_sparse = _build_laplacian_sparse(
        smplx_model.faces_tensor.cpu().numpy(), V, device
    )

    # ── Optimise ΔV_static ───────────────────────────────────────────────────
    delta_v   = nn.Parameter(torch.zeros(V, 3, device=device))
    optimizer = torch.optim.Adam([delta_v], lr=cfg.lr_offsets)

    sil_ok    = _pytorch3d_available()
    if not sil_ok:
        log.warning(
            "pytorch3d not available – Stage 2 silhouette loss disabled. "
            "ΔV_static will be near-zero (geometry refinement skipped)."
        )

    beta_exp = beta.expand(n_frames, -1)
    total    = cfg.n_offset_epochs

    _cb(progress_cb, 0, total, 0.0, {}, note='Optimising vertex offsets…')

    loss_sil = loss_lap = loss_l2 = torch.tensor(0.0, device=device)

    for epoch in range(total):
        optimizer.zero_grad()

        # Temporarily modify SMPL-X rest-pose template
        smplx_model.v_template.data = v_template_orig + delta_v

        out = smplx_model(
            betas         = beta_exp,
            body_pose     = theta_t,
            global_orient = orient_t,
            transl        = transl_t,
        )
        # out.vertices: (n_frames, V, 3)

        if sil_ok:
            from .stage1 import _silhouette_loss_batch
            loss_sil = _silhouette_loss_batch(
                out.vertices, smplx_model.faces_tensor, masks_sel, cam, device
            )
        else:
            loss_sil = torch.zeros(1, device=device)

        loss_lap = _laplacian_loss(delta_v, L_sparse)
        loss_l2  = (delta_v ** 2).mean()

        loss = (cfg.w_silhouette * loss_sil
              + cfg.w_laplacian  * loss_lap
              + cfg.w_l2         * loss_l2)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            _cb(progress_cb, epoch, total, float(loss), {
                'silhouette': round(float(loss_sil), 5),
                'laplacian':  round(float(loss_lap), 5),
                'l2':         round(float(loss_l2), 5),
            })

    # Restore original v_template so the model is not mutated permanently
    smplx_model.v_template.data = v_template_orig

    delta_v_np = delta_v.detach().cpu().numpy()
    offset_rms = float(np.sqrt((delta_v_np ** 2).sum(axis=-1).mean()))

    return Stage2Result(
        delta_v_static  = delta_v_np,
        cluster_ids     = cluster_ids,
        fitting_quality = {
            'stage2_sil_loss':   round(float(loss_sil), 5),
            'stage2_lap_loss':   round(float(loss_lap), 5),
            'stage2_l2_loss':    round(float(loss_l2), 5),
            'stage2_n_frames':   n_frames,
            'stage2_n_clusters': int(cfg.n_pose_clusters),
            'stage2_offset_rms': round(offset_rms, 5),
        },
    )


def save_stage2_result(result: Stage2Result, data_path: str) -> None:
    """Write Stage 2 outputs into the avatar data folder."""
    os.makedirs(data_path, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_path, 'geometry.npz'),
        delta_v_static = result.delta_v_static,
        cluster_ids    = result.cluster_ids,
    )

    meta_path = os.path.join(data_path, 'metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta.setdefault('fitting_quality', {}).update(result.fitting_quality)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info("Stage 2 results saved → %s", data_path)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_stage1(data_path: str) -> dict:
    """Load Stage 1 outputs: poses.npz + metadata.json."""
    poses_path = os.path.join(data_path, 'poses.npz')
    meta_path  = os.path.join(data_path, 'metadata.json')

    if not os.path.exists(poses_path):
        raise RuntimeError(
            f"Stage 1 output not found at {poses_path}. Run Stage 1 first."
        )
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Stage 1 metadata not found at {meta_path}. Run Stage 1 first."
        )

    poses = np.load(poses_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return {
        'theta_t':          poses['theta_t'],
        'global_orient_t':  poses['global_orient_t'],
        'transl_t':         poses['transl_t'],
        'beta':             np.array(meta['beta'], dtype=np.float32),
        'camera_intrinsics': meta['camera_intrinsics'],
        'num_betas':        len(meta['beta']),
    }


# ── SMPL-X ────────────────────────────────────────────────────────────────────

def _load_smplx(num_betas: int, device: torch.device, batch_size: int = 1):
    from django.conf import settings
    import smplx

    model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
    model = smplx.create(
        model_path     = model_dir,
        model_type     = 'smplx',
        gender         = 'neutral',
        num_betas      = num_betas,
        use_pca        = False,
        flat_hand_mean = True,
        batch_size     = batch_size,
    ).to(device)
    model.eval()
    return model


# ── Pose clustering ────────────────────────────────────────────────────────────

def _cluster_poses(theta: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster N pose vectors (N, 63) into n_clusters groups via K-means.
    Returns integer cluster assignment array (N,).

    Falls back gracefully when N <= n_clusters.
    """
    N = theta.shape[0]
    k = min(n_clusters, N)

    if k <= 1:
        return np.zeros(N, dtype=np.int32)

    # Initialise centroids as evenly-spaced frames
    init_idx   = np.linspace(0, N - 1, k, dtype=int)
    centroids  = theta[init_idx].copy()

    assignments = np.zeros(N, dtype=np.int32)

    for _ in range(30):
        # E-step: assign each frame to nearest centroid
        dists       = np.linalg.norm(theta[:, None, :] - centroids[None, :, :], axis=-1)
        new_assign  = dists.argmin(axis=1).astype(np.int32)

        if np.array_equal(new_assign, assignments):
            break
        assignments = new_assign

        # M-step: update centroids
        for c in range(k):
            members = theta[assignments == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    return assignments


# ── Laplacian ─────────────────────────────────────────────────────────────────

def _build_laplacian_sparse(
    faces:  np.ndarray,       # (F, 3) int
    V:      int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a sparse uniform Laplacian matrix of shape (V, V).

    L_ii  =  degree(i)
    L_ij  = -1          for each edge (i, j)

    ||L @ ΔV||²  penalises deviation of a vertex from the mean of its neighbours,
    promoting spatial smoothness.
    """
    degree  = np.zeros(V, dtype=np.float32)
    edge_set: set = set()

    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            if a != b:
                edge_set.add((min(a, b), max(a, b)))

    rows, cols, vals = [], [], []

    for a, b in edge_set:
        rows += [a, b]
        cols += [b, a]
        vals += [-1.0, -1.0]
        degree[a] += 1.0
        degree[b] += 1.0

    for i in range(V):
        rows.append(i)
        cols.append(i)
        vals.append(degree[i])

    indices = torch.tensor([rows, cols], dtype=torch.long,   device=device)
    values  = torch.tensor(vals,         dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (V, V)).coalesce()


def _laplacian_loss(delta_v: torch.Tensor, L_sparse: torch.Tensor) -> torch.Tensor:
    """
    Smooth Laplacian loss: mean squared norm of (L @ ΔV).
    Penalises high-frequency vertex displacements.
    """
    Lv = torch.sparse.mm(L_sparse, delta_v)   # (V, 3)
    return (Lv ** 2).mean()


# ── PyTorch3D availability ─────────────────────────────────────────────────────

def _pytorch3d_available() -> bool:
    try:
        import pytorch3d   # noqa: F401
        return True
    except ImportError:
        return False


# ── Progress helper ────────────────────────────────────────────────────────────

def _cb(
    cb:           Optional[ProgressCB],
    epoch:        int,
    total_epochs: int,
    loss:         float,
    loss_terms:   dict,
    note:         str = '',
) -> None:
    if cb is None:
        return
    payload = {
        'type':         'progress',
        'stage':        '2',
        'stage_name':   'Static Vertex Offsets',
        'epoch':        epoch,
        'total_epochs': total_epochs,
        'loss':         loss,
        'loss_terms':   loss_terms,
        'preview_jpg':  None,
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
