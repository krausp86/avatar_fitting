"""
ROMP direct render — kein Optimizer, kein SMPL-X-Fit.

ROMP liefert smpl_thetas + cam. Wir berechnen die Vertices selbst über
smplx (model_type='smpl'), damit wir nicht auf ROPMs render_mesh=True
angewiesen sind (das würde die API-Signatur ändern und ist langsamer).

ROMP weak-perspective Kamera-Konvention:
  cam = [s, tx, ty]
  x_norm = s * X_3d + tx      (X in SMPL body frame, ~Meter)
  y_norm = s * Y_3d + ty      (Y-up)
  x_px = (x_norm + 1) / 2 * W
  y_px = (0.5 - y_norm / 2) * H    ← Y-up → Y-down
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_romp_model_cache: dict = {}
_smpl_model_cache: dict = {}


# ── ROMP laden ─────────────────────────────────────────────────────────────────

def _load_romp_model(model_path: str, smpl_path: Optional[str]):
    key = (model_path, smpl_path)
    if key in _romp_model_cache:
        return _romp_model_cache[key]

    try:
        import torch
        import romp

        settings = romp.romp_settings(input_args=[])
        settings.model_path   = model_path
        if smpl_path:
            settings.smpl_path = smpl_path
        settings.mode         = 'image'
        settings.show         = False
        settings.render_mesh  = False
        settings.show_largest = False
        settings.GPU = 0 if torch.cuda.is_available() else -1

        model = romp.ROMP(settings)
        _romp_model_cache[key] = model
        log.info("ROMP geladen: %s  GPU=%s", model_path, settings.GPU)
    except Exception as exc:
        log.warning("ROMP load failed: %s", exc)
        _romp_model_cache[key] = None

    return _romp_model_cache[key]


# ── SMPL Forward-Model (für Vertices + Faces) ──────────────────────────────────

def _load_smpl_model():
    """Lädt smplx SMPL-Modell (neutral). Sucht in models/SMPL/ (case-insensitive)."""
    if 'smpl' in _smpl_model_cache:
        return _smpl_model_cache['smpl']

    try:
        import torch
        import smplx
        from django.conf import settings as _s

        base = getattr(_s, 'SMPLX_MODEL_DIR', 'models')

        # smplx sucht <base>/smpl/ (lowercase). Unser Ordner heißt models/SMPL/.
        # Direkten Pfad zur pkl-Datei übergeben damit Groß/Kleinschreibung egal ist.
        candidates = [
            os.path.join(base, 'SMPL', 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'),
            os.path.join(base, 'smpl', 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'),
            os.path.join(base, 'SMPL', 'SMPL_NEUTRAL.pkl'),
            os.path.join(base, 'smpl', 'SMPL_NEUTRAL.pkl'),
        ]
        pkl_path = next((p for p in candidates if os.path.isfile(p)), None)

        if pkl_path is None:
            # Letzter Versuch: smplx mit model_type='smpl' und Verzeichnis
            for smpl_dir in [os.path.join(base, 'SMPL'), os.path.join(base, 'smpl'), base]:
                if os.path.isdir(smpl_dir):
                    try:
                        m = smplx.create(smpl_dir, model_type='smpl',
                                         gender='neutral', batch_size=1).eval()
                        _smpl_model_cache['smpl'] = m
                        log.info("SMPL geladen aus Verzeichnis: %s", smpl_dir)
                        return m
                    except Exception:
                        pass
            log.warning("SMPL-Modell nicht gefunden in %s", base)
            _smpl_model_cache['smpl'] = None
            return None

        m = smplx.create(pkl_path, model_type='smpl',
                         gender='neutral', batch_size=1).eval()
        _smpl_model_cache['smpl'] = m
        log.info("SMPL geladen: %s", pkl_path)
    except Exception as exc:
        log.warning("SMPL load failed: %s", exc)
        _smpl_model_cache['smpl'] = None

    return _smpl_model_cache['smpl']


def _smpl_forward(theta: np.ndarray, beta: np.ndarray):
    """
    SMPL Forward-Pass mit ROMP-Parametern.

    theta: (72,) [global_orient(3) | body_pose(69)]
    beta:  (10,) shape params

    Returns (verts (6890,3), faces (F,3)) oder (None, None).
    """
    smpl = _load_smpl_model()
    if smpl is None:
        return None, None

    try:
        import torch
        with torch.no_grad():
            out = smpl(
                global_orient=torch.tensor(theta[:3],    dtype=torch.float32).unsqueeze(0),
                body_pose=    torch.tensor(theta[3:72],  dtype=torch.float32).unsqueeze(0),
                betas=        torch.tensor(beta[:10],    dtype=torch.float32).unsqueeze(0),
            )
        verts = out.vertices[0].numpy().astype(np.float32)   # (6890, 3)
        faces = smpl.faces.astype(np.int32)                   # (F, 3)
        return verts, faces
    except Exception as exc:
        log.warning("SMPL forward failed: %s", exc)
        return None, None


# ── Weak-Perspective-Renderer ──────────────────────────────────────────────────

def _render_weak_perspective(
    frame_bgr: np.ndarray,
    verts:     np.ndarray,           # (6890, 3) SMPL body frame, Y-up
    faces:     np.ndarray,           # (F, 3) int32
    cam:       np.ndarray,           # (3,) [s, tx, ty]
    joints_2d: Optional[np.ndarray] = None,  # (J, 2) Pixel-Joints
) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    s, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])

    x_norm = s * verts[:, 0] + tx
    y_norm = s * verts[:, 1] + ty       # Y-up
    x2d    = (x_norm + 1.0) / 2.0 * W
    y2d    = (0.5 - y_norm / 2.0) * H  # Y-up → Y-down

    z_v = verts[:, 2]
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    px0, py0 = x2d[f0], y2d[f0]
    px1, py1 = x2d[f1], y2d[f1]
    px2, py2 = x2d[f2], y2d[f2]
    cross_z = (px1 - px0) * (py2 - py0) - (py1 - py0) * (px2 - px0)

    M = 20
    in_screen = (
        (np.minimum(px0, np.minimum(px1, px2)) < W + M) &
        (np.maximum(px0, np.maximum(px1, px2)) > -M) &
        (np.minimum(py0, np.minimum(py1, py2)) < H + M) &
        (np.maximum(py0, np.maximum(py1, py2)) > -M)
    )

    mask_ccw = in_screen & (cross_z < 0)
    mask_cw  = in_screen & (cross_z > 0)
    vis_mask = mask_ccw if mask_ccw.sum() >= mask_cw.sum() else mask_cw

    if vis_mask.sum() == 0:
        log.warning("romp_render: keine sichtbaren Faces — Original-Frame zurück")
        return frame_bgr.copy()

    vis_faces = faces[vis_mask]
    mean_z    = (z_v[vis_faces[:, 0]] + z_v[vis_faces[:, 1]] + z_v[vis_faces[:, 2]]) / 3.0
    order     = np.argsort(-mean_z)
    sf        = vis_faces[order]

    x_pts = np.clip(x2d[sf], -32000, 32000).astype(np.int32)
    y_pts = np.clip(y2d[sf], -32000, 32000).astype(np.int32)
    pts   = np.stack([x_pts, y_pts], axis=2)

    overlay  = frame_bgr.copy()
    cv2.fillPoly(overlay, list(pts), (130, 168, 210))
    result = cv2.addWeighted(frame_bgr, 0.3, overlay, 0.7, 0)

    if joints_2d is not None:
        for jx, jy in joints_2d:
            cv2.circle(result, (int(jx), int(jy)), 4, (220, 80, 0), -1)
            cv2.circle(result, (int(jx), int(jy)), 4, (255, 255, 255), 1)

    return result


# ── Öffentliche API ────────────────────────────────────────────────────────────

def romp_infer_params(
    frame_bgr:  np.ndarray,
    model_path: Optional[str] = None,
    smpl_path:  Optional[str] = None,
) -> Optional[dict]:
    """
    Nur ROMP-Inferenz, kein Rendering.

    Returns dict mit keys: theta (72,), beta (10,), cam (3,), joints_2d (J,2)|None
    oder None wenn ROMP nicht verfügbar/keine Person erkannt.
    """
    if not model_path:
        return None
    romp_model = _load_romp_model(model_path, smpl_path)
    if romp_model is None:
        return None
    try:
        result = romp_model(frame_bgr)
    except Exception as exc:
        log.warning("ROMP Inferenz fehlgeschlagen: %s", exc)
        return None
    if result is None:
        return None
    thetas = result.get('smpl_thetas')
    if thetas is None or len(thetas) == 0:
        return None
    return {
        'source':    'romp',
        'theta':     np.array(thetas[0],                   dtype=np.float32).tolist(),
        'beta':      np.array(result['smpl_betas'][0],     dtype=np.float32).tolist(),
        'cam':       np.array(result['cam'][0],            dtype=np.float32).tolist(),
        'joints_2d': np.array(result['pj2d'][0],           dtype=np.float32).tolist()
                     if result.get('pj2d') is not None else None,
        'n_persons': int(len(thetas)),
    }


def render_romp_from_params(
    frame_bgr: np.ndarray,
    params:    dict,
) -> Optional[np.ndarray]:
    """Rendert SMPL-Mesh aus bereits berechneten ROMP-Parametern."""
    theta     = np.array(params['theta'], dtype=np.float32)
    beta      = np.array(params['beta'],  dtype=np.float32)
    cam       = np.array(params['cam'],   dtype=np.float32)
    joints_2d = np.array(params['joints_2d'], dtype=np.float32) if params.get('joints_2d') else None

    verts, faces = _smpl_forward(theta, beta)
    if verts is None:
        return None
    return _render_weak_perspective(frame_bgr, verts, faces, cam, joints_2d)


def romp_render_frame(
    frame_bgr:  np.ndarray,
    model_path: Optional[str] = None,
    smpl_path:  Optional[str] = None,
) -> tuple[Optional[np.ndarray], dict]:
    """
    ROMP-Inferenz + SMPL Forward + Rendering.

    Returns (rendered_bgr, meta) oder (None, {}).
    meta enthält auch 'pose_params' für den Smoothing-Cache.
    """
    params = romp_infer_params(frame_bgr, model_path, smpl_path)
    if params is None:
        return None, {}

    rendered = render_romp_from_params(frame_bgr, params)
    if rendered is None:
        return None, {}

    meta = {
        'source':      'romp',
        'beta':        [round(float(v), 4) for v in params['beta']],
        'n_persons':   params['n_persons'],
        'pose_params': params,
    }
    log.info("romp_render: OK — %d Person(en)", params['n_persons'])
    return rendered, meta
