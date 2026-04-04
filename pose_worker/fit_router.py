"""
FastAPI-Router für SMPL-X Fitting Jobs.

Endpoints:
  POST /fit/phase-a          → startet Phase-A-Job, gibt job_id zurück
  POST /fit/phase-b          → startet Phase-B-Job, gibt job_id zurück
  GET  /fit/{job_id}         → pollt Job-Status und Ergebnis
  DELETE /fit/{job_id}       → räumt Job auf

Nur ein Fitting-Job läuft gleichzeitig (_fitting_lock), da der Prozess
die GPU vollständig belegt. Weitere Jobs werden in einen Thread-Queue gestellt
und warten bis der Lock frei ist.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store: job_id → {status, progress, result, error}
_jobs: dict[str, dict] = {}
# Verhindert gleichzeitiges Fitting (GPU-Ressource)
_fitting_lock = threading.Lock()


# ── Pydantic-Schemas ───────────────────────────────────────────────────────────

class PhaseARequest(BaseModel):
    frames_data:      list[dict]
    n_phase1_epochs:  int  = 300
    n_phase2_epochs:  int  = 200
    use_smplx_init:   bool = True


class PhaseBRequest(BaseModel):
    frames_data:      list[dict]
    betas:            list[float]           # 10 Shape-Koeffizienten
    phase_a_per_frame: Optional[list[dict]] = None  # Body-Pose-Init pro Frame
    n_b1_epochs:      int  = 100
    n_b2_epochs:      int  = 150
    use_smplx_init:   bool = True


# ── Job-Thread-Funktionen ──────────────────────────────────────────────────────

def _run_phase_a(job_id: str, req: PhaseARequest) -> None:
    """Läuft in Daemon-Thread. Belegt _fitting_lock für die gesamte Dauer."""
    from fitting import run_phase_a, _read_frame_for_fd, _fetch_smplx_init_for_frame

    with _fitting_lock:
        _jobs[job_id]['status'] = 'running'
        try:
            frames = req.frames_data

            smplx_inits = [None] * len(frames)
            if req.use_smplx_init:
                n_ok = 0
                for i, fd in enumerate(frames):
                    try:
                        frame_bgr = _read_frame_for_fd(fd)
                        if frame_bgr is not None:
                            smplx_inits[i] = _fetch_smplx_init_for_frame(frame_bgr)
                            if smplx_inits[i]:
                                n_ok += 1
                    except Exception:
                        pass
                log.info("Phase A init: %d/%d frames mit SMPLer-X/HMR2", n_ok, len(frames))

            total_epochs = req.n_phase1_epochs + req.n_phase2_epochs

            def _progress(data: dict) -> None:
                epoch_all = data.get('epoch_all', data.get('epoch', 0))
                _jobs[job_id]['progress']    = epoch_all / max(total_epochs, 1)
                _jobs[job_id]['latest_loss'] = data.get('loss', 0.0)

            result = run_phase_a(
                frames_data     = frames,
                smplx_inits     = smplx_inits,
                n_phase1_epochs = req.n_phase1_epochs,
                n_phase2_epochs = req.n_phase2_epochs,
                progress_cb     = _progress,
            )

            _jobs[job_id]['status'] = 'done'
            _jobs[job_id]['result'] = result

        except Exception as e:
            log.exception("Phase A job %s failed", job_id)
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error']  = str(e)


def _run_phase_b(job_id: str, req: PhaseBRequest) -> None:
    """Läuft in Daemon-Thread. Belegt _fitting_lock für die gesamte Dauer."""
    import torch
    from fitting import (
        run_phase_b_batch, _load_smplx_phase_b,
        _try_load_vposer, _read_frame_for_fd, _fetch_smplx_init_for_frame,
        _smplx_cache,
    )

    with _fitting_lock:
        _jobs[job_id]['status'] = 'running'
        try:
            device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            frames  = req.frames_data
            n_b     = len(frames)

            smplx_b = _load_smplx_phase_b(device, batch_size=n_b)
            vposer  = _try_load_vposer(device)
            betas_t = torch.tensor([req.betas], dtype=torch.float32, device=device)

            # Phase-A-Pose-Init pro Frame aufbauen
            phase_a_by_key: dict = {}
            if req.phase_a_per_frame:
                phase_a_by_key = {
                    (p['person_id'], p['frame_idx']): p
                    for p in req.phase_a_per_frame
                }
            default_pose = {'body_pose': [0.0] * 63, 'global_orient': [0.0] * 3, 'transl': [0.0, 0.0, 3.0]}
            phase_a_poses = [
                phase_a_by_key.get((fd.get('person_id'), fd.get('frame_idx')), default_pose)
                for fd in frames
            ]

            # SMPLer-X/HMR2-Initialisierungen vorab laden (optional)
            smplx_inits = [None] * n_b
            if req.use_smplx_init:
                n_ok = 0
                for i, fd in enumerate(frames):
                    try:
                        frame_bgr = _read_frame_for_fd(fd)
                        if frame_bgr is not None:
                            smplx_inits[i] = _fetch_smplx_init_for_frame(frame_bgr)
                            if smplx_inits[i]:
                                n_ok += 1
                    except Exception:
                        pass
                log.info("Phase B init: %d/%d frames mit SMPLer-X/HMR2", n_ok, n_b)

            total_epochs = req.n_b1_epochs + req.n_b2_epochs

            def _progress(data: dict) -> None:
                phase  = data.get('phase', 'B1')
                ep     = data.get('epoch', 0)
                offset = 0 if phase == 'B1' else req.n_b1_epochs
                ep_all = offset + ep
                _jobs[job_id]['progress']    = ep_all / max(total_epochs, 1)
                _jobs[job_id]['latest_loss'] = data.get('loss', 0.0)

            frame_results = []
            try:
                results = run_phase_b_batch(
                    frames_data   = frames,
                    betas         = betas_t,
                    smplx_model   = smplx_b,
                    vposer        = vposer,
                    phase_a_poses = phase_a_poses,
                    smplx_inits   = smplx_inits,
                    n_b1_epochs   = req.n_b1_epochs,
                    n_b2_epochs   = req.n_b2_epochs,
                    device        = device,
                    progress_cb   = _progress,
                )
                frame_results = [dataclasses.asdict(r) for r in results]

            finally:
                _smplx_cache.clear()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            _jobs[job_id]['progress']    = 1.0
            _jobs[job_id]['latest_loss'] = frame_results[-1].get('loss_body', 0.0) if frame_results else 0.0
            _jobs[job_id]['status'] = 'done'
            _jobs[job_id]['result'] = {'frame_results': frame_results}

        except Exception as e:
            log.exception("Phase B job %s failed", job_id)
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error']  = str(e)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/phase-a")
def start_phase_a(req: PhaseARequest):
    if not req.frames_data:
        raise HTTPException(400, "frames_data ist leer")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {'status': 'queued', 'progress': 0.0, 'result': None, 'error': None}
    t = threading.Thread(target=_run_phase_a, args=(job_id, req), daemon=True)
    t.start()
    log.info("Phase A job %s gestartet (%d frames)", job_id, len(req.frames_data))
    return {"job_id": job_id}


@router.post("/phase-b")
def start_phase_b(req: PhaseBRequest):
    if not req.frames_data:
        raise HTTPException(400, "frames_data ist leer")
    if len(req.betas) != 10:
        raise HTTPException(400, f"betas muss 10 Werte haben, got {len(req.betas)}")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {'status': 'queued', 'progress': 0.0, 'result': None, 'error': None}
    t = threading.Thread(target=_run_phase_b, args=(job_id, req), daemon=True)
    t.start()
    log.info("Phase B job %s gestartet (%d frames)", job_id, len(req.frames_data))
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} nicht gefunden")
    return {
        'status':      job['status'],
        'progress':    job['progress'],
        'result':      job.get('result'),
        'error':       job.get('error'),
        'latest_loss': job.get('latest_loss', 0.0),
    }


@router.delete("/{job_id}")
def delete_job(job_id: str):
    _jobs.pop(job_id, None)
    return {"deleted": job_id}
