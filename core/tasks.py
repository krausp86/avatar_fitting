"""
Fitting task runner.
In production: replace threading.Thread with Celery or RQ.
Progress is sent via Django Channels WebSocket to the browser.
"""
import logging
import os
import threading
from django.conf import settings
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .models import Avatar, FittingJob

log = logging.getLogger(__name__)


def start_fitting_job(avatar: Avatar, config: dict) -> FittingJob:
    job = FittingJob.objects.create(avatar=avatar, status='queued')
    avatar.status = Avatar.Status.FITTING
    avatar.save()

    thread = threading.Thread(target=_run_fitting, args=(job.id, config), daemon=True)
    thread.start()
    return job


def _send_progress(job_id, data: dict):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'job_{job_id}',
        {'type': 'fitting_progress', 'data': data}
    )


def _run_fitting(job_id, config: dict):
    from .models import FittingJob, Avatar

    job = FittingJob.objects.get(pk=job_id)
    job.status     = 'running'
    job.started_at = timezone.now()
    job.save()

    # Ensure avatar has a data folder
    avatar = job.avatar
    if not avatar.data_path:
        safe_name      = "".join(c if c.isalnum() or c in '-_' else '_' for c in avatar.name)
        avatar.data_path = os.path.join(
            getattr(settings, 'AVATAR_DATA_ROOT', 'avatar_data'),
            f"{safe_name}_v{avatar.version}_{avatar.id}",
        )
        avatar.save()

    stages = config.get('stages', ['0', '1', '1.5', '2', '2.5', '3', '4'])
    n_stages = len(stages)

    STAGE_NAMES = {
        '0':   'Shape Fit',
        '1':   'SMPL-X Fitting',
        '1.5': 'Face Refinement',
        '2':   'Static Offsets',
        '2.5': 'Texture & Appearance',
        '3':   'Cage Initialisation',
        '4':   'Physics Parameters',
        '5':   'Prior MLP',
    }

    try:
        for s_idx, stage in enumerate(stages):
            job.current_stage = stage
            job.save()
            stage_name = STAGE_NAMES.get(stage, f'Stage {stage}')

            # ── Progress callback shared by all stages ─────────────────────
            def _progress_cb(payload: dict, _s_idx=s_idx):
                epoch        = payload.get('epoch', 0)
                total_epochs = payload.get('total_epochs', 1)
                job.progress = (_s_idx + epoch / max(total_epochs, 1)) / n_stages
                entry = {
                    'stage': stage,
                    'epoch': epoch,
                    'loss':  payload.get('loss', 0),
                }
                if payload.get('note'):
                    entry['note'] = payload['note']
                job.log.append(entry)
                job.save()
                payload['person_id'] = str(avatar.id)
                _send_progress(job_id, payload)

            # ── Dispatch to stage implementation ──────────────────────────
            if stage == '0':
                _run_stage0(avatar, config, _progress_cb)

            elif stage == '1':
                _run_stage1(avatar, config, _progress_cb)

            elif stage == '1.5':
                _run_stub(stage, stage_name, _progress_cb)   # TODO: implement

            elif stage == '2':
                _run_stage2(avatar, config, _progress_cb)

            elif stage == '2.5':
                _run_stub(stage, stage_name, _progress_cb)   # TODO: implement

            elif stage == '3':
                _run_stub(stage, stage_name, _progress_cb)   # TODO: implement

            elif stage == '4':
                _run_stub(stage, stage_name, _progress_cb)   # TODO: implement

            elif stage == '5':
                _run_stub(stage, stage_name, _progress_cb)   # TODO: implement

            else:
                log.warning("Unknown stage %r – skipping", stage)

            _send_progress(job_id, {
                'type':       'stage_complete',
                'stage':      stage,
                'stage_name': stage_name,
            })

        # All done
        job.status      = 'done'
        job.progress    = 1.0
        job.finished_at = timezone.now()
        job.save()

        avatar.status = Avatar.Status.DONE
        avatar.save()

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'job_{job_id}',
            {'type': 'fitting_complete', 'data': {'job_id': str(job_id)}}
        )

    except Exception as e:
        log.exception("Fitting job %s failed", job_id)
        job.status      = 'failed'
        job.error       = str(e)
        job.finished_at = timezone.now()
        job.save()
        avatar.status = Avatar.Status.FAILED
        avatar.save()
        _send_progress(job_id, {'type': 'error', 'error': str(e)})


# ── Stage implementations ──────────────────────────────────────────────────────

def _run_stage0(avatar, config: dict, progress_cb) -> None:
    """
    Stage 0 (NEU): Phase A — Beta-Schätzung via GMoF + VPoser über alle Clips.
    Ersetzt den alten shape_fit/_run_shape_job Ansatz.
    """
    from .fitting.fit_smplx import run_phase_a, save_phase_a_result, _collect_frames, _fetch_smplx_init_for_frame, _read_frame_for_fd

    if not avatar.group_id:
        log.warning("Stage 0: avatar has no group – skipped")
        progress_cb({'type': 'progress', 'epoch': 1, 'total_epochs': 1, 'loss': 0})
        return

    from .models import PersonGroup, PersonFrameKeypoints
    group = PersonGroup.objects.get(pk=avatar.group_id)

    max_frames_a   = int(config.get('max_frames_a', 150))
    n_a1_epochs    = int(config.get('n_a1_epochs',  300))
    n_a2_epochs    = int(config.get('n_a2_epochs',  200))
    use_smplx_init = bool(config.get('use_smplx_init', True))
    total_epochs   = n_a1_epochs + n_a2_epochs

    # ── Keypoint-Extraktion: PersonFrameKeypoints befüllen wenn leer ──────────
    from .shape_tasks import _compute_keypoints, _sample_indices
    from .models import ShapeFitSettings
    cfg = ShapeFitSettings.get()
    for person in group.persons.select_related('video').all():
        video_path = person.video.path
        if not os.path.exists(video_path):
            log.warning("Stage 0: Video nicht gefunden: %s", video_path)
            continue
        indices = _sample_indices(
            person.frame_start, person.frame_end,
            cfg.frames_per_clip, cfg.frame_stride,
        )
        cached = {kp.frame_idx for kp in
                  PersonFrameKeypoints.objects.filter(person=person, frame_idx__in=indices)}
        n_new = 0
        for frame_idx in indices:
            if frame_idx in cached:
                continue
            kp_data = _compute_keypoints(video_path, frame_idx)
            if kp_data:
                PersonFrameKeypoints.objects.get_or_create(
                    person=person,
                    frame_idx=frame_idx,
                    defaults={
                        'body_landmarks': kp_data['body_landmarks'],
                        'rtm_landmarks':  kp_data['rtm_landmarks'],
                        'seg_mask_b64':   kp_data.get('seg_mask_b64', ''),
                    }
                )
                n_new += 1
        log.info("Stage 0 keypoints: person %s – %d neu, %d gecacht",
                 person.id, n_new, len(cached))

    frames = _collect_frames(group, max_frames=max_frames_a, max_total=max_frames_a)
    if not frames:
        raise RuntimeError("Stage 0: keine Frames mit Keypoints in der PersonGroup")

    smplx_inits = [None] * len(frames)
    if use_smplx_init:
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
        log.info("Stage 0: SMPLer-X init %d/%d frames", n_ok, len(frames))

    def _cb(data):
        progress_cb({
            'type':         'progress',
            'stage':        '0',
            'stage_name':   'Beta Fitting (Phase A)',
            'epoch':        data.get('epoch_all', data.get('epoch', 0)),
            'total_epochs': total_epochs,
            'loss':         data.get('loss', 0.0),
            'loss_terms':   data.get('loss_terms', {}),
        })

    phase_a = run_phase_a(
        frames_data     = frames,
        smplx_inits     = smplx_inits,
        n_phase1_epochs = n_a1_epochs,
        n_phase2_epochs = n_a2_epochs,
        progress_cb     = _cb,
    )
    save_phase_a_result(phase_a, group)

    log.info("Stage 0 complete: kp_loss=%.4f  focal_scale=%.3f  n_frames=%d",
             phase_a['kp_loss'], phase_a['focal_scale'], phase_a['n_frames'])
    progress_cb({
        'type': 'progress', 'epoch': total_epochs, 'total_epochs': total_epochs,
        'loss': phase_a['kp_loss'],
        'loss_terms': {'kp': phase_a['kp_loss']},
    })


def _run_stage1(avatar, config: dict, progress_cb) -> None:
    """
    Stage 1 (NEU): Phase B — Pro-Frame Full SMPL-X (body + face + hands).
    Beta kommt aus Stage 0 (PersonShape.betas).
    """
    from .fitting.fit_smplx import (
        run_phase_b_frame, save_phase_b_results,
        _collect_frames, _load_smplx_phase_b, _try_load_vposer,
        _fetch_smplx_init_for_frame, _read_frame_for_fd,
        _smplx_cache,
    )
    import torch

    if not avatar.group_id:
        log.warning("Stage 1: avatar has no group – skipped")
        return

    from .models import PersonGroup, PersonShape
    group = PersonGroup.objects.get(pk=avatar.group_id)

    try:
        shape = PersonShape.objects.get(group=group)
        betas = shape.betas
    except PersonShape.DoesNotExist:
        raise RuntimeError("Stage 1: PersonShape fehlt – bitte zuerst Stage 0 ausführen")

    max_frames_b   = int(config.get('max_frames_b',   200))
    n_b1_epochs    = int(config.get('n_b1_epochs',    100))
    n_b2_epochs    = int(config.get('n_b2_epochs',    150))
    use_smplx_init = bool(config.get('use_smplx_init', True))

    frames = _collect_frames(group, max_frames=max_frames_b)
    if not frames:
        raise RuntimeError("Stage 1: keine Frames")

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smplx_b = _load_smplx_phase_b(device)
    vposer  = _try_load_vposer(device)
    betas_t = torch.tensor([betas], dtype=torch.float32, device=device)

    n_b = len(frames)
    frame_results = []

    try:
        for i, fd in enumerate(frames):
            phase_a_pose = {'body_pose': [0.0]*63, 'global_orient': [0.0]*3, 'transl': [0.0, 0.0, 3.0]}

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

            if i % 10 == 0:
                progress_cb({
                    'type':         'progress',
                    'stage':        '1',
                    'stage_name':   'Per-Frame SMPL-X (Phase B)',
                    'epoch':        i + 1,
                    'total_epochs': n_b,
                    'loss':         result.loss_body,
                    'loss_terms':   {'body': result.loss_body, 'face': result.loss_face},
                    'notes':        f'Frame {i+1}/{n_b}',
                })
    finally:
        _smplx_cache.clear()

    save_phase_b_results(frame_results)

    # Write poses.npz + metadata.json so Stage 2 (and avatar_detail) can read them
    _save_stage1_files(avatar, frames, frame_results, betas)

    try:
        from .fitting.pose_smoothing import smooth_new_fields
        person_ids = list({r.person_id for r in frame_results})
        smooth_new_fields(person_ids)
    except Exception:
        log.exception("Phase B smoothing fehlgeschlagen – weiter ohne")

    log.info("Stage 1 complete: %d frames für avatar %s", n_b, avatar.id)
    last_loss = frame_results[-1].loss_body if frame_results else 0.0
    progress_cb({
        'type': 'progress', 'epoch': n_b, 'total_epochs': n_b,
        'loss': last_loss,
    })


def _save_stage1_files(avatar, frames: list, frame_results, betas: list) -> None:
    """Write poses.npz + metadata.json to avatar.data_path after Phase B."""
    import numpy as np
    import json

    data_path = avatar.data_path
    os.makedirs(data_path, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_path, 'poses.npz'),
        theta_t         = np.array([r.body_pose     for r in frame_results], dtype=np.float32),
        global_orient_t = np.array([r.global_orient for r in frame_results], dtype=np.float32),
        transl_t        = np.array([r.transl         for r in frame_results], dtype=np.float32),
    )

    meta_path = os.path.join(data_path, 'metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    fd0 = frames[0] if frames else {}
    W, H = fd0.get('W', 1920), fd0.get('H', 1080)
    meta.update({
        'beta': betas,
        'camera_intrinsics': meta.get('camera_intrinsics', {
            'fx': float(max(W, H)), 'fy': float(max(W, H)),
            'cx': W / 2.0,          'cy': H / 2.0,
        }),
        'stage1_n_frames': len(frame_results),
    })
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info("Stage 1 files written → %s  (%d frames, beta_rms=%.3f)",
             data_path, len(frame_results),
             float(sum(b**2 for b in betas) / len(betas)) ** 0.5 if betas else 0)


def _run_stage2(avatar, config: dict, progress_cb) -> None:
    from .fitting.stage2 import run_stage2, save_stage2_result
    result = run_stage2(avatar, config, progress_cb=progress_cb)
    save_stage2_result(result, avatar.data_path)
    log.info("Stage 2 complete for avatar %s", avatar.id)


def _run_stub(stage: str, stage_name: str, progress_cb) -> None:
    """Placeholder for stages not yet implemented."""
    import math, random, time
    n_epochs = 100
    for epoch in range(0, n_epochs + 1, 10):
        decay = math.exp(-epoch / 40)
        loss  = decay + random.uniform(0, 0.01)
        progress_cb({
            'type':         'progress',
            'stage':        stage,
            'stage_name':   stage_name,
            'epoch':        epoch,
            'total_epochs': n_epochs,
            'loss':         round(loss, 4),
            'loss_terms':   {'stub': round(loss, 4)},
            'preview_jpg':  None,
            'mesh_obj':     None,
            'texture_jpg':  None,
            'heatmap_jpg':  None,
        })
        time.sleep(0.02)   # remove when implemented
