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
    Stage 0: Fit SMPL-X shape (beta) across all clips of the PersonGroup.
    Result is stored in PersonShape and used by Stage 1 as a fixed initialisation.
    """
    from .models import PersonShape, PersonGroup
    from .shape_tasks import _run_shape_job

    if not avatar.group_id:
        log.warning("Stage 0: avatar has no group – shape fit skipped")
        progress_cb({'type': 'progress', 'epoch': 1, 'total_epochs': 1, 'loss': 0})
        return

    group = PersonGroup.objects.get(pk=avatar.group_id)
    shape, _ = PersonShape.objects.get_or_create(group=group)
    shape.betas       = []
    shape.log         = []
    shape.fit_quality = {}
    shape.render_b64  = ''
    shape.error       = ''
    shape.fitted_at   = None
    shape.status      = PersonShape.Status.RUNNING
    shape.save()

    # Run synchronously in this thread (shape_tasks handles its own DB updates)
    _run_shape_job(str(shape.id))

    shape.refresh_from_db()
    if shape.status != PersonShape.Status.DONE:
        raise RuntimeError(f"Shape-Fit fehlgeschlagen: {shape.error}")

    log.info("Stage 0 complete: kp_loss=%.4f  focal=%.3f",
             shape.fit_quality.get('kp_loss', 0),
             shape.focal_scale)
    progress_cb({'type': 'progress', 'epoch': 1, 'total_epochs': 1,
                 'loss': shape.fit_quality.get('kp_loss', 0)})


def _run_stage1(avatar, config: dict, progress_cb) -> None:
    # Free the shape_fit SMPL-X model (Stage 0 residual) before loading
    # stage1's model – both are ~300-700 MB and must not coexist in RAM.
    from .fitting import shape_fit as _sf
    _sf._smplx_cache.clear()

    from .fitting.stage1 import run_stage1, save_stage1_result, generate_stage1_previews
    from .fitting import stage1 as _s1
    from .fitting.pose_smoothing import store_and_smooth_poses
    try:
        result = run_stage1(avatar, config, progress_cb=progress_cb)
        save_stage1_result(result, avatar.data_path)
        try:
            store_and_smooth_poses(avatar, result)
        except Exception:
            log.exception("Pose smoothing failed for avatar %s — continuing without", avatar.id)
        generate_stage1_previews(result, avatar, avatar.data_path)
        log.info("Stage 1 complete for avatar %s", avatar.id)
    finally:
        # Release stage1 model after save+previews are done
        _s1._smplx_cache.clear()


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
