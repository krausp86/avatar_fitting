"""
Shape fitting job runner.

Flow:
  1. Sample frame indices for each DetectedPerson (clip) in the group
  2. For each sampled frame: check PersonFrameKeypoints cache, else call pose-worker
  3. Collect all frames_data and call run_shape_fit()
  4. Save fitted parameters + T-pose render to PersonShape
"""
from __future__ import annotations

import base64
import logging
import os
import threading

import cv2

from django.utils import timezone

log = logging.getLogger(__name__)

# Latest preview image per shape_id – stored outside shape.log to avoid
# bloating the DB with large base64 strings on every progress tick.
_latest_preview: dict = {}   # shape_id (str) → 'data:image/jpeg;base64,...'

# Cancel flags: shape_id (str) → threading.Event  (set = cancel requested)
_cancel_flags: dict = {}

# Active threads per group_id (str) → threading.Thread
_active_threads: dict = {}
_threads_lock = threading.Lock()


def start_shape_fit_job(group) -> 'PersonShape':
    """Create or reset PersonShape for group and start background fitting thread."""
    from .models import PersonShape

    group_id = str(group.pk)

    # Prevent duplicate jobs for the same group
    with _threads_lock:
        existing = _active_threads.get(group_id)
        if existing and existing.is_alive():
            log.warning("Shape fit already running for group %s – ignoring start request", group_id)
            return PersonShape.objects.get(group=group)

    shape, _ = PersonShape.objects.get_or_create(group=group)
    shape.betas          = []
    shape.log            = []
    shape.fit_quality    = {}
    shape.render_b64     = ''
    shape.error          = ''
    shape.fitted_at      = None
    shape.status         = PersonShape.Status.RUNNING
    shape.save()

    cancel_evt = threading.Event()
    with _threads_lock:
        _cancel_flags[str(shape.id)] = cancel_evt

    thread = threading.Thread(
        target=_run_shape_job, args=(str(shape.id), cancel_evt), daemon=True)
    with _threads_lock:
        _active_threads[group_id] = thread
    thread.start()
    return shape


def _run_shape_job(shape_id: str, cancel_evt: threading.Event) -> None:
    """Main job runner – executes in daemon thread."""
    from django.db import close_old_connections
    from .models import PersonShape, PersonFrameKeypoints, ShapeFitSettings
    from .fitting.shape_fit import run_shape_fit, render_tpose

    close_old_connections()

    shape    = PersonShape.objects.get(pk=shape_id)
    group    = shape.group
    cfg      = ShapeFitSettings.get()
    persons  = list(group.persons.select_related('video').order_by('id'))
    n_clips  = len(persons)

    try:
        _log(shape, f'Shape-Fit gestartet – {n_clips} Clip(s)', save=True)

        # ── Step 1: collect / compute keypoints ──────────────────────────────
        all_frames: list[dict] = []

        for clip_idx, person in enumerate(persons):
            video_path = person.video.path
            if not os.path.exists(video_path):
                _log(shape, f'⚠ Video nicht gefunden: {os.path.basename(video_path)}', save=True)
                continue

            W, H = _video_dimensions(video_path)
            indices = _sample_indices(
                person.frame_start, person.frame_end,
                cfg.frames_per_clip, cfg.frame_stride,
            )

            cached_qs = {kp.frame_idx: kp for kp in
                         PersonFrameKeypoints.objects.filter(person=person,
                                                             frame_idx__in=indices)}
            n_computed = 0
            for order, frame_idx in enumerate(indices):
                if frame_idx in cached_qs:
                    kp = cached_qs[frame_idx]
                else:
                    kp_data = _compute_keypoints(video_path, frame_idx)
                    if kp_data is None:
                        continue
                    kp = PersonFrameKeypoints.objects.create(
                        person=person,
                        frame_idx=frame_idx,
                        body_landmarks=kp_data['body_landmarks'],
                        rtm_landmarks=kp_data['rtm_landmarks'],
                        seg_mask_b64=kp_data.get('seg_mask_b64', ''),
                    )
                    n_computed += 1

                all_frames.append({
                    'W':              W,
                    'H':              H,
                    'body_landmarks': kp.body_landmarks,
                    'clip_id':        str(person.id),
                    'sample_order':   order,
                })

            _log(shape,
                 f'Clip {clip_idx + 1}/{n_clips}: {len(indices)} Frames '
                 f'({n_computed} neu berechnet)',
                 save=True)

        if not all_frames:
            raise RuntimeError('Keine verwertbaren Keypoints gefunden – Pose-Worker erreichbar?')

        _log(shape,
             f'Keypoint-Extraktion abgeschlossen: {len(all_frames)} Frames aus '
             f'{n_clips} Clip(s)',
             save=True)
        # Check VPoser availability and log it (uses cache, no double-load)
        import torch as _torch
        from .fitting.shape_fit import _try_load_vposer as _tvp
        _dev = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        try:
            import human_body_prior  # noqa: F401
            _hbp_ok = True
        except ImportError:
            _hbp_ok = False
        if not _hbp_ok:
            _log(shape, '⚠ human-body-prior nicht installiert – Container Rebuild nötig', save=True)
            _vp = None
        else:
            try:
                _vp = _tvp(_dev)
                if _vp:
                    _log(shape, '✓ VPoser V02_05 geladen', save=True)
                else:
                    from django.conf import settings as _dj
                    _vdir = getattr(_dj, 'VPOSER_MODEL_DIR', '?')
                    _log(shape, f'⚠ VPoser Weights nicht gefunden: {_vdir}', save=True)
            except Exception as _vp_exc:
                _log(shape, f'⚠ VPoser Fehler: {_vp_exc}', save=True)
                _vp = None
        _log(shape, 'Starte SMPL-X Shape-Fit…', save=True)

        if cancel_evt.is_set():
            raise RuntimeError('Abgebrochen vor Shape-Fit')

        # ── Step 2: shape fit ─────────────────────────────────────────────────
        def _progress(info: dict) -> None:
            if cancel_evt.is_set():
                raise RuntimeError('Manuell abgebrochen')
            _log_progress(shape, info)

        result = run_shape_fit(
            all_frames,
            n_phase1_epochs=cfg.n_phase1_epochs,
            n_phase2_epochs=cfg.n_phase2_epochs,
            progress_cb=_progress,
        )

        # Free GPU memory immediately after fitting – the batch-size-specific
        # SMPL-X model is no longer needed and the VRAM must be released before
        # the T-pose render (which uses a different batch_size=1 model).
        from .fitting.shape_fit import _smplx_cache as _shape_cache
        _shape_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # ── Step 3: T-pose render ─────────────────────────────────────────────
        _log(shape, 'Rendere T-Pose Vorschau…', save=True)
        render_bgr = render_tpose(result['betas'])
        _, buf = cv2.imencode('.jpg', render_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        render_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

        # ── Step 4: persist ───────────────────────────────────────────────────
        shape.betas          = result['betas']
        shape.hip_correction = result['hip_correction']
        shape.focal_scale    = result['focal_scale']
        shape.fit_quality    = {
            'kp_loss':  result['kp_loss'],
            'n_frames': result['n_frames'],
            'n_clips':  result['n_clips'],
        }
        shape.render_b64 = render_b64
        shape.status     = PersonShape.Status.DONE
        shape.fitted_at  = timezone.now()
        _log(shape, f'✓ Shape-Fit abgeschlossen  '
                    f'loss={result["kp_loss"]:.4f}  '
                    f'focal={result["focal_scale"]:.3f}',
             save=False)
        shape.save()

    except Exception as exc:
        log.exception("Shape fit job %s failed", shape_id)
        shape.status = PersonShape.Status.FAILED
        shape.error  = str(exc)
        _log(shape, f'❌ Fehler: {exc}', save=False)
        shape.save()
    finally:
        _latest_preview.pop(shape_id, None)
        _cancel_flags.pop(shape_id, None)
        group_id = str(shape.group_id)
        with _threads_lock:
            _active_threads.pop(group_id, None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_indices(frame_start: int, frame_end: int,
                    frames_per_clip: int, stride: int) -> list[int]:
    """
    Sample `frames_per_clip` indices with given `stride`, centred in the clip.
    Deduplicates clamped values so we never submit the same frame twice.
    """
    window = (frames_per_clip - 1) * stride
    mid    = frame_start + (frame_end - frame_start) // 2
    start  = mid - window // 2
    raw    = [start + i * stride for i in range(frames_per_clip)]
    return sorted(set(max(frame_start, min(frame_end, idx)) for idx in raw))


def _video_dimensions(video_path: str) -> tuple[int, int]:
    """Return (W, H) of the video; fall back to 1920×1080 on failure."""
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (W or 1920, H or 1080)


def _compute_keypoints(video_path: str, frame_idx: int) -> dict | None:
    """
    Read one video frame and call the pose-worker for combined analysis.
    Returns dict with body_landmarks, rtm_landmarks, seg_mask_b64.
    Returns None on any failure.
    """
    from .detection.backends import combined_analyze

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = max(0, min(total - 1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.warning("Could not read frame %d from %s", frame_idx, video_path)
        return None

    try:
        data = combined_analyze(frame)
    except Exception as exc:
        log.warning("Pose-worker call failed for frame %d: %s", frame_idx, exc)
        return None

    rtm = (data.get('face_landmarks', []) +
           data.get('lhand_landmarks', []) +
           data.get('rhand_landmarks', []))
    return {
        'body_landmarks': data.get('body_landmarks', []),
        'rtm_landmarks':  rtm,
        'seg_mask_b64':   '',   # MediaPipe call can be added here later
    }


def _log(shape, msg: str, *, save: bool = False) -> None:
    """Append an info message to shape.log, optionally persisting."""
    entry = {'type': 'info', 'msg': msg, 'ts': timezone.now().isoformat()}
    shape.log.append(entry)
    _trim_log(shape)
    if save:
        shape.save(update_fields=['log', 'updated_at'])


def _log_progress(shape, info: dict) -> None:
    """Append a fitting-progress entry; save to DB every 40 epochs.

    preview_jpg is stored in _latest_preview (not in shape.log) to avoid
    persisting large base64 strings to the DB on every progress tick.
    """
    preview = info.pop('preview_jpg', None)
    if preview:
        _latest_preview[str(shape.pk)] = preview
    entry = {'type': 'progress', 'ts': timezone.now().isoformat(), **info}
    shape.log.append(entry)
    _trim_log(shape)
    if info.get('epoch_all', 0) % 40 == 0:
        shape.save(update_fields=['log', 'updated_at'])


def _trim_log(shape, max_entries: int = 300) -> None:
    """Keep log from growing unboundedly."""
    if len(shape.log) > max_entries:
        shape.log = shape.log[-200:]
