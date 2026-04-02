import os
import json
import functools
from pathlib import Path
import numpy as np

# Module-level progress store for single-frame SMPL-X fitting (keyed by fit_id).
# Entries are created by the fit view and polled by video_debug_smplx_progress.
_smplx_fit_progress: dict = {}

# Module-level pose-param cache for temporal smoothing in pose debug.
# Key: (video_pk_str, frame_idx_int) → pose_params dict (source, theta/body_pose, etc.)
_debug_pose_cache: dict = {}

from django.shortcuts import render, get_object_or_404, redirect, aget_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from django.utils import timezone
from django.db.models import Count, Max

from .models import VideoSource, DetectedPerson, PersonGroup, Avatar, AvatarEdit, FittingJob
from .scanner import scan_video_folder
from .tasks import start_fitting_job


# ─── Dashboard ───────────────────────────────────────────────────────────────

def dashboard(request):
    ctx = {
        'video_count':   VideoSource.objects.count(),
        'person_count':  DetectedPerson.objects.count(),
        'group_count':   PersonGroup.objects.count(),
        'avatar_count':  Avatar.objects.count(),
        'recent_avatars': Avatar.objects.order_by('-updated_at')[:6],
        'running_jobs':   FittingJob.objects.filter(status='running').select_related('avatar'),
    }
    return render(request, 'core/dashboard.html', ctx)


# ─── Videos ──────────────────────────────────────────────────────────────────

def video_list(request):
    folders = (VideoSource.objects
               .values('folder')
               .annotate(count=Count('id'))
               .order_by('folder'))
    videos  = VideoSource.objects.prefetch_related('persons').order_by('folder', 'filename')
    return render(request, 'core/video_list.html', {
        'videos': videos, 'folders': folders,
        'scan_root': settings.VIDEO_SCAN_ROOT,
    })


@require_POST
def scan_videos(request):
    folder = request.POST.get('folder', settings.VIDEO_SCAN_ROOT)
    added  = scan_video_folder(folder)
    return JsonResponse({'added': added, 'folder': folder})


@require_POST
def detect_persons(request, pk):
    import threading
    video = get_object_or_404(VideoSource, pk=pk)
    if video.detection_status == 'detecting':
        return JsonResponse({'status': 'already_running'})
    video.detection_status = 'detecting'
    video.save()

    def _run(video_id):
        import logging
        from django.db import close_old_connections
        from .models import VideoSource
        from .scanner import detect_persons_for_video
        close_old_connections()
        log = logging.getLogger(__name__)
        try:
            v = VideoSource.objects.get(pk=video_id)
            detect_persons_for_video(v)
            v.detection_status = 'done'
            v.save()
        except Exception as e:
            log.exception("Detection failed for video %s", video_id)
            try:
                close_old_connections()
                v = VideoSource.objects.get(pk=video_id)
                v.detection_status = 'failed'
                v.save()
            except Exception:
                pass

    threading.Thread(target=_run, args=(video.pk,), daemon=True).start()
    return JsonResponse({'status': 'started', 'video_id': str(video.id)})


def detect_persons_status(request, pk):
    video = get_object_or_404(VideoSource, pk=pk)
    return JsonResponse({
        'status': video.detection_status,
        'person_count': video.persons.count(),
    })


@require_POST
def detect_persons_cancel(request, pk):
    video = get_object_or_404(VideoSource, pk=pk)
    if video.detection_status == 'detecting':
        video.detection_status = 'pending'
        video.save()
    return JsonResponse({'status': video.detection_status})


def video_detail(request, pk):
    video   = get_object_or_404(VideoSource, pk=pk)
    persons = video.persons.all().order_by('-frame_count')
    return render(request, 'core/video_detail.html', {'video': video, 'persons': persons})


def _read_video_frame(video_path, frame_param):
    """Open video, read one frame. Returns (frame_bgr, frame_idx, total_frames, fps) or raises."""
    import cv2, random as rnd
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Video kann nicht geöffnet werden')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if frame_param is None:
        frame_idx = rnd.randint(0, max(0, total_frames - 1))
    else:
        frame_idx = max(0, min(int(frame_param), total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Frame {frame_idx} konnte nicht gelesen werden')
    return frame, frame_idx, total_frames, fps


def _frame_to_b64(img, quality=88):
    import base64, cv2
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def _render_segmentation(frame_bgr, mask_float):
    import cv2
    import numpy as np
    seg = frame_bgr.copy()
    mask_bin = (mask_float > 0.5).astype(np.uint8)
    overlay = seg.copy()
    overlay[mask_bin == 1] = [0, 160, 80]
    seg = cv2.addWeighted(seg, 0.55, overlay, 0.45, 0)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 100), 2)
    return seg


async def video_debug_frame(request, pk):
    """Original-frame + MediaPipe segmentation + landmarks (used by debug tab)."""
    import asyncio
    from django.core.cache import cache
    from .detection.backends import MediaPipeBackend

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    try:
        frame, frame_idx, total_frames, fps = await asyncio.to_thread(
            _read_video_frame, video.path, request.GET.get('frame'))
    except RuntimeError as e:
        return JsonResponse({'error': str(e)}, status=500)

    h, w = frame.shape[:2]
    result = {
        'frame_idx': frame_idx, 'total_frames': total_frames,
        'fps': round(fps, 2), 'width': w, 'height': h,
        'detection': False, 'landmark_data': [],
        'original': _frame_to_b64(frame),
        'segmentation': None, 'landmarks': None, 'error': None,
    }

    cache_key = f'pose_debug:mediapipe:{pk}:{frame_idx}'
    cached = await cache.aget(cache_key)
    if cached:
        result.update(cached)
        return JsonResponse(result)

    try:
        data = await MediaPipeBackend.async_analyze(frame, include_segmentation=True)
        payload = {
            'segmentation':  _frame_to_b64(data['segmentation'] if data['segmentation'] is not None else frame),
            'landmarks':     _frame_to_b64(data['render']),
            'detection':     data['detection'],
            'landmark_data': data['landmarks'],
        }
        await cache.aset(cache_key, payload)
        result.update(payload)
    except Exception as exc:
        log.exception("video_debug_frame failed for video %s frame %s", pk, frame_idx)
        result['error'] = str(exc)
        result['segmentation'] = result['original']
        result['landmarks']    = result['original']

    return JsonResponse(result)


async def video_debug_frame_combined(request, pk):
    """Run ViTPose + RTMPose on a frame and return the combined body+face+hands render."""
    import asyncio
    from django.core.cache import cache
    from .detection.backends import async_combined_analyze, RTMPoseBackend, ViTPoseBackend

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    frame_param = request.GET.get('frame')
    if frame_param is None:
        return JsonResponse({'error': 'frame parameter fehlt'}, status=400)

    frame, frame_idx, _, _ = await asyncio.to_thread(
        _read_video_frame, video.path, frame_param)

    cache_key = f'pose_debug:combined:{pk}:{frame_idx}'
    cached = await cache.aget(cache_key)
    if cached:
        return JsonResponse(cached)

    if not (RTMPoseBackend.available and ViTPoseBackend.available):
        return JsonResponse({'error': 'RTMPose und/oder ViTPose nicht verfügbar'}, status=503)

    try:
        data = await async_combined_analyze(frame)
        response = {
            'frame_idx':      frame_idx,
            'detection':      data['detection'],
            'body_count':     data['body_count'],
            'face_count':     data['face_count'],
            'lhand_count':    data['lhand_count'],
            'rhand_count':    data['rhand_count'],
            'render':         _frame_to_b64(data['render']),
        }
        await cache.aset(cache_key, response)
        return JsonResponse(response)
    except Exception as exc:
        log.exception("video_debug_frame_combined failed: video=%s frame=%s", pk, frame_param)
        return JsonResponse({'error': str(exc)}, status=500)


async def video_debug_frame_smplx(request, pk):
    """
    Fit SMPL-X to ViTPose keypoints for a single debug frame and return
    a rendered mesh overlay image.

    GET params:
        frame  – frame index (required)
        fx, fy, cx, cy – camera intrinsics in pixels (optional; estimated if absent)
    """
    import asyncio
    from .detection.backends import async_combined_analyze, RTMPoseBackend, ViTPoseBackend, MediaPipeBackend
    from .fitting.single_frame_fit import fit_and_render

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    frame_param = request.GET.get('frame')
    if frame_param is None:
        return JsonResponse({'error': 'frame parameter fehlt'}, status=400)

    frame, frame_idx, _, _ = await asyncio.to_thread(
        _read_video_frame, video.path, frame_param)

    H, W = frame.shape[:2]

    # Camera intrinsics from query params or auto-estimate
    try:
        fx = float(request.GET['fx'])
        fy = float(request.GET['fy'])
        cx = float(request.GET['cx'])
        cy = float(request.GET['cy'])
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    except (KeyError, ValueError):
        f = max(W, H) * 1.2
        intrinsics = {'fx': f, 'fy': f, 'cx': W / 2.0, 'cy': H / 2.0}

    if not (RTMPoseBackend.available and ViTPoseBackend.available):
        return JsonResponse({'error': 'RTMPose und/oder ViTPose nicht verfügbar – '
                                      'Pose-Worker läuft nicht?'}, status=503)

    # Get combined landmarks + MediaPipe segmentation mask in parallel
    try:
        data, mp_data = await asyncio.gather(
            async_combined_analyze(frame),
            MediaPipeBackend.async_analyze(frame, include_segmentation=True),
            return_exceptions=True,
        )
        if isinstance(data, Exception):
            raise data
        if isinstance(mp_data, Exception):
            log.warning("video_debug_frame_smplx: mediapipe segmentation failed: %s", mp_data)
            mp_data = {}
    except Exception as exc:
        log.exception("video_debug_frame_smplx: combined_analyze failed video=%s frame=%s", pk, frame_param)
        return JsonResponse({'error': str(exc)}, status=500)

    body_landmarks = data.get('body_landmarks', [])
    if not body_landmarks:
        return JsonResponse({'error': 'Keine Körper-Landmarks gefunden (ViTPose)'}, status=422)

    seg_mask = mp_data.get('mask') if isinstance(mp_data, dict) else None

    # Use pre-fitted PersonShape betas + focal_scale if available.
    fixed_betas = None
    try:
        from .models import PersonShape
        from asgiref.sync import sync_to_async
        shape = await sync_to_async(
            lambda: PersonShape.objects.filter(
                group__persons__video=video, status='done',
            ).exclude(betas=[]).first()
        )()
        if shape:
            fixed_betas = shape.betas
            # Apply the same 1.2-base as shape_fit so focal lengths are consistent.
            if shape.focal_scale and shape.focal_scale != 1.0:
                f_ps = max(W, H) * 1.2 * float(shape.focal_scale)
                intrinsics = {'fx': f_ps, 'fy': f_ps, 'cx': W / 2.0, 'cy': H / 2.0}
                log.info("video_debug_frame_smplx: PersonShape focal=%.1f px (scale=%.3f)",
                         f_ps, shape.focal_scale)
            log.info("video_debug_frame_smplx: using PersonShape betas (group %s)", shape.group_id)
    except Exception:
        pass

    n_orient_epochs = int(request.GET.get('n_orient_epochs', 600))
    n_pose_epochs   = int(request.GET.get('n_pose_epochs',   900))

    from .fitting.romp_render import romp_infer_params
    from django.conf import settings as _settings

    # ── ROMP als Orientierungs-Init für den SMPL-X Optimizer ──────────────────
    romp_raw = await asyncio.to_thread(
        romp_infer_params, frame,
        getattr(_settings, 'ROMP_MODEL_PATH', None),
        getattr(_settings, 'ROMP_SMPL_PATH',  None),
    )
    romp_init_smplx = None
    if romp_raw is not None:
        romp_init_smplx = {'thetas': [romp_raw['theta']], 'beta': romp_raw['beta']}
        log.info("video_debug_frame_smplx: ROMP orient als Init verfügbar")
    else:
        log.info("video_debug_frame_smplx: ROMP nicht verfügbar — PnP+Winding als Fallback")

    # ── SMPL-X Gradient-Optimizer ─────────────────────────────────────────────
    fit_id = request.GET.get('fit_id', '')
    if fit_id:
        _smplx_fit_progress[fit_id] = {'status': 'starting', 'epoch_all': 0,
                                        'total_all': n_orient_epochs + n_pose_epochs}

    def progress_cb(info):
        if fit_id:
            _smplx_fit_progress[fit_id] = {'status': 'running', **info}

    try:
        rendered_bgr, quality = await asyncio.to_thread(
            functools.partial(fit_and_render, frame, body_landmarks, intrinsics,
                              seg_mask=seg_mask, fixed_betas=fixed_betas,
                              romp_init=romp_init_smplx,
                              n_orient_epochs=n_orient_epochs,
                              n_pose_epochs=n_pose_epochs,
                              progress_cb=progress_cb))
    except Exception as exc:
        if fit_id:
            _smplx_fit_progress.pop(fit_id, None)
        log.exception("video_debug_frame_smplx: fit_and_render failed video=%s frame=%s", pk, frame_param)
        return JsonResponse({'error': str(exc)}, status=500)

    if fit_id:
        _smplx_fit_progress.pop(fit_id, None)

    _debug_pose_cache[(str(pk), frame_idx)] = quality.pop('pose_params', None)
    phase_renders = [
        {'label': label, 'img': _frame_to_b64(img)}
        for label, img in quality.pop('phase_renders', [])
    ]
    return JsonResponse({
        'frame_idx':     frame_idx,
        'render':        _frame_to_b64(rendered_bgr),
        'quality':       quality,
        'intrinsics':    intrinsics,
        'phase_renders': phase_renders,
        'romp':          None,
    })


async def video_debug_smplx_progress(request, pk):
    """Return current SMPL-X fitting progress for a given fit_id."""
    fit_id = request.GET.get('fit_id', '')
    progress = _smplx_fit_progress.get(fit_id)
    if progress is None:
        return JsonResponse({'status': 'unknown'})
    return JsonResponse(progress)


async def video_debug_frame_smplx_smooth(request, pk):
    """
    Fit target frame + last N past frames, apply Savitzky-Golay, return smoothed render.

    GET params:
        frame      – target frame index (required)
        n_past     – number of past frames to include (default 5)
        k_stride   – frame stride between history frames (default 3)
        sg_window  – Savitzky-Golay window length, must be odd (default 11)
        sg_poly    – polynomial order (default 3)
        fx,fy,cx,cy – camera intrinsics (optional)
    """
    import asyncio, functools
    from scipy.signal import savgol_filter
    from .detection.backends import async_combined_analyze, RTMPoseBackend, ViTPoseBackend, MediaPipeBackend
    from .fitting.single_frame_fit import fit_and_render, render_smplx_from_params
    from .fitting.romp_render import romp_infer_params, render_romp_from_params
    from django.conf import settings as _settings

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    frame_param = request.GET.get('frame')
    if frame_param is None:
        return JsonResponse({'error': 'frame parameter fehlt'}, status=400)

    n_past    = int(request.GET.get('n_past',    5))
    k_stride  = int(request.GET.get('k_stride',  3))
    sg_window = int(request.GET.get('sg_window', 11))
    sg_poly   = int(request.GET.get('sg_poly',   3))

    target_frame, target_idx, _, _ = await asyncio.to_thread(
        _read_video_frame, video.path, frame_param)
    H, W = target_frame.shape[:2]

    try:
        fx = float(request.GET['fx'])
        fy = float(request.GET['fy'])
        cx = float(request.GET['cx'])
        cy = float(request.GET['cy'])
        intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    except (KeyError, ValueError):
        f = max(W, H) * 1.2
        intrinsics = {'fx': f, 'fy': f, 'cx': W / 2.0, 'cy': H / 2.0}

    # Fixed betas from PersonShape if available
    fixed_betas = None
    try:
        from .models import PersonShape
        from asgiref.sync import sync_to_async
        shape = await sync_to_async(
            lambda: PersonShape.objects.filter(
                group__persons__video=video, status='done',
            ).exclude(betas=[]).first()
        )()
        if shape:
            fixed_betas = shape.betas
            if shape.focal_scale and shape.focal_scale != 1.0:
                f_ps = max(W, H) * 1.2 * float(shape.focal_scale)
                intrinsics = {'fx': f_ps, 'fy': f_ps, 'cx': W / 2.0, 'cy': H / 2.0}
    except Exception:
        pass

    romp_model_path = getattr(_settings, 'ROMP_MODEL_PATH', None)
    romp_smpl_path  = getattr(_settings, 'ROMP_SMPL_PATH',  None)

    # Build frame list: [target - n_past*k_stride, ..., target - k_stride, target]
    frame_indices = [max(0, target_idx - (n_past - i) * k_stride) for i in range(n_past)] + [target_idx]
    # Deduplicate while preserving order (can happen when target_idx is near 0)
    seen = set()
    frame_indices = [fi for fi in frame_indices if not (fi in seen or seen.add(fi))]

    async def _get_params(fi):
        """Return pose_params for frame fi — from cache or freshly fitted."""
        cache_key = (str(pk), fi)
        if cache_key in _debug_pose_cache and _debug_pose_cache[cache_key] is not None:
            return _debug_pose_cache[cache_key]

        frm, _, _, _ = await asyncio.to_thread(_read_video_frame, video.path, str(fi))

        # ROMP als Orient-Init holen (auch wenn SMPL-X Optimizer läuft)
        romp_raw = await asyncio.to_thread(
            romp_infer_params, frm, romp_model_path, romp_smpl_path)
        romp_init_smplx = None
        if romp_raw is not None:
            romp_init_smplx = {'thetas': [romp_raw['theta']], 'beta': romp_raw['beta']}

        # SMPL-X Optimizer — mit ROMP als Init wenn verfügbar
        is_target = (fi == target_idx)
        n_orient = int(request.GET.get('n_orient_epochs', 600)) if is_target else 200
        n_pose   = int(request.GET.get('n_pose_epochs',   900)) if is_target else 300

        if not (RTMPoseBackend.available and ViTPoseBackend.available):
            return None
        try:
            data = await async_combined_analyze(frm)
        except Exception:
            return None
        body_landmarks = data.get('body_landmarks', [])
        if not body_landmarks:
            return None
        try:
            _, quality = await asyncio.to_thread(
                functools.partial(fit_and_render, frm, body_landmarks, intrinsics,
                                  fixed_betas=fixed_betas, romp_init=romp_init_smplx,
                                  n_orient_epochs=n_orient, n_pose_epochs=n_pose))
            params = quality.get('pose_params')
            _debug_pose_cache[cache_key] = params
            return params
        except Exception:
            log.exception("smooth: fit_and_render failed for frame %d", fi)
            return None

    # Collect params for all frames (sequentially to avoid GPU contention)
    all_params = []
    for fi in frame_indices:
        p = await _get_params(fi)
        all_params.append(p)

    # Filter out None entries (failed frames)
    valid = [(fi, p) for fi, p in zip(frame_indices, all_params) if p is not None]
    if not valid:
        return JsonResponse({'error': 'Kein einziger Frame konnte gefittet werden'}, status=422)

    valid_indices, valid_params = zip(*valid)
    source = valid_params[0]['source']

    # Stack parameters for SG smoothing
    if source == 'romp':
        theta_arr = np.array([p['theta'] for p in valid_params], dtype=np.float32)  # (N, 72)
        cam_arr   = np.array([p['cam']   for p in valid_params], dtype=np.float32)  # (N, 3)
    else:
        bp_arr     = np.array([p['body_pose']     for p in valid_params], dtype=np.float32)  # (N, 63)
        orient_arr = np.array([p['global_orient'] for p in valid_params], dtype=np.float32)  # (N, 3)
        transl_arr = np.array([p['transl']        for p in valid_params], dtype=np.float32)  # (N, 3)

    N = len(valid_params)
    w = sg_window if N >= sg_window else (N if N % 2 == 1 else max(N - 1, 1))
    if w < sg_poly + 1:
        w = sg_poly + 1 if (sg_poly + 1) % 2 == 1 else sg_poly + 2

    def _sg(arr):
        return savgol_filter(arr, window_length=w, polyorder=min(sg_poly, w - 1), axis=0)

    # Smoothed params for the target frame (last valid entry closest to target)
    target_smooth_idx = next(
        (i for i, fi in reversed(list(enumerate(valid_indices))) if fi == target_idx),
        len(valid_params) - 1,
    )

    if source == 'romp':
        smooth_theta = _sg(theta_arr)[target_smooth_idx]
        smooth_cam   = _sg(cam_arr)[target_smooth_idx]
        smoothed_params = {**valid_params[target_smooth_idx],
                           'theta': smooth_theta.tolist(),
                           'cam':   smooth_cam.tolist()}
        rendered_bgr = await asyncio.to_thread(render_romp_from_params, target_frame, smoothed_params)
    else:
        tp = valid_params[target_smooth_idx]
        smoothed_params = {
            'body_pose':     _sg(bp_arr)[target_smooth_idx].tolist(),
            'global_orient': _sg(orient_arr)[target_smooth_idx].tolist(),
            'transl':        _sg(transl_arr)[target_smooth_idx].tolist(),
            'betas':         tp['betas'],
            'kp_px':         tp['kp_px'],
            'fx': tp['fx'], 'fy': tp['fy'], 'cx': tp['cx'], 'cy': tp['cy'],
        }
        rendered_bgr = await asyncio.to_thread(
            render_smplx_from_params, target_frame, **smoothed_params)

    if rendered_bgr is None:
        return JsonResponse({'error': 'Rendering mit geglätteten Parametern fehlgeschlagen'}, status=500)

    return JsonResponse({
        'frame_idx':    target_idx,
        'render':       _frame_to_b64(rendered_bgr),
        'n_frames_used': N,
        'sg_window':    w,
        'source':       source,
    })


async def video_debug_backends(request, pk):
    """List all backends and their availability (proxied from pose-worker)."""
    from .detection.backends import ALL_BACKENDS, async_refresh_availability
    await async_refresh_availability()
    return JsonResponse({'backends': [
        {'id': b.backend_id, 'name': b.display_name, 'available': b.available}
        for b in ALL_BACKENDS
    ]})


async def video_debug_frame_backend(request, pk):
    """Run a single backend on a specific frame for the comparison view."""
    import asyncio
    from django.core.cache import cache
    from .detection.backends import BACKEND_BY_ID

    backend_id = request.GET.get('backend', 'mediapipe')
    backend = BACKEND_BY_ID.get(backend_id)
    if backend is None:
        return JsonResponse({'error': f'Unbekanntes Backend: {backend_id}'}, status=400)

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    frame_param = request.GET.get('frame')
    if frame_param is None:
        return JsonResponse({'error': 'frame parameter fehlt'}, status=400)

    frame, frame_idx, _, _ = await asyncio.to_thread(
        _read_video_frame, video.path, frame_param)

    cache_key = f'pose_debug:{backend_id}:{pk}:{frame_idx}'
    cached = await cache.aget(cache_key)
    if cached:
        return JsonResponse(cached)

    result = {
        'frame_idx':      frame_idx,
        'backend':        backend_id,
        'backend_name':   backend.display_name,
        'available':      backend.available,
        'detection':      False,
        'landmark_count': 0,
        'landmark_data':  [],
        'landmarks_image': None,
        'error':          None,
    }

    if not backend.available:
        result['error'] = 'Backend nicht verfügbar (pose-worker nicht erreichbar oder Paket fehlt)'
        return JsonResponse(result)

    try:
        data = await backend.async_analyze(frame)
        result['detection']       = data['detection']
        result['landmark_count']  = data['landmark_count']
        result['landmark_data']   = data['landmarks']
        result['landmarks_image'] = _frame_to_b64(data['render'])
        await cache.aset(cache_key, result)
    except Exception as exc:
        log.exception("video_debug_frame_backend failed: video=%s frame=%s backend=%s",
                      pk, frame_param, backend_id)
        result['error'] = str(exc)

    return JsonResponse(result)


async def video_debug_phase_b(request, pk):
    """
    Render already-fitted Phase B SMPL-X parameters for a specific person frame
    from the database — no re-fitting.

    GET params:
        person  – DetectedPerson UUID (required)
        frame   – frame index (required)
        smooth  – 1 to use Savitzky-Golay smoothed variants (default 0)
    """
    import asyncio
    from asgiref.sync import sync_to_async
    from .models import PersonFramePose, DetectedPerson as _DetectedPerson, PersonShape
    from .fitting.single_frame_fit import render_smplx_from_params

    person_id  = request.GET.get('person')
    frame_param = request.GET.get('frame')
    use_smooth  = request.GET.get('smooth', '0') == '1'

    if not person_id or frame_param is None:
        return JsonResponse({'error': 'person und frame Parameter erforderlich'}, status=400)

    try:
        frame_idx = int(frame_param)
    except ValueError:
        return JsonResponse({'error': 'frame muss eine Ganzzahl sein'}, status=400)

    video = await aget_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        return JsonResponse({'error': 'Videodatei nicht gefunden'}, status=404)

    pose_row = await sync_to_async(
        lambda: PersonFramePose.objects.filter(
            person_id=person_id, frame_idx=frame_idx
        ).select_related('person').first()
    )()
    if pose_row is None:
        return JsonResponse(
            {'error': f'Keine Phase-B Daten für Person {person_id} Frame {frame_idx}'},
            status=404,
        )

    # Load betas from PersonShape (fall back to zeros if unavailable)
    betas = await sync_to_async(
        lambda: next(
            iter(PersonShape.objects.filter(
                group__persons__id=person_id
            ).values_list('betas', flat=True)[:1]),
            [0.0] * 10,
        )
    )()

    # Pick raw or smoothed variants
    if use_smooth and pose_row.body_pose_smooth:
        body_pose    = pose_row.body_pose_smooth
        global_orient = pose_row.global_orient_smooth
        transl        = pose_row.transl_smooth
    else:
        body_pose    = pose_row.body_pose
        global_orient = pose_row.global_orient
        transl        = pose_row.transl

    # Read the video frame
    frame_bgr, _, _, _ = await asyncio.to_thread(
        _read_video_frame, video.path, str(frame_idx))
    H, W = frame_bgr.shape[:2]

    # Build minimal kp_px from the frame's PersonFrameKeypoints if available
    from .models import PersonFrameKeypoints
    kp_row = await sync_to_async(
        lambda: PersonFrameKeypoints.objects.filter(
            person_id=person_id, frame_idx=frame_idx
        ).first()
    )()
    kp_px = []
    if kp_row and kp_row.body_landmarks:
        lm_map = {d['idx']: d for d in kp_row.body_landmarks}
        from .fitting.single_frame_fit import _COCO_IDX
        for coco_idx in _COCO_IDX:
            d = lm_map.get(coco_idx)
            if d:
                kp_px.append([d['x'] * W, d['y'] * H, d['visibility']])
            else:
                kp_px.append([0.0, 0.0, 0.0])

    f = max(W, H) * 1.2
    # Weak-perspective cam params stored in Phase B
    cam_scale = pose_row.cam_scale
    if cam_scale and cam_scale > 0:
        fx = fy = float(cam_scale) * max(W, H)
        cx = W / 2.0 + (pose_row.cam_tx or 0.0) * fx
        cy = H / 2.0 + (pose_row.cam_ty or 0.0) * fy
    else:
        fx = fy = f
        cx = W / 2.0
        cy = H / 2.0

    rendered = await asyncio.to_thread(
        render_smplx_from_params,
        frame_bgr, body_pose, global_orient, transl, betas, kp_px,
        fx, fy, cx, cy,
    )

    return JsonResponse({
        'frame_idx':   frame_idx,
        'person_id':   person_id,
        'smooth':      use_smooth,
        'render':      _frame_to_b64(rendered),
        'has_face':    bool(pose_row.expression),
        'has_hands':   bool(pose_row.left_hand_pose or pose_row.right_hand_pose),
    })


# ─── Persons ─────────────────────────────────────────────────────────────────

def person_list(request):
    groups   = (PersonGroup.objects
                .prefetch_related('persons__video')
                .annotate(person_count=Count('persons'))
                .order_by('-updated_at'))
    ungrouped = DetectedPerson.objects.filter(groups=None).select_related('video')
    return render(request, 'core/person_list.html', {
        'groups': groups, 'ungrouped': ungrouped,
    })


@require_POST
def merge_persons(request):
    ids     = request.POST.getlist('person_ids')
    label   = request.POST.get('label', '')
    group_id = request.POST.get('existing_group_id', '')
    persons = DetectedPerson.objects.filter(pk__in=ids)
    if not persons.exists():
        return JsonResponse({'error': 'No persons found'}, status=400)

    if group_id:
        group = get_object_or_404(PersonGroup, pk=group_id)
        group.persons.add(*persons)
    else:
        group = PersonGroup.objects.create(label=label)
        group.persons.set(persons)
    return JsonResponse({'group_id': str(group.id), 'label': str(group)})


def group_detail(request, pk):
    from .models import PersonShape
    group        = get_object_or_404(PersonGroup, pk=pk)
    persons      = group.persons.select_related('video').all()
    avatars      = Avatar.objects.filter(group=group).order_by('name', '-version')
    other_groups = PersonGroup.objects.exclude(pk=pk).order_by('-updated_at')
    shape = getattr(group, 'shape', None)
    return render(request, 'core/group_detail.html', {
        'group': group, 'persons': persons, 'avatars': avatars,
        'other_groups': other_groups, 'shape': shape,
    })


@require_POST
def group_delete(request, pk):
    group = get_object_or_404(PersonGroup, pk=pk)
    group.delete()
    return redirect('person_list')


@require_POST
def unmerge_person(request, pk, person_pk):
    group  = get_object_or_404(PersonGroup, pk=pk)
    person = get_object_or_404(DetectedPerson, pk=person_pk)
    group.persons.remove(person)
    if group.persons.count() == 0:
        group.delete()
        return redirect('person_list')
    return redirect('group_detail', pk=pk)


@require_POST
def group_shape_fit(request, pk):
    """Start or restart a shape fitting job for a PersonGroup."""
    from .models import PersonGroup
    from .shape_tasks import start_shape_fit_job
    group = get_object_or_404(PersonGroup, pk=pk)
    shape = start_shape_fit_job(group)
    return JsonResponse({'status': shape.status, 'shape_id': str(shape.id)})


def group_shape_progress(request, pk):
    """Return current shape fit status and the last log entries."""
    from .models import PersonGroup, PersonShape
    from .shape_tasks import _latest_preview
    group = get_object_or_404(PersonGroup, pk=pk)
    try:
        shape = group.shape
    except PersonShape.DoesNotExist:
        return JsonResponse({'status': 'none'})
    return JsonResponse({
        'status':      shape.status,
        'log':         shape.log[-30:],
        'fit_quality': shape.fit_quality,
        'render_b64':  shape.render_b64 if shape.status == 'done' else '',
        'error':       shape.error,
        'preview_jpg': _latest_preview.get(str(shape.pk), ''),
    })


@require_POST
def group_shape_cancel(request, pk):
    """Cancel a running shape fit by signalling the cancel event and updating DB."""
    from .models import PersonGroup, PersonShape
    from .shape_tasks import _cancel_flags
    group = get_object_or_404(PersonGroup, pk=pk)
    try:
        shape = group.shape
        if shape.status == 'running':
            # Signal the running thread to stop at the next progress callback
            _cancel_flags.get(str(shape.pk), None) and _cancel_flags[str(shape.pk)].set()
            shape.status = 'failed'
            shape.error  = 'Manuell abgebrochen'
            shape.save()
    except PersonShape.DoesNotExist:
        pass
    return JsonResponse({'status': 'cancelled'})


def settings_shape(request):
    """GET/POST: shape fit settings page."""
    from .models import ShapeFitSettings
    cfg = ShapeFitSettings.get()
    if request.method == 'POST':
        try:
            cfg.frames_per_clip = max(1, int(request.POST.get('frames_per_clip', cfg.frames_per_clip)))
            cfg.frame_stride    = max(1, int(request.POST.get('frame_stride',    cfg.frame_stride)))
            cfg.n_phase1_epochs = max(10, int(request.POST.get('n_phase1_epochs', cfg.n_phase1_epochs)))
            cfg.n_phase2_epochs = max(10, int(request.POST.get('n_phase2_epochs', cfg.n_phase2_epochs)))
            cfg.save()
        except (ValueError, TypeError):
            pass
        return redirect('settings_shape')
    return render(request, 'core/settings_shape.html', {'cfg': cfg})


# ─── Avatars ─────────────────────────────────────────────────────────────────

def avatar_list(request):
    # Group by name, show latest version per name
    avatars = (Avatar.objects
               .values('name')
               .annotate(latest_version=Max('version'), count=Count('id'))
               .order_by('name'))

    avatar_objects = []
    for a in avatars:
        obj = Avatar.objects.filter(name=a['name']).order_by('-version').first()
        avatar_objects.append(obj)

    return render(request, 'core/avatar_list.html', {'avatars': avatar_objects})


def avatar_create(request):
    if request.method == 'POST':
        name     = request.POST.get('name', '').strip()
        group_id = request.POST.get('group_id')
        if not name:
            return JsonResponse({'error': 'Name required'}, status=400)

        group = None
        if group_id:
            group = get_object_or_404(PersonGroup, pk=group_id)

        max_v = Avatar.objects.filter(name=name).aggregate(
            Max('version'))['version__max'] or 0
        avatar = Avatar.objects.create(
            name    = name,
            group   = group,
            version = max_v + 1,
            status  = Avatar.Status.PENDING,
        )
        return redirect('avatar_detail', pk=avatar.pk)

    groups = PersonGroup.objects.annotate(pc=Count('persons')).order_by('-updated_at')
    return render(request, 'core/avatar_create.html', {'groups': groups})


def avatar_detail(request, pk):
    avatar   = get_object_or_404(Avatar, pk=pk)
    versions = Avatar.objects.filter(name=avatar.name).order_by('-version')
    edits    = avatar.edits.all()
    jobs     = avatar.jobs.order_by('-started_at')[:5]
    latest_job = jobs[0] if jobs else None

    # Load fitting metadata if available
    fitting_meta = None
    if avatar.data_path:
        meta_path = os.path.join(avatar.data_path, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                fitting_meta = json.load(f)

    # Count available previews
    n_previews = 0
    if avatar.data_path:
        preview_dir = os.path.join(avatar.data_path, 'previews')
        if os.path.isdir(preview_dir):
            n_previews = len([f for f in os.listdir(preview_dir) if f.endswith('.jpg')])

    # PersonShape log (Stage 0)
    person_shape = None
    if avatar.group_id:
        from .models import PersonShape
        person_shape = PersonShape.objects.filter(group_id=avatar.group_id).first()

    return render(request, 'core/avatar_detail.html', {
        'avatar': avatar, 'versions': versions, 'edits': edits, 'jobs': jobs,
        'latest_job': latest_job,
        'fitting_meta': fitting_meta, 'n_previews': n_previews,
        'person_shape': person_shape,
    })


@require_POST
def avatar_rename(request, pk):
    avatar   = get_object_or_404(Avatar, pk=pk)
    new_name = request.POST.get('name', '').strip()
    if not new_name:
        return JsonResponse({'error': 'Name required'}, status=400)
    Avatar.objects.filter(name=avatar.name).update(name=new_name)
    return redirect('avatar_detail', pk=pk)


@require_POST
def avatar_fork(request, pk):
    avatar  = get_object_or_404(Avatar, pk=pk)
    new_av  = avatar.create_new_version()
    return redirect('avatar_detail', pk=new_av.pk)


@require_POST
def avatar_delete(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    name   = avatar.name
    avatar.delete()
    # If other versions exist, go to latest; else list
    remaining = Avatar.objects.filter(name=name).order_by('-version').first()
    if remaining:
        return redirect('avatar_detail', pk=remaining.pk)
    return redirect('avatar_list')


@require_POST
def avatar_fit(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    stages = request.POST.getlist('stages') or ['1', '1.5', '2', '2.5', '3', '4']
    config = {
        'stages':            stages,
        'gender':            request.POST.get('gender', 'neutral'),
        'static_threshold':  float(request.POST.get('static_threshold', 0.05)),
        'max_frames':        int(request.POST.get('max_frames', 150)),
        'n_warmup_epochs':   int(request.POST.get('n_warmup_epochs', 50)),
        'n_shape_epochs':    int(request.POST.get('n_shape_epochs', 200)),
        'n_poseref_epochs':  int(request.POST.get('n_poseref_epochs', 100)),
        'n_pose_epochs':     int(request.POST.get('n_pose_epochs', 300)),
        'n_cage':            int(request.POST.get('n_cage', 60)),
        'tex_res_body':      int(request.POST.get('tex_res_body', 1024)),
        'tex_res_face':      int(request.POST.get('tex_res_face', 2048)),
        'inpainting':        request.POST.get('inpainting', 'prior'),
    }
    job = start_fitting_job(avatar, config)
    return JsonResponse({'job_id': str(job.id)})


@require_POST
def avatar_fit_cancel(request, pk):
    from django.utils import timezone
    avatar = get_object_or_404(Avatar, pk=pk)
    if avatar.status == Avatar.Status.FITTING:
        avatar.status = Avatar.Status.FAILED
        avatar.save()
        FittingJob.objects.filter(avatar=avatar, status__in=['queued', 'running']).update(
            status='failed',
            error='Manuell abgebrochen',
            finished_at=timezone.now(),
        )
    return JsonResponse({'status': avatar.status})


def avatar_edit(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    base_beta = []
    if avatar.data_path:
        meta_path = os.path.join(avatar.data_path, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                base_beta = json.load(f).get('beta', [])
    return render(request, 'core/avatar_editor.html', {
        'avatar': avatar,
        'base_beta': base_beta,
    })


import logging
log = logging.getLogger(__name__)

# Module-level smplx model cache (loaded once per process, keyed by model_dir+num_betas)
_smplx_cache: dict = {}   # key: (model_dir, num_betas, gender)

def _get_smplx(model_dir: str, num_betas: int, gender: str = 'neutral'):
    key = (model_dir, num_betas, gender)
    if key not in _smplx_cache:
        import torch
        import smplx as smplx_lib
        m = smplx_lib.create(
            model_path     = model_dir,
            model_type     = 'smplx',
            gender         = gender,
            num_betas      = num_betas,
            use_pca        = False,
            flat_hand_mean = True,
            batch_size     = 1,
        )
        m.eval()
        _smplx_cache[key] = m
        log.info("smplx model loaded and cached (gender=%s, num_betas=%d)", gender, num_betas)
    return _smplx_cache[key]


# Slider name → beta component index (rough SMPL-X PCA mapping)
_SLIDER_BETA = {
    'height':   0,
    'weight':   1,
    'shoulder': 2,
    'waist':    3,
    'bust':     4,
    'hip':      5,
    'leg':      6,
}


def avatar_mesh(request, pk):
    """Serve pre-generated T-pose SMPL-X mesh (mesh.obj). Generated during fitting."""
    avatar = get_object_or_404(Avatar, pk=pk)

    if not avatar.data_path:
        raise Http404

    obj_path = os.path.join(avatar.data_path, 'mesh.obj')

    # If not yet generated, try to regenerate from metadata (needs smplx in this container)
    if not os.path.exists(obj_path):
        meta_path = os.path.join(avatar.data_path, 'metadata.json')
        if not os.path.exists(meta_path):
            raise Http404
        with open(meta_path) as f:
            meta = json.load(f)
        beta_list = meta.get('beta')
        if not beta_list:
            raise Http404
        try:
            from .fitting.stage1 import _save_mesh_obj
            import numpy as np
            _save_mesh_obj(np.array(beta_list), meta.get('gender', 'neutral'), avatar.data_path)
        except Exception:
            pass

    if not os.path.exists(obj_path):
        raise Http404

    return FileResponse(open(obj_path, 'rb'), content_type='text/plain; charset=utf-8')


def avatar_mesh_morph(request, pk):
    """Return updated vertex positions (flat float array) for slider delta values.

    Query params: height, weight, shoulder, waist, bust, hip, leg  (each -2..+2)
    Response: {"verts": [x0,y0,z0, x1,y1,z1, ...]}
    """
    avatar = get_object_or_404(Avatar, pk=pk)
    if not avatar.data_path:
        return JsonResponse({'error': 'no data_path'}, status=404)

    meta_path = os.path.join(avatar.data_path, 'metadata.json')
    if not os.path.exists(meta_path):
        return JsonResponse({'error': 'not fitted'}, status=404)

    with open(meta_path) as f:
        meta = json.load(f)

    base_beta = meta.get('beta')
    if not base_beta:
        return JsonResponse({'error': 'no beta'}, status=404)

    # Apply per-beta deltas: query params b0, b1, b2, … bn
    import numpy as np
    beta = list(base_beta)
    for i in range(len(beta)):
        delta = float(request.GET.get(f'b{i}', 0))
        beta[i] += delta

    try:
        import torch
        model_dir = getattr(settings, 'SMPLX_MODEL_DIR', 'models')
        gender    = meta.get('gender', 'neutral')
        smplx_model = _get_smplx(model_dir, len(base_beta), gender)

        with torch.no_grad():
            out = smplx_model(betas=torch.tensor([beta], dtype=torch.float32))

        verts = out.vertices[0].numpy()   # (V, 3)
        faces = smplx_model.faces          # (F, 3)

        lines = []
        for v in verts:
            lines.append(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}')
        for face in faces:
            lines.append(f'f {face[0]+1} {face[1]+1} {face[2]+1}')

        return HttpResponse('\n'.join(lines), content_type='text/plain; charset=utf-8')

    except Exception as exc:
        log.exception("avatar_mesh_morph failed for %s", pk)
        return JsonResponse({'error': str(exc)}, status=500)


@require_POST
def avatar_mesh_rebuild(request, pk):
    """Explicitly (re-)generate mesh.obj from fitted beta values."""
    avatar = get_object_or_404(Avatar, pk=pk)
    if not avatar.data_path:
        return JsonResponse({'ok': False, 'error': 'Kein data_path'}, status=400)

    meta_path = os.path.join(avatar.data_path, 'metadata.json')
    if not os.path.exists(meta_path):
        return JsonResponse({'ok': False, 'error': 'metadata.json fehlt – Avatar noch nicht gefittet'}, status=400)

    with open(meta_path) as f:
        meta = json.load(f)

    beta_list = meta.get('beta')
    if not beta_list:
        return JsonResponse({'ok': False, 'error': 'Keine Beta-Werte in metadata.json'}, status=400)

    try:
        import numpy as np
        from .fitting.stage1 import _save_mesh_obj
        _save_mesh_obj(np.array(beta_list), meta.get('gender', 'neutral'), avatar.data_path)
    except Exception as exc:
        import traceback
        return JsonResponse({'ok': False, 'error': traceback.format_exc()}, status=500)

    obj_path = os.path.join(avatar.data_path, 'mesh.obj')
    if not os.path.exists(obj_path):
        return JsonResponse({'ok': False, 'error': '_save_mesh_obj hat kein mesh.obj erzeugt'}, status=500)

    return JsonResponse({'ok': True})


@require_POST
def avatar_edit_save(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    delta  = json.loads(request.body)
    label  = delta.pop('label', '')
    AvatarEdit.objects.create(avatar=avatar, label=label, delta=delta)
    return JsonResponse({'status': 'saved'})


# ─── API ─────────────────────────────────────────────────────────────────────

import zipfile, io, mimetypes
from django.http import HttpResponse, FileResponse, Http404

@require_POST
def video_delete(request, pk):
    video = get_object_or_404(VideoSource, pk=pk)
    video.delete()
    return redirect('video_list')


def video_stream(request, pk):
    video = get_object_or_404(VideoSource, pk=pk)
    if not os.path.exists(video.path):
        raise Http404

    content_type, _ = mimetypes.guess_type(video.path)
    content_type = content_type or 'video/mp4'
    file_size = os.path.getsize(video.path)

    range_header = request.META.get('HTTP_RANGE', '')
    if range_header:
        range_match = range_header.strip().replace('bytes=', '').split('-')
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1
        length = end - start + 1

        with open(video.path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        response = HttpResponse(data, status=206, content_type=content_type)
        response['Content-Range'] = f'bytes {start}-{end}/{file_size}'
        response['Accept-Ranges'] = 'bytes'
        response['Content-Length'] = length
        return response

    response = FileResponse(open(video.path, 'rb'), content_type=content_type)
    response['Accept-Ranges'] = 'bytes'
    response['Content-Length'] = file_size
    return response


def avatar_export(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    buf    = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if avatar.data_path and os.path.exists(avatar.data_path):
            for root, _, files in os.walk(avatar.data_path):
                for f in files:
                    full = os.path.join(root, f)
                    zf.write(full, os.path.relpath(full, avatar.data_path))
        else:
            zf.writestr('metadata.json', json.dumps({
                'name': avatar.name, 'version': avatar.version,
                'status': avatar.status, 'note': 'No data fitted yet'
            }, indent=2))
    buf.seek(0)
    fname = f"{avatar.name}_v{avatar.version}.zip".replace(' ', '_')
    resp  = HttpResponse(buf, content_type='application/zip')
    resp['Content-Disposition'] = f'attachment; filename="{fname}"'
    return resp


@require_POST
def group_rename(request, pk):
    group = get_object_or_404(PersonGroup, pk=pk)
    label = request.POST.get('label', '').strip()
    if label:
        group.label = label
        group.save()
    return redirect('group_detail', pk=pk)


@require_POST
def group_merge(request, pk):
    """Merge another group's persons into this group, then delete the other group."""
    group  = get_object_or_404(PersonGroup, pk=pk)
    other  = get_object_or_404(PersonGroup, pk=request.POST.get('other_group_id', ''))
    if other.pk != group.pk:
        group.persons.add(*other.persons.all())
        other.delete()
    return redirect('group_detail', pk=pk)


def job_status(request, pk):
    job = get_object_or_404(FittingJob, pk=pk)
    return JsonResponse({
        'status':        job.status,
        'stage':         job.current_stage,
        'progress':      job.progress,
        'log':           job.log[-20:],
        'error':         job.error,
    })


def avatar_log(request, pk):
    """Return combined fitting log (PersonShape + FittingJob) as JSON for polling."""
    from .models import PersonShape
    avatar = get_object_or_404(Avatar, pk=pk)
    lines  = []

    # Shape fit log (Stage 0)
    if avatar.group_id:
        shape = PersonShape.objects.filter(group_id=avatar.group_id).first()
        if shape:
            for e in shape.log:
                if e.get('type') == 'info':
                    lines.append({'source': 'shape', 'ts': e.get('ts', ''),
                                  'msg': e.get('msg', '')})
                elif e.get('type') == 'progress':
                    lines.append({'source': 'shape', 'ts': e.get('ts', ''),
                                  'msg': (f"[{e.get('phase','')}] "
                                          f"epoch {e.get('epoch_all','')}/{e.get('total_all','')} "
                                          f"loss={e.get('loss','')}")})
            if shape.error:
                lines.append({'source': 'shape', 'ts': '', 'msg': f'❌ {shape.error}'})

    # Latest fitting job log
    latest = avatar.jobs.order_by('-started_at').first()
    if latest:
        seen_stage = None
        for e in latest.log:
            stage = e.get('stage', '')
            if stage != seen_stage:
                lines.append({'source': 'job', 'ts': '',
                              'msg': f'── Stage {stage} ──'})
                seen_stage = stage
            if e.get('note'):
                # Human-readable milestone message — show prominently
                lines.append({'source': 'job', 'ts': '', 'msg': f'  ▶ {e["note"]}'})
            elif e.get('loss') is not None and e.get('epoch', 0) % 50 == 0:
                # Show every 50th epoch to avoid flooding the log
                lines.append({'source': 'job', 'ts': '',
                              'msg': f"  ep {e.get('epoch',''):>4}  loss={e.get('loss', 0):.5f}"})
        if latest.error:
            lines.append({'source': 'job', 'ts': '', 'msg': f'❌ {latest.error}'})

    return JsonResponse({'lines': lines, 'avatar_status': avatar.status})


def avatar_preview_image(request, pk, n):
    """Serve previews/preview_NN.jpg from the avatar data folder."""
    avatar = get_object_or_404(Avatar, pk=pk)
    if not avatar.data_path:
        raise Http404
    img_path = os.path.join(avatar.data_path, 'previews', f'preview_{n:02d}.jpg')
    if not os.path.exists(img_path):
        raise Http404
    return FileResponse(open(img_path, 'rb'), content_type='image/jpeg')


def avatar_versions(request, pk):
    avatar   = get_object_or_404(Avatar, pk=pk)
    versions = Avatar.objects.filter(name=avatar.name).order_by('-version')
    data = [{
        'id':      str(v.id),
        'version': v.version,
        'status':  v.status,
        'notes':   v.notes,
        'created': v.created_at.isoformat(),
    } for v in versions]
    return JsonResponse({'versions': data})
