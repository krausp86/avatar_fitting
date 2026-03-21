import os
import json
from pathlib import Path
from django.shortcuts import render, get_object_or_404, redirect
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


def video_detail(request, pk):
    video   = get_object_or_404(VideoSource, pk=pk)
    persons = video.persons.all().order_by('-frame_count')
    return render(request, 'core/video_detail.html', {'video': video, 'persons': persons})


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
    group        = get_object_or_404(PersonGroup, pk=pk)
    persons      = group.persons.select_related('video').all()
    avatars      = Avatar.objects.filter(group=group).order_by('name', '-version')
    other_groups = PersonGroup.objects.exclude(pk=pk).order_by('-updated_at')
    return render(request, 'core/group_detail.html', {
        'group': group, 'persons': persons, 'avatars': avatars,
        'other_groups': other_groups,
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

    return render(request, 'core/avatar_detail.html', {
        'avatar': avatar, 'versions': versions, 'edits': edits, 'jobs': jobs,
        'fitting_meta': fitting_meta, 'n_previews': n_previews,
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
        'stages':           stages,
        'gender':           request.POST.get('gender', 'neutral'),
        'static_threshold': float(request.POST.get('static_threshold', 0.05)),
        'max_frames':       int(request.POST.get('max_frames', 150)),
        'n_cage':           int(request.POST.get('n_cage', 60)),
        'tex_res_body':     int(request.POST.get('tex_res_body', 1024)),
        'tex_res_face':     int(request.POST.get('tex_res_face', 2048)),
        'inpainting':       request.POST.get('inpainting', 'prior'),
    }
    job = start_fitting_job(avatar, config)
    return JsonResponse({'job_id': str(job.id)})


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
