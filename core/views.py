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
    """Re-run person detection for a single video (replaces existing tracks)."""
    from .scanner import detect_persons_for_video
    video   = get_object_or_404(VideoSource, pk=pk)
    created = detect_persons_for_video(video)
    return JsonResponse({'created': created, 'video_id': str(video.id)})


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
    ids    = request.POST.getlist('person_ids')
    label  = request.POST.get('label', '')
    persons = DetectedPerson.objects.filter(pk__in=ids)
    if not persons.exists():
        return JsonResponse({'error': 'No persons found'}, status=400)

    group = PersonGroup.objects.create(label=label)
    group.persons.set(persons)
    return JsonResponse({'group_id': str(group.id), 'label': str(group)})


def group_detail(request, pk):
    group   = get_object_or_404(PersonGroup, pk=pk)
    persons = group.persons.select_related('video').all()
    avatars = Avatar.objects.filter(group=group).order_by('name', '-version')
    return render(request, 'core/group_detail.html', {
        'group': group, 'persons': persons, 'avatars': avatars,
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

        avatar = Avatar.objects.create(
            name    = name,
            group   = group,
            version = 1,
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
    return render(request, 'core/avatar_detail.html', {
        'avatar': avatar, 'versions': versions, 'edits': edits, 'jobs': jobs,
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
        'static_threshold': float(request.POST.get('static_threshold', 0.05)),
        'n_cage':           int(request.POST.get('n_cage', 60)),
        'tex_res_body':     int(request.POST.get('tex_res_body', 1024)),
        'tex_res_face':     int(request.POST.get('tex_res_face', 2048)),
        'inpainting':       request.POST.get('inpainting', 'prior'),
    }
    job = start_fitting_job(avatar, config)
    return JsonResponse({'job_id': str(job.id)})


def avatar_edit(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    return render(request, 'core/avatar_editor.html', {'avatar': avatar})


@require_POST
def avatar_edit_save(request, pk):
    avatar = get_object_or_404(Avatar, pk=pk)
    delta  = json.loads(request.body)
    label  = delta.pop('label', '')
    AvatarEdit.objects.create(avatar=avatar, label=label, delta=delta)
    return JsonResponse({'status': 'saved'})


# ─── API ─────────────────────────────────────────────────────────────────────

import zipfile, io
from django.http import HttpResponse

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


def job_status(request, pk):
    job = get_object_or_404(FittingJob, pk=pk)
    return JsonResponse({
        'status':        job.status,
        'stage':         job.current_stage,
        'progress':      job.progress,
        'log':           job.log[-20:],
        'error':         job.error,
    })


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
