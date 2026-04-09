import os
import threading
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse, Http404
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings
from django.db.models import Count

from .models import VideoSource, DetectedPerson, PersonGroup
from .scanner import scan_video_folder


# ─── Dashboard ───────────────────────────────────────────────────────────────

def dashboard(request):
    ctx = {
        'video_count':  VideoSource.objects.count(),
        'person_count': DetectedPerson.objects.count(),
        'group_count':  PersonGroup.objects.count(),
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
def refresh_video_metadata(request):
    from .scanner import _populate_metadata
    qs = VideoSource.objects.filter(fps__isnull=True) | VideoSource.objects.filter(resolution='')
    updated = 0
    for vs in qs.distinct():
        _populate_metadata(vs)
        updated += 1
    return JsonResponse({'updated': updated})


@require_POST
def detect_persons(request, pk):
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
        except Exception:
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
        'status':       video.detection_status,
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


def video_stream(request, pk):
    """Stream a video file with byte-range support."""
    video = get_object_or_404(VideoSource, pk=pk)
    path  = video.path
    if not os.path.exists(path):
        raise Http404

    file_size = os.path.getsize(path)
    range_header = request.META.get('HTTP_RANGE', '')
    if range_header.startswith('bytes='):
        start, _, end = range_header[6:].partition('-')
        start = int(start) if start else 0
        end   = int(end)   if end   else file_size - 1
        end   = min(end, file_size - 1)
        length = end - start + 1

        def _iter():
            with open(path, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        resp = StreamingHttpResponse(_iter(), status=206, content_type='video/mp4')
        resp['Content-Range']  = f'bytes {start}-{end}/{file_size}'
        resp['Content-Length'] = str(length)
        resp['Accept-Ranges']  = 'bytes'
        return resp

    def _iter_full():
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    resp = StreamingHttpResponse(_iter_full(), content_type='video/mp4')
    resp['Content-Length'] = str(file_size)
    resp['Accept-Ranges']  = 'bytes'
    return resp


@require_POST
def video_delete(request, pk):
    video = get_object_or_404(VideoSource, pk=pk)
    video.delete()
    return redirect('video_list')


# ─── Person groups ────────────────────────────────────────────────────────────

def person_list(request):
    groups    = (PersonGroup.objects
                 .prefetch_related('persons__video')
                 .annotate(person_count=Count('persons'))
                 .order_by('-updated_at'))
    ungrouped = DetectedPerson.objects.filter(groups=None).select_related('video')
    return render(request, 'core/person_list.html', {
        'groups': groups, 'ungrouped': ungrouped,
    })


@require_POST
def merge_persons(request):
    ids      = request.POST.getlist('person_ids')
    label    = request.POST.get('label', '')
    group_id = request.POST.get('existing_group_id', '')
    persons  = DetectedPerson.objects.filter(pk__in=ids)
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
    other_groups = PersonGroup.objects.exclude(pk=pk).order_by('-updated_at')
    return render(request, 'core/group_detail.html', {
        'group': group, 'persons': persons, 'other_groups': other_groups,
    })


@require_POST
def group_delete(request, pk):
    group = get_object_or_404(PersonGroup, pk=pk)
    group.delete()
    return redirect('person_list')


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
    """Merge another group's persons into this group, then delete the other."""
    group = get_object_or_404(PersonGroup, pk=pk)
    other = get_object_or_404(PersonGroup, pk=request.POST.get('other_group_id', ''))
    if other.pk != group.pk:
        group.persons.add(*other.persons.all())
        other.delete()
    return redirect('group_detail', pk=pk)


@require_POST
def unmerge_person(request, pk, person_pk):
    group  = get_object_or_404(PersonGroup, pk=pk)
    person = get_object_or_404(DetectedPerson, pk=person_pk)
    group.persons.remove(person)
    if group.persons.count() == 0:
        group.delete()
        return redirect('person_list')
    return redirect('group_detail', pk=pk)
