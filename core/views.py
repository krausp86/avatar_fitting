import os
import threading
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse, Http404
from django.views.decorators.http import require_POST
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


@require_POST
def person_delete(request, pk):
    person = get_object_or_404(DetectedPerson, pk=pk)
    video_pk = person.video_id
    person.delete()   # cascade removes group memberships automatically
    return JsonResponse({'status': 'deleted', 'video_id': str(video_pk)})


@require_POST
def person_set_frames(request, pk):
    """Manually correct frame_start / frame_end of a DetectedPerson track."""
    person = get_object_or_404(DetectedPerson, pk=pk)
    try:
        frame_start = int(request.POST['frame_start'])
        frame_end   = int(request.POST['frame_end'])
    except (KeyError, ValueError):
        return JsonResponse({'error': 'frame_start and frame_end required'}, status=400)
    if frame_start < 0 or frame_end < frame_start:
        return JsonResponse({'error': 'Invalid frame range'}, status=400)
    person.frame_start  = frame_start
    person.frame_end    = frame_end
    person.frame_count  = frame_end - frame_start + 1
    person.save(update_fields=['frame_start', 'frame_end', 'frame_count'])
    return JsonResponse({
        'status':      'ok',
        'frame_start': person.frame_start,
        'frame_end':   person.frame_end,
        'frame_count': person.frame_count,
    })


@require_POST
def person_create(request, pk):
    """Manually create a new DetectedPerson track for a video."""
    import uuid as _uuid
    video = get_object_or_404(VideoSource, pk=pk)
    try:
        frame_start = int(request.POST['frame_start'])
        frame_end   = int(request.POST['frame_end'])
    except (KeyError, ValueError):
        return JsonResponse({'error': 'frame_start and frame_end required'}, status=400)
    if frame_start < 0 or frame_end < frame_start:
        return JsonResponse({'error': 'Invalid frame range'}, status=400)

    # generate a unique manual track_id that won't collide with auto-detected ones
    track_id = 'm_' + _uuid.uuid4().hex[:8]
    person = DetectedPerson.objects.create(
        video       = video,
        track_id    = track_id,
        frame_start = frame_start,
        frame_end   = frame_end,
        frame_count = frame_end - frame_start + 1,
        visibility  = 1.0,
        meta        = {'manual': True},
    )

    # grab thumbnail from best-effort frame extraction
    try:
        import cv2, io
        from django.core.files.base import ContentFile
        from PIL import Image as PILImage
        frame_idx = (frame_start + frame_end) // 2
        cap = cv2.VideoCapture(video.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            h, w = frame.shape[:2]
            scale = min(256 / w, 256 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            PILImage.fromarray(rgb).save(buf, format='JPEG', quality=85)
            buf.seek(0)
            person.thumbnail.save(f'person_{video.id}_{track_id}.jpg', ContentFile(buf.read()), save=True)
    except Exception:
        pass

    return JsonResponse({
        'status':      'created',
        'person_id':   str(person.pk),
        'track_id':    person.track_id,
        'frame_start': person.frame_start,
        'frame_end':   person.frame_end,
        'frame_count': person.frame_count,
        'thumbnail':   person.thumbnail.url if person.thumbnail else None,
        'set_frames_url': request.build_absolute_uri(f'/persons/{person.pk}/set-frames/'),
        'delete_url':     request.build_absolute_uri(f'/persons/{person.pk}/delete/'),
    })


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
