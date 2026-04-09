from django.db import models
import uuid


class VideoSource(models.Model):
    """A scanned video file from the filesystem."""
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    path         = models.TextField(unique=True)
    filename     = models.CharField(max_length=512)
    folder       = models.CharField(max_length=512)
    duration_s   = models.FloatField(null=True, blank=True)
    fps          = models.FloatField(null=True, blank=True)
    resolution   = models.CharField(max_length=32, blank=True)
    scanned_at   = models.DateTimeField(auto_now_add=True)
    thumbnail    = models.ImageField(upload_to='thumbnails/videos/', null=True, blank=True)
    detection_status = models.CharField(
        max_length=16,
        choices=[('pending','Pending'),('detecting','Detecting'),('done','Done'),('failed','Failed')],
        default='pending',
    )

    class Meta:
        ordering = ['folder', 'filename']

    def __str__(self):
        return f"{self.folder}/{self.filename}"


class DetectedPerson(models.Model):
    """A person track detected in a video."""
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video        = models.ForeignKey(VideoSource, on_delete=models.CASCADE, related_name='persons')
    track_id     = models.CharField(max_length=64)
    frame_start  = models.IntegerField()
    frame_end    = models.IntegerField()
    frame_count  = models.IntegerField()
    visibility   = models.FloatField(default=0.0)   # 0-1 average visibility score
    thumbnail    = models.ImageField(upload_to='thumbnails/persons/', null=True, blank=True)
    meta         = models.JSONField(default=dict)

    class Meta:
        ordering = ['-frame_count']
        unique_together = [('video', 'track_id')]

    def __str__(self):
        return f"{self.video.filename} / track {self.track_id}"


class PersonGroup(models.Model):
    """One or more DetectedPerson tracks merged into a single identity."""
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    label        = models.CharField(max_length=256, blank=True)
    persons      = models.ManyToManyField(DetectedPerson, related_name='groups')
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)
    thumbnail    = models.ImageField(upload_to='thumbnails/groups/', null=True, blank=True)

    def __str__(self):
        return self.label or str(self.id)

    @property
    def total_frames(self):
        return sum(p.frame_count for p in self.persons.all())


class PersonFrameKeypoints(models.Model):
    """Cached keypoints for a single frame of a DetectedPerson track."""
    id             = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    person         = models.ForeignKey(DetectedPerson, on_delete=models.CASCADE, related_name='keypoints')
    frame_idx      = models.IntegerField()
    body_landmarks = models.JSONField(default=list)   # ViTPose COCO-17
    rtm_landmarks  = models.JSONField(default=list)   # RTMPose wholebody
    seg_mask_b64   = models.TextField(blank=True)     # PNG base64 segmentation mask
    computed_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('person', 'frame_idx')]
        ordering = ['person', 'frame_idx']

    def __str__(self):
        return f"{self.person} frame {self.frame_idx}"
