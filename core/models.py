from django.db import models
import uuid
import json


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


class Avatar(models.Model):
    """A named, versioned avatar fitted from a PersonGroup."""

    class Status(models.TextChoices):
        PENDING   = 'pending',   'Pending'
        FITTING   = 'fitting',   'Fitting'
        DONE      = 'done',      'Done'
        FAILED    = 'failed',    'Failed'

    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name         = models.CharField(max_length=256)
    group        = models.ForeignKey(PersonGroup, on_delete=models.SET_NULL,
                                     null=True, blank=True, related_name='avatars')
    version      = models.PositiveIntegerField(default=1)
    parent       = models.ForeignKey('self', on_delete=models.SET_NULL,
                                     null=True, blank=True, related_name='children')
    status       = models.CharField(max_length=16, choices=Status.choices,
                                     default=Status.PENDING)
    data_path    = models.TextField(blank=True)   # path to person_{id}/ folder
    fitting_log  = models.JSONField(default=list)  # [{stage, epoch, loss, ...}]
    notes        = models.TextField(blank=True)
    hair_style   = models.CharField(max_length=64, blank=True)   # long_straight, short_curly, etc.
    hair_meta    = models.JSONField(default=dict)                 # color_rgb, length_cm, has_reconstruction
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)
    thumbnail    = models.ImageField(upload_to='thumbnails/avatars/', null=True, blank=True)

    class Meta:
        ordering = ['name', '-version']
        unique_together = [('name', 'version')]

    def __str__(self):
        return f"{self.name} v{self.version}"

    def create_new_version(self):
        """Fork this avatar into a new version."""
        import shutil, os
        from django.conf import settings

        max_v = Avatar.objects.filter(name=self.name).aggregate(
            models.Max('version'))['version__max'] or 0

        new_avatar = Avatar.objects.create(
            name       = self.name,
            group      = self.group,
            version    = max_v + 1,
            parent     = self,
            status     = Avatar.Status.DONE,
            notes      = f"Forked from v{self.version}",
        )

        # Copy data folder
        if self.data_path and os.path.exists(self.data_path):
            new_path = os.path.join(settings.AVATAR_DATA_ROOT,
                                    f"{self.name}_v{new_avatar.version}_{new_avatar.id}")
            shutil.copytree(self.data_path, new_path)
            new_avatar.data_path = new_path
            new_avatar.save()

        return new_avatar


class AvatarEdit(models.Model):
    """A delta edit applied on top of a fitted avatar (non-destructive)."""
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    avatar       = models.ForeignKey(Avatar, on_delete=models.CASCADE, related_name='edits')
    label        = models.CharField(max_length=256, blank=True)
    delta        = models.JSONField(default=dict)   # {β_delta, physics_delta, texture_delta}
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Edit on {self.avatar} – {self.label or self.created_at}"


class FittingJob(models.Model):
    """A running or completed fitting job."""

    class Status(models.TextChoices):
        QUEUED   = 'queued',   'Queued'
        RUNNING  = 'running',  'Running'
        DONE     = 'done',     'Done'
        FAILED   = 'failed',   'Failed'

    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    avatar       = models.ForeignKey(Avatar, on_delete=models.CASCADE, related_name='jobs')
    status       = models.CharField(max_length=16, choices=Status.choices,
                                     default=Status.QUEUED)
    current_stage = models.CharField(max_length=16, blank=True)
    progress     = models.FloatField(default=0.0)   # 0-1
    log          = models.JSONField(default=list)
    error        = models.TextField(blank=True)
    started_at   = models.DateTimeField(null=True, blank=True)
    finished_at  = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Job {self.id} – {self.avatar} [{self.status}]"
