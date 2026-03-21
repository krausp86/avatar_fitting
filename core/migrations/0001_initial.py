from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='VideoSource',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('path', models.TextField(unique=True)),
                ('filename', models.CharField(max_length=512)),
                ('folder', models.CharField(max_length=512)),
                ('duration_s', models.FloatField(blank=True, null=True)),
                ('fps', models.FloatField(blank=True, null=True)),
                ('resolution', models.CharField(blank=True, max_length=32)),
                ('scanned_at', models.DateTimeField(auto_now_add=True)),
                ('thumbnail', models.ImageField(blank=True, null=True, upload_to='thumbnails/videos/')),
            ],
            options={'ordering': ['folder', 'filename']},
        ),
        migrations.CreateModel(
            name='DetectedPerson',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='persons', to='core.videosource')),
                ('track_id', models.CharField(max_length=64)),
                ('frame_start', models.IntegerField()),
                ('frame_end', models.IntegerField()),
                ('frame_count', models.IntegerField()),
                ('visibility', models.FloatField(default=0.0)),
                ('thumbnail', models.ImageField(blank=True, null=True, upload_to='thumbnails/persons/')),
                ('meta', models.JSONField(default=dict)),
            ],
            options={'ordering': ['-frame_count']},
        ),
        migrations.AddConstraint(
            model_name='detectedperson',
            constraint=models.UniqueConstraint(fields=('video', 'track_id'), name='unique_video_track'),
        ),
        migrations.AlterUniqueTogether(
            name='detectedperson',
            unique_together={('video', 'track_id')},
        ),
        migrations.CreateModel(
            name='PersonGroup',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('label', models.CharField(blank=True, max_length=256)),
                ('persons', models.ManyToManyField(related_name='groups', to='core.detectedperson')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('thumbnail', models.ImageField(blank=True, null=True, upload_to='thumbnails/groups/')),
            ],
        ),
        migrations.CreateModel(
            name='Avatar',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=256)),
                ('group', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='avatars', to='core.persongroup')),
                ('version', models.PositiveIntegerField(default=1)),
                ('parent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='children', to='core.avatar')),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('fitting', 'Fitting'), ('done', 'Done'), ('failed', 'Failed')], default='pending', max_length=16)),
                ('data_path', models.TextField(blank=True)),
                ('fitting_log', models.JSONField(default=list)),
                ('notes', models.TextField(blank=True)),
                ('hair_style', models.CharField(blank=True, max_length=64)),
                ('hair_meta', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('thumbnail', models.ImageField(blank=True, null=True, upload_to='thumbnails/avatars/')),
            ],
            options={'ordering': ['name', '-version']},
        ),
        migrations.AlterUniqueTogether(
            name='avatar',
            unique_together={('name', 'version')},
        ),
        migrations.CreateModel(
            name='AvatarEdit',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('avatar', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='edits', to='core.avatar')),
                ('label', models.CharField(blank=True, max_length=256)),
                ('delta', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='FittingJob',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('avatar', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='jobs', to='core.avatar')),
                ('status', models.CharField(choices=[('queued', 'Queued'), ('running', 'Running'), ('done', 'Done'), ('failed', 'Failed')], default='queued', max_length=16)),
                ('current_stage', models.CharField(blank=True, max_length=16)),
                ('progress', models.FloatField(default=0.0)),
                ('log', models.JSONField(default=list)),
                ('error', models.TextField(blank=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
            ],
        ),
    ]
