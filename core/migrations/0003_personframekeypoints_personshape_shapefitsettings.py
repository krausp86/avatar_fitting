from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_videosource_detection_status'),
    ]

    operations = [
        migrations.CreateModel(
            name='ShapeFitSettings',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False)),
                ('frames_per_clip', models.IntegerField(default=10, help_text='Frames to sample per clip')),
                ('frame_stride', models.IntegerField(default=5, help_text='Stride between sampled frames within a clip')),
                ('n_phase1_epochs', models.IntegerField(default=300, help_text='Phase 1 epochs (shape + orient + transl)')),
                ('n_phase2_epochs', models.IntegerField(default=500, help_text='Phase 2 epochs (full pose)')),
            ],
            options={'verbose_name': 'Shape Fit Settings'},
        ),
        migrations.CreateModel(
            name='PersonFrameKeypoints',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('frame_idx', models.IntegerField()),
                ('body_landmarks', models.JSONField(default=list)),
                ('rtm_landmarks', models.JSONField(default=list)),
                ('seg_mask_b64', models.TextField(blank=True)),
                ('computed_at', models.DateTimeField(auto_now_add=True)),
                ('person', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='keypoints',
                    to='core.detectedperson',
                )),
            ],
            options={'ordering': ['person', 'frame_idx']},
        ),
        migrations.AddConstraint(
            model_name='personframekeypoints',
            constraint=models.UniqueConstraint(fields=['person', 'frame_idx'], name='unique_person_frame'),
        ),
        migrations.CreateModel(
            name='PersonShape',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('betas', models.JSONField(default=list)),
                ('hip_correction', models.FloatField(default=1.0)),
                ('focal_scale', models.FloatField(default=1.0)),
                ('status', models.CharField(
                    choices=[('pending','Pending'),('running','Running'),('done','Done'),('failed','Failed')],
                    default='pending', max_length=16,
                )),
                ('log', models.JSONField(default=list)),
                ('fit_quality', models.JSONField(default=dict)),
                ('render_b64', models.TextField(blank=True)),
                ('error', models.TextField(blank=True)),
                ('fitted_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('group', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='shape',
                    to='core.persongroup',
                )),
            ],
        ),
    ]
