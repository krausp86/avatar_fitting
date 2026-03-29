import uuid
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_remove_detectedperson_unique_video_track_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='PersonFramePose',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('frame_idx', models.IntegerField()),
                ('body_pose', models.JSONField(default=list)),
                ('global_orient', models.JSONField(default=list)),
                ('transl', models.JSONField(default=list)),
                ('body_pose_smooth', models.JSONField(default=list)),
                ('global_orient_smooth', models.JSONField(default=list)),
                ('transl_smooth', models.JSONField(default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('person', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='frame_poses',
                    to='core.detectedperson',
                )),
            ],
            options={
                'ordering': ['person', 'frame_idx'],
                'unique_together': {('person', 'frame_idx')},
            },
        ),
    ]
