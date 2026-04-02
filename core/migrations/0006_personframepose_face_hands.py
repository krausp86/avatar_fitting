from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_personframepose'),
    ]

    operations = [
        # Face + Expression
        migrations.AddField(
            model_name='personframepose',
            name='expression',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='jaw_pose',
            field=models.JSONField(default=list),
        ),
        # Hand pose (MANO PCA, 12 components each)
        migrations.AddField(
            model_name='personframepose',
            name='left_hand_pose',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='right_hand_pose',
            field=models.JSONField(default=list),
        ),
        # Per-frame weak-perspective camera
        migrations.AddField(
            model_name='personframepose',
            name='cam_scale',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='cam_tx',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='cam_ty',
            field=models.FloatField(blank=True, null=True),
        ),
        # Smoothed variants
        migrations.AddField(
            model_name='personframepose',
            name='expression_smooth',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='jaw_pose_smooth',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='left_hand_pose_smooth',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='personframepose',
            name='right_hand_pose_smooth',
            field=models.JSONField(default=list),
        ),
    ]
