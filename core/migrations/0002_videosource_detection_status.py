from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='videosource',
            name='detection_status',
            field=models.CharField(
                choices=[('pending','Pending'),('detecting','Detecting'),('done','Done'),('failed','Failed')],
                default='pending',
                max_length=16,
            ),
        ),
    ]
