from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Videos
    path('videos/', views.video_list, name='video_list'),
    path('videos/scan/', views.scan_videos, name='scan_videos'),
    path('videos/refresh-metadata/', views.refresh_video_metadata, name='refresh_video_metadata'),
    path('videos/<uuid:pk>/', views.video_detail, name='video_detail'),
    path('videos/<uuid:pk>/detect/', views.detect_persons, name='detect_persons'),
    path('videos/<uuid:pk>/detect/status/', views.detect_persons_status, name='detect_persons_status'),
    path('videos/<uuid:pk>/detect/cancel/', views.detect_persons_cancel, name='detect_persons_cancel'),
    path('videos/<uuid:pk>/stream/', views.video_stream, name='video_stream'),
    path('videos/<uuid:pk>/delete/', views.video_delete, name='video_delete'),
    path('persons/<uuid:pk>/delete/', views.person_delete, name='person_delete'),

    # Person groups
    path('persons/', views.person_list, name='person_list'),
    path('persons/merge/', views.merge_persons, name='merge_persons'),
    path('persons/group/<uuid:pk>/', views.group_detail, name='group_detail'),
    path('persons/group/<uuid:pk>/delete/', views.group_delete, name='group_delete'),
    path('persons/group/<uuid:pk>/rename/', views.group_rename, name='group_rename'),
    path('persons/group/<uuid:pk>/merge/', views.group_merge, name='group_merge'),
    path('persons/group/<uuid:pk>/unmerge/<uuid:person_pk>/', views.unmerge_person, name='unmerge_person'),
]
