from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Videos
    path('videos/', views.video_list, name='video_list'),
    path('videos/scan/', views.scan_videos, name='scan_videos'),
    path('videos/<uuid:pk>/', views.video_detail, name='video_detail'),
    path('videos/<uuid:pk>/detect/', views.detect_persons, name='detect_persons'),
    path('videos/<uuid:pk>/detect/status/', views.detect_persons_status, name='detect_persons_status'),
    path('videos/<uuid:pk>/detect/cancel/', views.detect_persons_cancel, name='detect_persons_cancel'),
    path('videos/<uuid:pk>/debug/frame/', views.video_debug_frame, name='video_debug_frame'),
    path('videos/<uuid:pk>/debug/frame/backend/', views.video_debug_frame_backend, name='video_debug_frame_backend'),
    path('videos/<uuid:pk>/debug/frame/combined/', views.video_debug_frame_combined, name='video_debug_frame_combined'),
    path('videos/<uuid:pk>/debug/frame/smplx/', views.video_debug_frame_smplx, name='video_debug_frame_smplx'),
    path('videos/<uuid:pk>/debug/frame/smplx/progress/', views.video_debug_smplx_progress, name='video_debug_smplx_progress'),
    path('videos/<uuid:pk>/debug/frame/smplx/smooth/', views.video_debug_frame_smplx_smooth, name='video_debug_frame_smplx_smooth'),
    path('videos/<uuid:pk>/debug/frame/phase-b/', views.video_debug_phase_b, name='video_debug_phase_b'),
    path('videos/<uuid:pk>/debug/backends/', views.video_debug_backends, name='video_debug_backends'),

    # Person groups
    path('persons/', views.person_list, name='person_list'),
    path('persons/merge/', views.merge_persons, name='merge_persons'),
    path('persons/group/<uuid:pk>/', views.group_detail, name='group_detail'),
    path('persons/group/<uuid:pk>/delete/', views.group_delete, name='group_delete'),
    path('persons/group/<uuid:pk>/unmerge/<uuid:person_pk>/', views.unmerge_person, name='unmerge_person'),

    # Avatars
    path('avatars/', views.avatar_list, name='avatar_list'),
    path('avatars/create/', views.avatar_create, name='avatar_create'),
    path('avatars/<uuid:pk>/', views.avatar_detail, name='avatar_detail'),
    path('avatars/<uuid:pk>/rename/', views.avatar_rename, name='avatar_rename'),
    path('avatars/<uuid:pk>/fork/', views.avatar_fork, name='avatar_fork'),
    path('avatars/<uuid:pk>/delete/', views.avatar_delete, name='avatar_delete'),
    path('avatars/<uuid:pk>/fit/', views.avatar_fit, name='avatar_fit'),
    path('avatars/<uuid:pk>/fit/cancel/', views.avatar_fit_cancel, name='avatar_fit_cancel'),
    path('avatars/<uuid:pk>/edit/', views.avatar_edit, name='avatar_edit'),
    path('avatars/<uuid:pk>/edit/save/', views.avatar_edit_save, name='avatar_edit_save'),
    path('avatars/<uuid:pk>/mesh.obj', views.avatar_mesh, name='avatar_mesh'),
    path('avatars/<uuid:pk>/mesh/rebuild/', views.avatar_mesh_rebuild, name='avatar_mesh_rebuild'),
    path('avatars/<uuid:pk>/mesh/morph/', views.avatar_mesh_morph, name='avatar_mesh_morph'),

    # API
    path('videos/<uuid:pk>/stream/', views.video_stream, name='video_stream'),
    path('videos/<uuid:pk>/delete/', views.video_delete, name='video_delete'),
    path('api/job/<uuid:pk>/status/', views.job_status, name='job_status'),
    path('api/avatar/<uuid:pk>/log/', views.avatar_log, name='avatar_log'),
    path('api/avatar/<uuid:pk>/export/', views.avatar_export, name='avatar_export'),
    path('api/avatar/<uuid:pk>/versions/', views.avatar_versions, name='avatar_versions'),
    path('avatars/<uuid:pk>/previews/<int:n>/', views.avatar_preview_image, name='avatar_preview_image'),
    path('persons/group/<uuid:pk>/rename/', views.group_rename, name='group_rename'),
    path('persons/group/<uuid:pk>/merge/', views.group_merge, name='group_merge'),
    path('persons/group/<uuid:pk>/shape/fit/', views.group_shape_fit, name='group_shape_fit'),
    path('persons/group/<uuid:pk>/shape/progress/', views.group_shape_progress, name='group_shape_progress'),
    path('persons/group/<uuid:pk>/shape/cancel/', views.group_shape_cancel, name='group_shape_cancel'),

    # Settings
    path('settings/shape/', views.settings_shape, name='settings_shape'),
]
