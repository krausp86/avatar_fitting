# Joint name lists and skeleton connections for all supported backends.

# ── MediaPipe Pose (33 landmarks) ────────────────────────────────────────────
MEDIAPIPE_JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

MEDIAPIPE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# ── COCO 17 keypoints (ViTPose default output) ────────────────────────────────
COCO17_JOINT_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

COCO17_CONNECTIONS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ── COCO-WholeBody 133 keypoints (RTMPose-L Wholebody) ───────────────────────
# 0-16:   body (COCO17 order)
# 17-22:  feet
# 23-90:  face (68 landmarks)
# 91-111: left hand (21 landmarks)
# 112-132: right hand (21 landmarks)
COCO_WHOLEBODY_JOINT_NAMES = (
    COCO17_JOINT_NAMES +
    ['left_big_toe', 'left_small_toe', 'left_heel',
     'right_big_toe', 'right_small_toe', 'right_heel'] +
    [f'face_{i}' for i in range(68)] +
    [f'lhand_{i}' for i in range(21)] +
    [f'rhand_{i}' for i in range(21)]
)

COCO_WHOLEBODY_CONNECTIONS = (
    COCO17_CONNECTIONS +
    [(15, 19), (15, 17), (15, 18),   # left ankle → foot
     (16, 22), (16, 20), (16, 21)]   # right ankle → foot
)
