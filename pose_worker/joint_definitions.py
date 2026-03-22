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
    [(15, 19), (15, 17), (15, 18),
     (16, 22), (16, 20), (16, 21)]
)

# ── 68-point face connections (absolute COCO-WholeBody indices) ───────────────
# face_0 = index 23, face_67 = index 90
_F = 23
FACE_68_CONNECTIONS = (
    # Jawline (face 0–16)
    [(i, i+1) for i in range(_F+0,  _F+16)] +
    # Right eyebrow (face 17–21)
    [(i, i+1) for i in range(_F+17, _F+21)] +
    # Left eyebrow (face 22–26)
    [(i, i+1) for i in range(_F+22, _F+26)] +
    # Nose bridge (face 27–30)
    [(i, i+1) for i in range(_F+27, _F+30)] +
    # Nose bottom (face 31–35)
    [(i, i+1) for i in range(_F+31, _F+35)] +
    # Right eye (face 36–41, closed loop)
    [(i, i+1) for i in range(_F+36, _F+41)] + [(_F+41, _F+36)] +
    # Left eye (face 42–47, closed loop)
    [(i, i+1) for i in range(_F+42, _F+47)] + [(_F+47, _F+42)] +
    # Outer lips (face 48–59, closed loop)
    [(i, i+1) for i in range(_F+48, _F+59)] + [(_F+59, _F+48)] +
    # Inner lips (face 60–67, closed loop)
    [(i, i+1) for i in range(_F+60, _F+67)] + [(_F+67, _F+60)]
)

# ── Hand connections (absolute COCO-WholeBody indices) ────────────────────────
# lhand_0 = 91 … lhand_20 = 111  |  rhand_0 = 112 … rhand_20 = 132
_HAND_REL = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm cross
]
_LH, _RH = 91, 112
LHAND_CONNECTIONS = [(a+_LH, b+_LH) for a, b in _HAND_REL]
RHAND_CONNECTIONS = [(a+_RH, b+_RH) for a, b in _HAND_REL]
