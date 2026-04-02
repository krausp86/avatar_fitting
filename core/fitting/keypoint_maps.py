"""
Joint-Mapping-Tabellen für die neue SMPL-X Fitting-Pipeline.

Quellen:
  - Body:  ViTPose COCO-17
  - Face:  RTMPose Wholebody (dlib-68 Format, Offset 23 in 133-Punkt-Array)
  - Hands: RTMPose Wholebody (MediaPipe-Stil, Left: 91-111, Right: 112-132)
  - SMPL-X joints: 55 Joints (22 Body + 3 Face + 15×2 Hände)

RTMPose 133-Punkt-Layout:
  0-16   : COCO-17 Body
  17-22  : Fuß-Refinement (6 Punkte)
  23-90  : Gesicht (68 dlib-Punkte)
  91-111 : Linke Hand (21 MediaPipe-Punkte)
  112-132: Rechte Hand (21 MediaPipe-Punkte)
"""
from __future__ import annotations
from typing import List, Tuple

# ── Body: SMPL-X joints → COCO-17 ──────────────────────────────────────────
# Konsistent mit bisherigem stage1.py _SMPLX_IDX / _MP_IDX.
# (SMPL-X joint index, COCO-17 index)
BODY_PAIRS: List[Tuple[int, int]] = [
    (1,  11),  # l_hip       → left_hip
    (2,  12),  # r_hip       → right_hip
    (4,  13),  # l_knee      → left_knee
    (5,  14),  # r_knee      → right_knee
    (7,  15),  # l_ankle     → left_ankle
    (8,  16),  # r_ankle     → right_ankle
    (16,  5),  # l_shoulder  → left_shoulder
    (17,  6),  # r_shoulder  → right_shoulder
    (18,  7),  # l_elbow     → left_elbow
    (19,  8),  # r_elbow     → right_elbow
    (20,  9),  # l_wrist     → left_wrist
    (21, 10),  # r_wrist     → right_wrist
]

BODY_SMPLX_IDX:  List[int] = [p[0] for p in BODY_PAIRS]
BODY_COCO17_IDX: List[int] = [p[1] for p in BODY_PAIRS]

# ── Face: SMPL-X FLAME-Landmarks → RTMPose dlib-68 ─────────────────────────
# RTMPose liefert Gesichts-Landmarks als rtm_landmarks[23 + dlib_i].
# Die 68 FLAME/dlib-Punkte werden via barycentric Interpolation auf SMPL-X
# Vertices projiziert (smplx_model.lmk_faces_idx + lmk_bary_coords).
# Diese Konstante ist der Offset im RTMPose-133-Array:
FACE_RTM_OFFSET = 23           # rtm_landmarks[FACE_RTM_OFFSET + dlib_i]
N_FACE_LANDMARKS = 68

# Direkte SMPL-X-Joint → dlib-Landmark Approximationen
# (nützlich als Fallback, wenn lmk_bary_coords nicht verfügbar)
# (SMPL-X joint idx, dlib-68 idx)
FACE_JOINT_PAIRS: List[Tuple[int, int]] = [
    (22, 8),   # jaw joint       → chin (dlib 8)
    (23, 42),  # l_eye_ball      → linkes Augen-Inneneck (dlib 42)
    (23, 45),  # l_eye_ball      → linkes Augen-Außeneck (dlib 45)
    (24, 36),  # r_eye_ball      → rechtes Augen-Inneneck (dlib 36)
    (24, 39),  # r_eye_ball      → rechtes Augen-Außeneck (dlib 39)
]

# ── Hände: SMPL-X joints → MediaPipe Hand-Landmarks ────────────────────────
# SMPL-X Linke Hand: Joints 25-39 (5 Finger × 3 Gelenke: MCP, PIP, DIP)
# SMPL-X Rechte Hand: Joints 40-54
# RTMPose Linke Hand: Indices 91-111 (21 MediaPipe-Punkte, Offset 91)
# RTMPose Rechte Hand: Indices 112-132 (Offset 112)
#
# MediaPipe Hand Topologie (21 Punkte):
#   0: Handgelenk (wrist) → bereits in BODY_PAIRS via SMPL-X 20/21 abgedeckt
#   1-4:   Daumen  (CMC, MCP, IP, tip)   tip=4 fehlt in SMPL-X
#   5-8:   Zeigefinger (MCP,PIP,DIP,tip)  tip=8 fehlt
#   9-12:  Mittelfinger                   tip=12 fehlt
#   13-16: Ringfinger                     tip=16 fehlt
#   17-20: Kleinfinger                    tip=20 fehlt
#
# SMPL-X Finger-Reihenfolge (links, 15 Joints):
#   Index(25,26,27), Middle(28,29,30), Pinky(31,32,33),
#   Ring(34,35,36), Thumb(37,38,39)
#
# Mapping (SMPL-X joint, MediaPipe local idx 0-20):
LHAND_PAIRS: List[Tuple[int, int]] = [
    # Daumen: MP 1,2,3 → SMPL-X 37(CMC-ish),38(MCP),39(IP)
    (37, 1), (38, 2), (39, 3),
    # Zeigefinger: MP 5,6,7 → SMPL-X 25,26,27
    (25, 5), (26, 6), (27, 7),
    # Mittelfinger: MP 9,10,11 → SMPL-X 28,29,30
    (28, 9), (29, 10), (30, 11),
    # Ringfinger: MP 13,14,15 → SMPL-X 34,35,36
    (34, 13), (35, 14), (36, 15),
    # Kleinfinger: MP 17,18,19 → SMPL-X 31,32,33
    (31, 17), (32, 18), (33, 19),
    # Handgelenk: MP 0 → SMPL-X joint 20 (l_wrist, in BODY_PAIRS)
    (20, 0),
]

# Rechte Hand: gleiche Finger-Topologie, SMPL-X 40-54 statt 25-39
# Index+15 für SMPL-X joint, +112 statt +91 für RTMPose Index
RHAND_PAIRS: List[Tuple[int, int]] = [
    # Daumen: MP 1,2,3 → SMPL-X 52,53,54
    (52, 1), (53, 2), (54, 3),
    # Zeigefinger: MP 5,6,7 → SMPL-X 40,41,42
    (40, 5), (41, 6), (42, 7),
    # Mittelfinger: MP 9,10,11 → SMPL-X 43,44,45
    (43, 9), (44, 10), (45, 11),
    # Ringfinger: MP 13,14,15 → SMPL-X 49,50,51
    (49, 13), (50, 14), (51, 15),
    # Kleinfinger: MP 17,18,19 → SMPL-X 46,47,48
    (46, 17), (47, 18), (48, 19),
    # Handgelenk: MP 0 → SMPL-X joint 21 (r_wrist, in BODY_PAIRS)
    (21, 0),
]

LHAND_SMPLX_IDX: List[int] = [p[0] for p in LHAND_PAIRS]
LHAND_MP_IDX:    List[int] = [p[1] for p in LHAND_PAIRS]
RHAND_SMPLX_IDX: List[int] = [p[0] for p in RHAND_PAIRS]
RHAND_MP_IDX:    List[int] = [p[1] for p in RHAND_PAIRS]

# RTMPose absolute indices
LHAND_RTM_IDX: List[int] = [91 + mp for mp in LHAND_MP_IDX]
RHAND_RTM_IDX: List[int] = [112 + mp for mp in RHAND_MP_IDX]

# ── Konfidenz-Schwellen ──────────────────────────────────────────────────────
BODY_VIS_THRESHOLD  = 0.3   # ViTPose body keypoints
FACE_VIS_THRESHOLD  = 0.3   # RTMPose face landmarks
HAND_VIS_THRESHOLD  = 0.5   # RTMPose hand landmarks (RTMPose hands sind lauter)
