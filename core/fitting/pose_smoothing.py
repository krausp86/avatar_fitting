"""
Post-hoc Savitzky-Golay smoothing of Stage 1 per-frame SMPL-X pose parameters.

Stores raw fitted parameters in PersonFramePose, then writes Savitzky-Golay
smoothed variants into the *_smooth fields.

The smoothing is applied per DetectedPerson track (not across tracks), so
identity boundaries are respected.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

_SG_WINDOW = 11   # frames — must be odd; ~1 s at typical 3-frame stride
_SG_POLY   = 3    # polynomial order


def store_and_smooth_poses(
    avatar,
    result,                          # Stage1Result
    sg_window: int = _SG_WINDOW,
    sg_poly:   int = _SG_POLY,
) -> None:
    """
    1. Bulk-create PersonFramePose rows from Stage1Result arrays.
    2. Apply Savitzky-Golay per dimension, grouped by DetectedPerson track.
    3. Bulk-update *_smooth fields.

    result.person_frame_pairs must be populated (set by run_stage1).
    """
    from scipy.signal import savgol_filter
    from ..models import PersonFramePose

    person_frame_pairs: List[Tuple] = result.person_frame_pairs or []
    N = len(person_frame_pairs)
    if N == 0:
        log.warning("store_and_smooth_poses: no frames, skipping")
        return
    if len(result.theta_t) != N:
        log.error(
            "store_and_smooth_poses: theta_t length %d != person_frame_pairs length %d — skipping",
            len(result.theta_t), N,
        )
        return

    # ── 1. Delete stale rows, bulk-create raw ────────────────────────────────
    person_ids = list({str(pfp[0]) for pfp in person_frame_pairs})
    PersonFramePose.objects.filter(person_id__in=person_ids).delete()

    objs = [
        PersonFramePose(
            person_id    = person_id,
            frame_idx    = frame_idx,
            body_pose    = result.theta_t[i].tolist(),
            global_orient= result.global_orient_t[i].tolist(),
            transl       = result.transl_t[i].tolist(),
        )
        for i, (person_id, frame_idx, _) in enumerate(person_frame_pairs)
    ]
    PersonFramePose.objects.bulk_create(objs)
    log.info("store_and_smooth_poses: %d raw pose rows stored", N)

    # ── 2. Savitzky-Golay per track ───────────────────────────────────────────
    track_indices: dict[str, list[int]] = defaultdict(list)
    for i, (person_id, _, _) in enumerate(person_frame_pairs):
        track_indices[str(person_id)].append(i)

    theta_smooth  = result.theta_t.copy()
    orient_smooth = result.global_orient_t.copy()
    transl_smooth = result.transl_t.copy()

    for person_id, idxs in track_indices.items():
        n = len(idxs)
        # Window must be <= n and odd
        w = sg_window if n >= sg_window else (n if n % 2 == 1 else n - 1)
        if w < sg_poly + 1:
            log.debug("store_and_smooth_poses: track %s too short (%d frames), skipping SG", person_id, n)
            continue

        theta_smooth[idxs]  = savgol_filter(result.theta_t[idxs],          w, sg_poly, axis=0)
        orient_smooth[idxs] = savgol_filter(result.global_orient_t[idxs],   w, sg_poly, axis=0)
        transl_smooth[idxs] = savgol_filter(result.transl_t[idxs],          w, sg_poly, axis=0)
        log.info("store_and_smooth_poses: track %s smoothed (%d frames, window=%d)", person_id, n, w)

    # ── 3. Write smoothed values back ─────────────────────────────────────────
    pair_lookup = {
        (str(person_id), int(frame_idx)): i
        for i, (person_id, frame_idx, _) in enumerate(person_frame_pairs)
    }

    rows = list(PersonFramePose.objects.filter(person_id__in=person_ids).order_by('person_id', 'frame_idx'))
    for row in rows:
        i = pair_lookup.get((str(row.person_id), row.frame_idx))
        if i is None:
            continue
        row.body_pose_smooth     = theta_smooth[i].tolist()
        row.global_orient_smooth = orient_smooth[i].tolist()
        row.transl_smooth        = transl_smooth[i].tolist()

    PersonFramePose.objects.bulk_update(
        rows,
        ['body_pose_smooth', 'global_orient_smooth', 'transl_smooth'],
        batch_size=500,
    )
    log.info("store_and_smooth_poses: smoothed values written (%d rows)", len(rows))


def smooth_new_fields(
    person_ids: list[str],
    sg_window:  int = _SG_WINDOW,
    sg_poly:    int = _SG_POLY,
) -> None:
    """
    Savitzky-Golay Smoothing für die neuen Phase-B Felder:
    expression, jaw_pose, left_hand_pose, right_hand_pose.

    Wird nach run_phase_b_results aufgerufen.
    """
    from scipy.signal import savgol_filter
    from ..models import PersonFramePose

    person_ids = list(set(person_ids))
    rows_all = list(PersonFramePose.objects
                    .filter(person_id__in=person_ids)
                    .order_by('person_id', 'frame_idx'))

    if not rows_all:
        return

    # Gruppieren nach Person
    by_person: dict[str, list] = defaultdict(list)
    for row in rows_all:
        by_person[str(row.person_id)].append(row)

    fields_to_smooth = [
        ('expression',      'expression_smooth'),
        ('jaw_pose',        'jaw_pose_smooth'),
        ('left_hand_pose',  'left_hand_pose_smooth'),
        ('right_hand_pose', 'right_hand_pose_smooth'),
    ]

    updated_rows = []
    for person_id, rows in by_person.items():
        n = len(rows)
        w = sg_window if n >= sg_window else (n if n % 2 == 1 else max(1, n - 1))
        apply_sg = w >= sg_poly + 1

        for raw_field, smooth_field in fields_to_smooth:
            raw_data = np.array([getattr(r, raw_field) or [] for r in rows],
                                dtype=np.float32)  # (n, D)
            if raw_data.ndim < 2 or raw_data.shape[1] == 0:
                continue
            if apply_sg:
                smoothed = savgol_filter(raw_data, w, sg_poly, axis=0)
            else:
                smoothed = raw_data
            for i, row in enumerate(rows):
                setattr(row, smooth_field, smoothed[i].tolist())
        updated_rows.extend(rows)
        log.info("smooth_new_fields: person %s smoothed (%d frames, window=%d)",
                 person_id, n, w)

    update_fields = [sf for _, sf in fields_to_smooth]
    PersonFramePose.objects.bulk_update(updated_rows, update_fields, batch_size=500)
    log.info("smooth_new_fields: %d rows aktualisiert", len(updated_rows))
