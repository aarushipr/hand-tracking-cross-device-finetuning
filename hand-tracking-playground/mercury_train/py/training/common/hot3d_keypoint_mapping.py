import numpy as np
# Maps each of this project's 21 keypoint slots to the corresponding
# index in HOT3D's own get_hand_landmarks() output (see
# hot3d/hot3d/data_loaders/hand_common.py's LANDMARK_INDEX_TO_NAMING
# for HOT3D's source ordering). A tuple entry means "average these two
# HOT3D indices" -- used only for thumb_mcp, since HOT3D has no
# separate thumb-base joint (only thumb_intermediate + thumb_distal);
# we approximate it as the midpoint between the wrist and
# thumb_intermediate rather than duplicating a single point outright.
_HOT3D_LANDMARK_PERMUTATION = [
    5,        # 0  wrist        <- HOT3D wrist_joint
    (5, 6),   # 1  thumb_mcp    <- midpoint(wrist, thumb_intermediate) [approximated]
    6,        # 2  thumb_pxm    <- HOT3D thumb_intermediate_frame
    7,        # 3  thumb_dst    <- HOT3D thumb_distal_frame
    0,        # 4  thumb_tip    <- HOT3D thumb_fingertip
    8,        # 5  index_pxm
    9,        # 6  index_int
    10,       # 7  index_dst
    1,        # 8  index_tip
    11,       # 9  middle_pxm
    12,       # 10 middle_int
    13,       # 11 middle_dst
    2,        # 12 middle_tip
    14,       # 13 ring_pxm
    15,       # 14 ring_int
    16,       # 15 ring_dst
    3,        # 16 ring_tip
    17,       # 17 pinky_pxm
    18,       # 18 pinky_int
    19,       # 19 pinky_dst
    4,        # 20 pinky_tip
    # HOT3D's index 20 (palm_center) has no counterpart here and is unused.
]


def hot3d_landmarks_to_project_keypoints(hot3d_landmarks):
    """
    hot3d_landmarks: (21, 3) array from
        umetrack_hand_data_provider.get_hand_landmarks(hand_pose_data),
        in HOT3D's own LANDMARK_INDEX_TO_NAMING order.

    Returns a (21, 3) array in this project's own keypoint convention
    (matching ArtificialData.py's _25_to_21 / Monado's KeyNet order).
    """
    hot3d_landmarks = np.asarray(hot3d_landmarks)
    out = np.zeros((21, 3), dtype=np.float32)
    for project_idx, source in enumerate(_HOT3D_LANDMARK_PERMUTATION):
        if isinstance(source, tuple):
            a, b = source
            out[project_idx] = (hot3d_landmarks[a] + hot3d_landmarks[b]) / 2.0
        else:
            out[project_idx] = hot3d_landmarks[source]
    return out