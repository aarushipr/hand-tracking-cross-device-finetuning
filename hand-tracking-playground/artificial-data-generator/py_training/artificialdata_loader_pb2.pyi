from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class numSequencesReply(_message.Message):
    __slots__ = ["num"]
    NUM_FIELD_NUMBER: _ClassVar[int]
    num: int
    def __init__(self, num: _Optional[int] = ...) -> None: ...

class numSequencesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class sampleReply(_message.Message):
    __slots__ = ["alpha_data", "alpha_data_valid", "dotprod_elbowdir_wristloc", "elbow_direction_px_x_normalized", "elbow_direction_px_y_normalized", "homothety_scale", "image_data", "image_width_height", "keypoints_gt_depth_rel_midpxm", "keypoints_gt_px_x", "keypoints_gt_px_y", "keypoints_pose_predicted_depth_rel_midpxm", "keypoints_pose_predicted_px_x", "keypoints_pose_predicted_px_y", "move_overall_variance", "move_per_joint_variance", "prediction_type", "sg_expand_val", "stereographic_radius"]
    ALPHA_DATA_FIELD_NUMBER: _ClassVar[int]
    ALPHA_DATA_VALID_FIELD_NUMBER: _ClassVar[int]
    DOTPROD_ELBOWDIR_WRISTLOC_FIELD_NUMBER: _ClassVar[int]
    ELBOW_DIRECTION_PX_X_NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    ELBOW_DIRECTION_PX_Y_NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    HOMOTHETY_SCALE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    IMAGE_WIDTH_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_GT_DEPTH_REL_MIDPXM_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_GT_PX_X_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_GT_PX_Y_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_POSE_PREDICTED_DEPTH_REL_MIDPXM_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_POSE_PREDICTED_PX_X_FIELD_NUMBER: _ClassVar[int]
    KEYPOINTS_POSE_PREDICTED_PX_Y_FIELD_NUMBER: _ClassVar[int]
    MOVE_OVERALL_VARIANCE_FIELD_NUMBER: _ClassVar[int]
    MOVE_PER_JOINT_VARIANCE_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    SG_EXPAND_VAL_FIELD_NUMBER: _ClassVar[int]
    STEREOGRAPHIC_RADIUS_FIELD_NUMBER: _ClassVar[int]
    alpha_data: bytes
    alpha_data_valid: bool
    dotprod_elbowdir_wristloc: float
    elbow_direction_px_x_normalized: float
    elbow_direction_px_y_normalized: float
    homothety_scale: float
    image_data: bytes
    image_width_height: int
    keypoints_gt_depth_rel_midpxm: _containers.RepeatedScalarFieldContainer[float]
    keypoints_gt_px_x: _containers.RepeatedScalarFieldContainer[float]
    keypoints_gt_px_y: _containers.RepeatedScalarFieldContainer[float]
    keypoints_pose_predicted_depth_rel_midpxm: _containers.RepeatedScalarFieldContainer[float]
    keypoints_pose_predicted_px_x: _containers.RepeatedScalarFieldContainer[float]
    keypoints_pose_predicted_px_y: _containers.RepeatedScalarFieldContainer[float]
    move_overall_variance: float
    move_per_joint_variance: float
    prediction_type: int
    sg_expand_val: float
    stereographic_radius: float
    def __init__(self, image_data: _Optional[bytes] = ..., image_width_height: _Optional[int] = ..., keypoints_gt_px_x: _Optional[_Iterable[float]] = ..., keypoints_gt_px_y: _Optional[_Iterable[float]] = ..., keypoints_gt_depth_rel_midpxm: _Optional[_Iterable[float]] = ..., keypoints_pose_predicted_px_x: _Optional[_Iterable[float]] = ..., keypoints_pose_predicted_px_y: _Optional[_Iterable[float]] = ..., keypoints_pose_predicted_depth_rel_midpxm: _Optional[_Iterable[float]] = ..., stereographic_radius: _Optional[float] = ..., homothety_scale: _Optional[float] = ..., move_overall_variance: _Optional[float] = ..., move_per_joint_variance: _Optional[float] = ..., sg_expand_val: _Optional[float] = ..., prediction_type: _Optional[int] = ..., elbow_direction_px_x_normalized: _Optional[float] = ..., elbow_direction_px_y_normalized: _Optional[float] = ..., dotprod_elbowdir_wristloc: _Optional[float] = ..., alpha_data: _Optional[bytes] = ..., alpha_data_valid: bool = ...) -> None: ...

class sampleRequest(_message.Message):
    __slots__ = ["frame_idx", "sequence_idx"]
    FRAME_IDX_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_IDX_FIELD_NUMBER: _ClassVar[int]
    frame_idx: int
    sequence_idx: int
    def __init__(self, sequence_idx: _Optional[int] = ..., frame_idx: _Optional[int] = ...) -> None: ...
