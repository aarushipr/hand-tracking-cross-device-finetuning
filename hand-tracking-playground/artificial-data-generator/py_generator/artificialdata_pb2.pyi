from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class sayonara(_message.Message):
    __slots__ = ["slot_idx"]
    SLOT_IDX_FIELD_NUMBER: _ClassVar[int]
    slot_idx: int
    def __init__(self, slot_idx: _Optional[int] = ...) -> None: ...

class sequenceReply(_message.Message):
    __slots__ = ["camera_info_csv", "dont_render", "elbowpose_csv", "fingerpose_csv", "framerate", "hand_poses_csv", "num_frames", "output_alpha_images_folder", "output_color_images_folder", "render_alpha", "use_exr_background", "valid_samples_csv", "wristpose_csv"]
    CAMERA_INFO_CSV_FIELD_NUMBER: _ClassVar[int]
    DONT_RENDER_FIELD_NUMBER: _ClassVar[int]
    ELBOWPOSE_CSV_FIELD_NUMBER: _ClassVar[int]
    FINGERPOSE_CSV_FIELD_NUMBER: _ClassVar[int]
    FRAMERATE_FIELD_NUMBER: _ClassVar[int]
    HAND_POSES_CSV_FIELD_NUMBER: _ClassVar[int]
    NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ALPHA_IMAGES_FOLDER_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_COLOR_IMAGES_FOLDER_FIELD_NUMBER: _ClassVar[int]
    RENDER_ALPHA_FIELD_NUMBER: _ClassVar[int]
    USE_EXR_BACKGROUND_FIELD_NUMBER: _ClassVar[int]
    VALID_SAMPLES_CSV_FIELD_NUMBER: _ClassVar[int]
    WRISTPOSE_CSV_FIELD_NUMBER: _ClassVar[int]
    camera_info_csv: str
    dont_render: bool
    elbowpose_csv: str
    fingerpose_csv: str
    framerate: float
    hand_poses_csv: str
    num_frames: int
    output_alpha_images_folder: str
    output_color_images_folder: str
    render_alpha: bool
    use_exr_background: bool
    valid_samples_csv: str
    wristpose_csv: str
    def __init__(self, num_frames: _Optional[int] = ..., wristpose_csv: _Optional[str] = ..., elbowpose_csv: _Optional[str] = ..., fingerpose_csv: _Optional[str] = ..., use_exr_background: bool = ..., render_alpha: bool = ..., hand_poses_csv: _Optional[str] = ..., camera_info_csv: _Optional[str] = ..., valid_samples_csv: _Optional[str] = ..., output_color_images_folder: _Optional[str] = ..., output_alpha_images_folder: _Optional[str] = ..., dont_render: bool = ..., framerate: _Optional[float] = ...) -> None: ...

class sequenceRequest(_message.Message):
    __slots__ = ["proportions_json", "slot_idx"]
    PROPORTIONS_JSON_FIELD_NUMBER: _ClassVar[int]
    SLOT_IDX_FIELD_NUMBER: _ClassVar[int]
    proportions_json: str
    slot_idx: int
    def __init__(self, slot_idx: _Optional[int] = ..., proportions_json: _Optional[str] = ...) -> None: ...
