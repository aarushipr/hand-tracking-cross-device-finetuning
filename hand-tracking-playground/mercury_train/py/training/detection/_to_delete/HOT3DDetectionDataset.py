"""
HOT3D (and UmeTrack, same underlying format) detection loader -- full
frames with bounding boxes derived from real 3D hand pose + real camera
calibration, for DetNet training/val/test.

Unlike SyntheticDetectionDataset (which reimplements a plain pinhole
projection, fine since our own Blender camera IS a plain pinhole), this
uses hand_tracking_toolkit's own CameraModel.world_to_window3(), which
composes the correct projection + distortion for whatever camera model
the data actually uses (Aria/Quest headsets use fisheye models -- a
hand-rolled pinhole projection would silently give wrong bboxes on this
data). See camera.py's CameraModel.world_to_window3 -- it's defined once
on the base class and works uniformly for every camera model in the
toolkit, fisheye included.

Uses build_hand_dataset(..., output_crops=False), which (per dataset.py)
returns one HandData per FRAME (not per hand): images/cameras per stream
ID, plus hand_poses as an Optional[Dict[HandSide, HandPoseCollection]] --
0, 1, or 2 entries, already keyed by real left/right handedness. That
maps directly onto this project's bbox_list = [left, right] convention,
no handedness guessing needed (unlike the synthetic single-hand data).

Usage (as a real dataset source):
    ds = HOT3DDetectionDataset(root="/storage/user/praa/hot3d_sample/train_aria",
                                sequence_names=["clip-001849"])

Wire into CombinedDataset.py similarly to SyntheticDetectionDataset, once
a visual check (verify_hot3d_visual.py) confirms the bboxes land correctly
-- same "verify before trusting" approach used for the UmeTrack keypoint
converter, since this is new code touching real (not synthetic) data.
"""
import numpy as np
import torch

import augmentation
from a_structs import ImageWithBoundingBoxes, bbox


class HOT3DDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, sequence_names: list, margin: float = 0.15):
        from hand_tracking_toolkit.dataset import build_hand_dataset, HandSide
        from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics

        self._HandSide = HandSide
        self._forward_kinematics = forward_kinematics
        self.margin = margin

        # webdataset-backed -- materialize into a list up front so this
        # behaves like a normal indexable torch Dataset (matches how
        # ArtificialDataset/RandoDataset/other loaders in this repo work,
        # and how CombinedDataset's num_times_to_repeat weighting expects
        # len()/__getitem__ to behave). Real per-source frame counts here
        # are small (single-digit thousands per clip), so this is fine.
        raw_samples = list(build_hand_dataset(
            root, sequence_names,
            load_monochrome=True, load_rgb=False,
            output_crops=False,
        ))

        # Aria/Quest headsets have multiple monochrome streams per frame
        # (e.g. side-facing wide-FOV SLAM cameras vs. others) -- an early
        # version of this loader picked whichever stream ID sorted first
        # alphabetically, which turned out to frequently NOT have a good
        # view of the hands (verified visually: boxes landing on empty
        # background in ~half of a random sample). Rather than guess which
        # specific stream ID is "the good one" without documentation, use
        # every available stream as its own training sample -- bad-angle
        # streams naturally end up with exists=0 via augment_image's
        # existing box_in_image visibility filter, good-angle streams
        # contribute real data. Also multiplies available data for free,
        # which the original UmeTrack converter's docstring flagged as a
        # natural follow-up.
        self.frame_stream_pairs = [
            (sample, stream_id)
            for sample in raw_samples
            for stream_id in sample.images.keys()
        ]

    def __len__(self):
        return len(self.frame_stream_pairs)

    def __getitem__(self, idx):
        sample, stream_id = self.frame_stream_pairs[idx]

        if not sample.images or not sample.cameras:
            return self._empty_sample()

        image = sample.images[stream_id]
        camera = sample.cameras[stream_id]

        bbox_list = [None, None]

        if sample.hand_poses and sample.hand_shape is not None:
            for hand_side, hand_pose_collection in sample.hand_poses.items():
                if hand_pose_collection.umetrack is None:
                    continue

                landmarks, _, _ = self._forward_kinematics(
                    hand_pose_collection.umetrack, sample.hand_shape.umetrack,
                    requires_mesh=False,
                )
                landmarks = landmarks.detach().cpu().numpy()  # (20, 3), world space

                wrist_pos = (
                    hand_pose_collection.umetrack.wrist_xform.detach().cpu().numpy()[:3, 3]
                )
                points_world = np.vstack([wrist_pos[None, :], landmarks])  # (21, 3)

                # Guard against fisheye distortion polynomials producing a
                # plausible-looking but wrong window coordinate for points
                # well outside the camera's real field of view (a known
                # limitation of polynomial distortion fits) -- same check
                # dataset.py's own warp_image() uses ("mask out points with
                # negative z coordinates"). Without this, a hand that's
                # actually behind/beside this particular camera stream can
                # still produce an in-bounds-looking bbox that doesn't
                # correspond to anything visible in the image -- this is
                # what verify_hot3d_visual.py caught (~half of a random
                # sample had boxes not landing on any visible hand).
                eye_pts = camera.world_to_eye(points_world)
                if np.any(eye_pts[:, 2] <= 0):
                    continue

                # world_to_window3 handles projection + distortion + window
                # scaling correctly for whatever camera model this stream
                # actually uses (fisheye for Aria/Quest, unlike our own
                # synthetic pinhole camera) -- see module docstring.
                win = camera.world_to_window3(points_world)  # (21, 3): u, v, depth
                u, v = win[:, 0], win[:, 1]

                x0, x1 = float(u.min()), float(u.max())
                y0, y1 = float(v.min()), float(v.max())
                w, h = x1 - x0, y1 - y0
                x0 -= w * self.margin
                x1 += w * self.margin
                y0 -= h * self.margin
                y1 += h * self.margin

                b = bbox((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
                slot = 0 if hand_side == self._HandSide.LEFT else 1
                bbox_list[slot] = b

        e = ImageWithBoundingBoxes(image=image, bboxes=bbox_list)
        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)
        return e

    def _empty_sample(self):
        # Extremely unlikely (a frame with literally no camera stream),
        # but keep __getitem__ total rather than crashing a whole epoch.
        import header
        blank = np.zeros((header.model_input_height, header.model_input_width), dtype=np.uint8)
        e = ImageWithBoundingBoxes(image=blank, bboxes=[None, None])
        return augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
    ds = HOT3DDetectionDataset(root=sys.argv[1], sequence_names=sys.argv[2:])
    print(f"{len(ds)} samples")
    samp = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})
