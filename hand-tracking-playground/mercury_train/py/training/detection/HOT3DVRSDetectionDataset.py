"""
HOT3D full (VRS-based) detection loader -- reads Meta's own pre-computed,
occlusion-aware 2D hand bounding boxes from box2d_hands.csv directly,
instead of re-deriving boxes from 3D keypoint projection the way
HOT3DDetectionDataset.py (built against the lightweight HOT3D-Clips /
WebDataset format) has to.

Why this exists (see prior HOT3DDetectionDataset.py visual verification):
projecting 3D hand keypoints through the camera model only tells you where
the hand *geometrically* is relative to the camera frustum -- it can't know
the hand is occluded by the body, an object, or the headset frame. That was
the irreducible remaining error after fixing camera-stream selection and
adding a behind-camera depth guard. The full VRS-based HOT3D dataset ships
box2d_hands.csv, which is Meta's own ground-truth 2D box per hand per
camera stream per timestamp, INCLUDING a visibility_ratio[%] field -- i.e.
already occlusion-aware, no projection math needed at all here.

Source of truth for the CSV schema / reader API (Apache 2.0, read not
vendored -- imported directly from the cloned hot3d repo, same repo already
cloned on the cluster to build the `hot3d` conda env for downloading):
    hot3d/hot3d/data_loaders/HandBox2dDataProvider.py
    hot3d/hot3d/data_loaders/PathProvider.py
    hot3d/hot3d/data_loaders/AriaDataProvider.py

Deliberately does NOT build a full Hot3dDataProvider (dataset_api.py) --
that requires an ObjectLibrary (object models + eval sets), which would mean
downloading the whole Hot3DAssets bundle just to read hand boxes we don't
need object data for at all. AriaDataProvider (image reads from
recording.vrs) + HandBox2dProvider (box2d_hands.csv) are self-contained and
don't need it.

hand_index -> left/right slot mapping is ASSUMED (0=left, 1=right, matching
UmeTrack's HandSide convention used everywhere else in this project) but
NOT yet confirmed against this specific CSV -- verify visually with
verify_hot3d_vrs_visual.py the same way EgoHands' class-index assumption
was checked (verify_egohands_classes.py) before trusting it for real
training.

Usage:
    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=["/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/P0003_c701bd11"],
        hot3d_repo_root="/storage/user/praa/hot3d_full_setup/hot3d/hot3d",
        min_visibility_ratio=20.0,
    )
"""
import sys

import numpy as np
import torch

import augmentation
from a_structs import ImageWithBoundingBoxes, bbox


class HOT3DVRSDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dirs: list, hot3d_repo_root: str,
                 min_visibility_ratio: float = 20.0, margin: float = 0.15):
        # hot3d's data_loaders package uses bare `from data_loaders.X import Y`
        # (relative to hot3d/hot3d), so that directory has to be on sys.path
        # before these imports work -- same pattern as this project's own
        # `import py.training.common.X` needing mercury_train root on
        # sys.path (see augmentation.py / trainer_detection.py __main__).
        if hot3d_repo_root not in sys.path:
            sys.path.insert(0, hot3d_repo_root)

        from data_loaders.PathProvider import Hot3dDataPathProvider
        from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
        from data_loaders.AriaDataProvider import AriaDataProvider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        from projectaria_tools.core.stream_id import StreamId

        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self._StreamId = StreamId
        self.min_visibility_ratio = min_visibility_ratio
        self.margin = margin

        # One (aria_provider, box2d_provider, stream_id, timestamp_ns) tuple
        # per sample. Iterate every image stream's own timestamps (not just
        # the box2d CSV's timestamps) so frames with zero visible hands
        # (both hand_index entries below the visibility threshold, or
        # missing from the CSV entirely) still show up as real exists=0
        # negatives -- same reasoning as EgoHands filling the "no hand
        # present" gap for DetNet.
        self.samples = []
        for seq_dir in sequence_dirs:
            paths = Hot3dDataPathProvider.fromRecordingFolder(seq_dir)
            if not paths.is_valid():
                print(f"WARNING: {seq_dir} missing required files, skipping")
                continue

            box2d_provider = load_box2d_trajectory_from_csv(paths.box2d_hands_filepath)
            if box2d_provider is None:
                print(f"WARNING: {seq_dir} has no box2d_hands.csv, skipping")
                continue

            # mps_folder_path intentionally omitted -- get_image()/
            # get_image_stream_ids() don't touch MPS data, and pulling in
            # eye-gaze/point-cloud calibration would be dead weight for a
            # detection-only loader.
            aria_provider = AriaDataProvider(paths.vrs_filepath, mps_folder_path=None)

            for stream_id in aria_provider.get_image_stream_ids():
                timestamps = aria_provider.get_sequence_timestamps(
                    stream_id, TimeDomain.TIME_CODE
                )
                for ts in timestamps:
                    self.samples.append((aria_provider, box2d_provider, stream_id, ts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aria_provider, box2d_provider, stream_id, ts = self.samples[idx]

        image = aria_provider.get_image(ts, stream_id)
        if image is None:
            return self._empty_sample()

        bbox_list = [None, None]

        if self._StreamId(str(stream_id)) in box2d_provider.stream_ids:
            result = box2d_provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=ts,
                time_query_options=self._TimeQueryOptions.CLOSEST,
                time_domain=self._TimeDomain.TIME_CODE,
            )
            if result is not None:
                for hand_index, hand_box in result.box2d_collection.box2ds.items():
                    if hand_box.box2d is None:
                        continue
                    if hand_box.visibility_ratio is None:
                        continue
                    if hand_box.visibility_ratio < self.min_visibility_ratio:
                        continue

                    b2d = hand_box.box2d
                    w, h = b2d.right - b2d.left, b2d.bottom - b2d.top
                    x0 = b2d.left - w * self.margin
                    x1 = b2d.right + w * self.margin
                    y0 = b2d.top - h * self.margin
                    y1 = b2d.bottom + h * self.margin

                    b = bbox((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
                    # ASSUMED 0=left, 1=right -- not yet visually confirmed
                    # for this CSV, see module docstring.
                    slot = 0 if hand_index == 0 else 1
                    bbox_list[slot] = b

        e = ImageWithBoundingBoxes(image=image, bboxes=bbox_list)
        e = augmentation.augment_image(e)
        e = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)
        return e

    def _empty_sample(self):
        import header
        blank = np.zeros((header.model_input_height, header.model_input_width), dtype=np.uint8)
        e = ImageWithBoundingBoxes(image=blank, bboxes=[None, None])
        return augmentation.imgwithboundingboxes320_to_heatmaps_2hand(e)


if __name__ == "__main__":
    import argparse
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dirs", required=True, nargs="+")
    parser.add_argument("--hot3d-repo-root", required=True)
    parser.add_argument("--min-visibility-ratio", type=float, default=20.0)
    args = parser.parse_args()

    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=args.sequence_dirs,
        hot3d_repo_root=args.hot3d_repo_root,
        min_visibility_ratio=args.min_visibility_ratio,
    )
    print(f"{len(ds)} samples")
    samp = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in samp.items()})
