"""
The cross-device participant split, shared by every HOT3D-consuming script in
this project.

This used to be defined only inside py/evaluation/hot3d_baseline_detection_eval.py.
Pulled out into its own module (Chapter 4 Section 4.1's third gap: DetNet's
training loop wasn't reading HOT3D data at all) so that CombinedDataset.py's
training-time filtering and hot3d_baseline_detection_eval.py's evaluation-time
filtering are guaranteed to agree -- importing the same constants and function
from one place, rather than keeping two hand-maintained copies that could
silently drift apart (e.g. one script's NO_GT_TEST_PARTICIPANTS getting
updated without the other). See FOUR_WEEK_SUBMISSION_PLAN.md and
THESIS_STRUCTURE_v3.md Chapter 5 for how this split was derived.
"""
import os

# Confirmed via participant-overlap check against the actual downloaded
# manifests (Hot3DAria_download_urls.json / Hot3DQuest_download_urls.json),
# not the HOT3D repo's own docs alone.
NO_GT_TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}
CROSS_DEVICE_HELD_OUT_PARTICIPANTS = {"P0002", "P0003", "P0010"}  # captured on both Aria + Quest, has GT


def participant_id_of(seq_dir):
    return os.path.basename(os.path.normpath(seq_dir)).split("_")[0]


def filter_sequence_dirs(all_seq_dirs, split):
    """split in {'train', 'cross_device_test', 'all_labeled'}. Always drops
    the no-GT official HOT3D test participants -- they have no annotations
    to train or evaluate against regardless of split."""
    usable = [d for d in all_seq_dirs if participant_id_of(d) not in NO_GT_TEST_PARTICIPANTS]
    if split == "all_labeled":
        return usable
    if split == "cross_device_test":
        return [d for d in usable if participant_id_of(d) in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    if split == "train":
        return [d for d in usable if participant_id_of(d) not in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    raise ValueError(f"unknown split {split!r}")


def list_sequence_dirs(dataset_root, split):
    """Convenience wrapper: scan dataset_root for P00xx_* sequence folders and
    filter them by `split`. Returns [] (not an error) if dataset_root doesn't
    exist, so callers can use the graceful-skip pattern already used
    throughout CombinedDataset.py rather than crashing on an unconfigured
    local_config path."""
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    all_dirs = sorted(
        os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("P0"))
    return filter_sequence_dirs(all_dirs, split)
