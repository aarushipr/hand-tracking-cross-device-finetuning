import json
import os
import random

# give sequence id
def participant_id_of(seq_dir):
    return os.path.basename(os.path.normpath(seq_dir)).split("_")[0]

# hot3d's own official test split, no ground truth available
NO_GT_TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}
# this is our test set because its captured on both aria and quest
CROSS_DEVICE_HELD_OUT_PARTICIPANTS = {"P0002", "P0003", "P0010"}


def headset_of(seq_dir):
    """
    Return "Aria" or "Quest3" for one HOT3D sequence folder, read straight
    from that sequence's own metadata.json -- the same "headset" field
    that Hot3dDataProvider.get_device_type() dispatches on (see
    hot3d/data_loaders/PathProvider.py, Hot3dDataPathProvider.fromRecordingFolder).

    Reading the metadata file directly, rather than constructing a data
    provider and asking it, is deliberate: it costs one small JSON read
    per sequence instead of opening that sequence's multi-gigabyte .vrs
    recording, so the whole catalogue can be partitioned by device in
    seconds even over network storage.

    Returns None if the file is missing or malformed, so a single broken
    sequence folder is quietly excluded from the device-specific splits
    rather than crashing an entire training run at startup.
    """
    try:
        with open(os.path.join(seq_dir, "metadata.json")) as f:
            return json.load(f).get("headset")
    except (OSError, ValueError):
        return None


def filter_sequence_dirs(all_seq_dirs, split):
    usable = [d for d in all_seq_dirs
              if participant_id_of(d) not in NO_GT_TEST_PARTICIPANTS]

    held_out = [d for d in usable
                if participant_id_of(d) in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    train_pool = [d for d in usable
                  if participant_id_of(d) not in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]

    if split == "all_labeled":
        return usable

    # Training is Aria-only by design. If Quest recordings were mixed into
    # the training set, then evaluating on Quest would measure
    # generalisation to unseen *subjects* on an already-seen device -- not
    # generalisation to an unseen *device*, which is the claim this thesis
    # actually makes.
    if split == "train":
        return [d for d in train_pool if headset_of(d) == "Aria"]

    # The same held-out participants, partitioned by capture device:
    #   test_aria  -- new subjects, device seen during training
    #   test_quest -- new subjects, device never seen during training
    # The difference between these two numbers isolates the cross-device
    # generalisation gap specifically, because subject identity is held
    # constant across them: the same people, recorded on both headsets.
    if split == "test_aria":
        return [d for d in held_out if headset_of(d) == "Aria"]
    if split == "test_quest":
        return [d for d in held_out if headset_of(d) == "Quest3"]

    # Both devices' held-out sequences together. Retained so the existing
    # hot3d_baseline_detection_eval.py caller keeps working unchanged.
    if split == "cross_device_test":
        return held_out

    raise ValueError(f"unknown split {split!r}")


def split_train_val(seq_dirs, val_fraction=0.1, seed=0):
    """
    Carve a validation subset out of an already-filtered sequence list.

    The split is at sequence level, so no two frames from the same
    recording can land on opposite sides of it (temporal leakage).
    Participants may appear in both train and val, and that is
    intentional: val exists only to monitor convergence and to select
    checkpoints. Generalisation is measured exclusively on the test_aria
    and test_quest splits, whose participants this function never sees.
    """
    seq_dirs = sorted(seq_dirs)
    rng = random.Random(seed)
    rng.shuffle(seq_dirs)

    val_size = int(len(seq_dirs) * val_fraction)
    val_dirs = seq_dirs[:val_size]
    train_dirs = seq_dirs[val_size:]
    return train_dirs, val_dirs


def list_sequence_dirs(dataset_root, split):
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    all_dirs = sorted(
        os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("P0"))
    return filter_sequence_dirs(all_dirs, split)


def _report(dataset_root):
    """Print the real split sizes for a dataset root. See __main__ below."""
    print(f"HOT3D catalogue at {dataset_root}\n")

    for split in ("all_labeled", "train", "test_aria", "test_quest"):
        dirs = list_sequence_dirs(dataset_root, split)
        participants = sorted({participant_id_of(d) for d in dirs})
        print(f"  {split:16s} {len(dirs):4d} sequences  "
              f"{len(participants):2d} participants  {' '.join(participants)}")

    train_dirs, val_dirs = split_train_val(list_sequence_dirs(dataset_root, "train"))
    print(f"\n  train split further carved into "
          f"{len(train_dirs)} train / {len(val_dirs)} val sequences")

    headsets = {}
    for d in list_sequence_dirs(dataset_root, "all_labeled"):
        h = headset_of(d)
        headsets[h] = headsets.get(h, 0) + 1
    print(f"  headset breakdown over all labeled sequences: {headsets}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _report(sys.argv[1])
    else:
        # Offline logic check only -- the device-aware splits need real
        # metadata.json files and will come back empty against fake paths.
        fake_dirs = [
            "/x/P0001_aaa", "/x/P0002_bbb", "/x/P0003_ccc",
            "/x/P0004_ddd", "/x/P0010_eee", "/x/P0099_fff",
        ]
        print("no dataset root given -- participant-level logic check only")
        print("cross_device_test:", filter_sequence_dirs(fake_dirs, "cross_device_test"))
        print("all_labeled:", filter_sequence_dirs(fake_dirs, "all_labeled"))
        print("\nrun with a dataset root for the real device-aware report:")
        print("  python py/training/common/hot3d_split.py <hot3d_dataset_dir>")
