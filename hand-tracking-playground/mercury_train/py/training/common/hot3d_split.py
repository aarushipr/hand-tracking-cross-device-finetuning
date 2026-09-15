"""
Participant-level train/test partitioning of the HOT3D catalogue. Two designs live
below: the current train_mixed / test_mixed, Aria and Quest pooled and split 80/20
by participant, which the thesis reports; and the archived Aria-only design, kept
unchanged so its checkpoints and results stay scorable. Nothing in the archived
block is reachable from the current path.
"""
import json
import os
import random

# --- The catalogue ---------------------------------------------------------

def participant_id_of(seq_dir):
    """'P0003' from '/.../P0003_c701bd11'."""
    return os.path.basename(os.path.normpath(seq_dir)).split("_")[0]


# HOT3D's official test split: public recordings with withheld GT, unusable here.
NO_GT_TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}


def headset_of(seq_dir):
    """
    Return "Aria" or "Quest3" for one HOT3D sequence folder, read straight
    from that sequence's own metadata.json; the same "headset" field
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


def usable_sequence_dirs(all_seq_dirs):
    """Every sequence with usable ground truth, both devices."""
    return [d for d in all_seq_dirs
            if participant_id_of(d) not in NO_GT_TEST_PARTICIPANTS]


# --- CURRENT DESIGN: mixed Aria + Quest, 80/20 by participant --------------
# Frozen as constants, not recomputed from disk: a partial copy would shift the split.
# From _derive_mixed_split() over 294 sequences: 136 Aria, 158 Quest, 13 participants.
#     Aria 109/27 (80.1%)   Quest3 126/32 (79.7%)   total 235/59 (79.9%)  train/test
# Re-derive with: python py/training/common/hot3d_split.py <dataset_root>

TRAIN_MIXED_PARTICIPANTS = {
    "P0002", "P0003", "P0009", "P0010", "P0011", "P0012", "P0013",
    "P0015", "P0018",
}
TEST_MIXED_PARTICIPANTS = {"P0001", "P0014", "P0017", "P0021"}


def _derive_mixed_split(seq_dirs, train_fraction=0.8):
    """
    Derive the mixed split from a sequence listing. This is the documented
    provenance of TRAIN_MIXED_PARTICIPANTS / TEST_MIXED_PARTICIPANTS above; the
    training and evaluation paths use those constants, not this function.

    Partitions PARTICIPANTS (not sequences), so no subject's recordings appear
    on both sides; a subject who appears in both train and test leaks
    person-specific appearance into the test score.

    The objective is per-device, not overall. Matching only the total sequence
    count lets the optimiser satisfy 80/20 with a test set drawn almost
    entirely from one headset: on the real catalogue, a total-count objective
    produces a test set of 5 Aria against 54 Quest sequences, which is a Quest
    test set with rounding error attached, not a mixed one. Minimising the
    WORST per-device deviation instead forces both devices to sit near the
    target ratio simultaneously, which is what "mixed" has to mean if the test
    score is to describe both devices.

    Exact search over all participant subsets: with ~13 usable participants
    this is 2**13 = 8192 subsets, instant to enumerate. Revisit with a greedy
    or dynamic-programming approach if the usable pool ever grows much past
    ~20 participants.

    Deterministic: the same input sequence list always yields the same split,
    no RNG involved. Ties are broken by preferring the lexicographically
    smallest participant-ID subset, purely so re-running this against an
    unchanged dataset is reproducible.

    Returns (train_participants, test_participants) as two sets.
    """
    from itertools import combinations

    per_device = {}
    for d in usable_sequence_dirs(seq_dirs):
        pid = participant_id_of(d)
        device = headset_of(d)
        if device is None:
            # Unreadable metadata.json; takes no part in the device-aware objective.
            continue
        per_device.setdefault(device, {})
        per_device[device][pid] = per_device[device].get(pid, 0) + 1

    participants = sorted({p for counts in per_device.values() for p in counts})
    if not participants:
        return set(), set()

    totals = {dev: sum(counts.values()) for dev, counts in per_device.items()}

    best_subset, best_cost = None, None
    for r in range(len(participants) + 1):
        for subset in combinations(participants, r):
            # Relative, not absolute, deviation: a smaller device can't drift further for free.
            cost = max(
                abs(sum(per_device[dev].get(p, 0) for p in subset) / totals[dev]
                    - train_fraction)
                for dev in per_device)
            if best_cost is None or cost < best_cost or (
                    cost == best_cost and subset < best_subset):
                best_cost, best_subset = cost, subset

    train_participants = set(best_subset)
    return train_participants, set(participants) - train_participants


def _assert_participants_assigned(usable_dirs):
    """
    Every usable participant on disk must appear in the frozen split.

    A participant that appears in neither set would otherwise be dropped from
    training and evaluation alike without a word; the exact silent-data-loss
    failure the freeze exists to prevent. Raising here costs a job submission;
    not raising costs a wrong result that looks fine.
    """
    assigned = TRAIN_MIXED_PARTICIPANTS | TEST_MIXED_PARTICIPANTS
    found = {participant_id_of(d) for d in usable_dirs}
    unassigned = sorted(found - assigned)
    if unassigned:
        raise ValueError(
            f"HOT3D participants {unassigned} have usable ground truth but are "
            f"in neither TRAIN_MIXED_PARTICIPANTS nor TEST_MIXED_PARTICIPANTS. "
            f"The catalogue has changed since the split was frozen -- re-derive "
            f"it with 'python py/training/common/hot3d_split.py <dataset_root>' "
            f"and update both constants in {__file__}.")


# --- ARCHIVED DESIGN: Aria-only training, cross-device evaluation ----------
# Superseded; kept unchanged so its checkpoints and results stay scorable.

# Recorded on both devices, which held subjects constant while varying the device.
CROSS_DEVICE_HELD_OUT_PARTICIPANTS = {"P0002", "P0003", "P0010"}


def _archived_split(usable, split):
    """The archived cross-device splits. See the block comment above."""
    held_out = [d for d in usable
                if participant_id_of(d) in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    train_pool = [d for d in usable
                  if participant_id_of(d) not in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]

    # Aria-only by design; mixing Quest in would test new subjects, not a new device.
    if split == "train":
        return [d for d in train_pool if headset_of(d) == "Aria"]

    # Same held-out participants split by device: test_aria seen device, test_quest unseen.
    if split == "test_aria":
        return [d for d in held_out if headset_of(d) == "Aria"]
    if split == "test_quest":
        return [d for d in held_out if headset_of(d) == "Quest3"]

    # Quest recordings of TRAINING participants: changes only the device, not the subjects.
    if split == "device_shift_quest":
        return [d for d in train_pool if headset_of(d) == "Quest3"]

    # Both devices' held-out sequences together.
    if split == "cross_device_test":
        return held_out

    return None


# --- Public interface ------------------------------------------------------

CURRENT_SPLITS = ("train_mixed", "test_mixed")
ARCHIVED_SPLITS = ("train", "test_aria", "test_quest", "device_shift_quest",
                   "cross_device_test")


def filter_sequence_dirs(all_seq_dirs, split):
    usable = usable_sequence_dirs(all_seq_dirs)

    if split == "all_labeled":
        return usable

    if split in CURRENT_SPLITS:
        _assert_participants_assigned(usable)
        target = (TRAIN_MIXED_PARTICIPANTS if split == "train_mixed"
                  else TEST_MIXED_PARTICIPANTS)
        return [d for d in usable if participant_id_of(d) in target]

    if split in ARCHIVED_SPLITS:
        return _archived_split(usable, split)

    raise ValueError(
        f"unknown split {split!r}; current: {CURRENT_SPLITS}, "
        f"archived: {ARCHIVED_SPLITS}, plus 'all_labeled'")


def split_train_val(seq_dirs, val_fraction=0.1, seed=0):
    """
    Carve a validation subset out of an already-filtered sequence list.

    The split is at sequence level, so no two frames from the same recording
    can land on opposite sides of it (temporal leakage). Participants may
    appear in both train and val, and that is intentional: val exists only to
    monitor convergence and to select checkpoints. Generalisation is measured
    exclusively on the held-out test split, whose participants this function
    never sees.
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


# --- Pre-flight report -----------------------------------------------------

def _device_counts(seq_dirs):
    counts = {}
    for d in seq_dirs:
        h = headset_of(d) or "unknown"
        counts[h] = counts.get(h, 0) + 1
    return counts


def _report(dataset_root):
    """Print the real split sizes for a dataset root. See __main__ below."""
    print(f"HOT3D catalogue at {dataset_root}\n")

    all_dirs = sorted(
        os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("P0"))
    usable = usable_sequence_dirs(all_dirs)
    print(f"  {len(all_dirs)} sequence folders, {len(usable)} with usable "
          f"ground truth, device breakdown {_device_counts(usable)}\n")

    for split in ("all_labeled",) + CURRENT_SPLITS + ARCHIVED_SPLITS:
        dirs = list_sequence_dirs(dataset_root, split)
        participants = sorted({participant_id_of(d) for d in dirs})
        tag = "" if split in CURRENT_SPLITS or split == "all_labeled" else "  (archived)"
        print(f"  {split:20s} {len(dirs):4d} seq  {_device_counts(dirs)}  "
              f"{' '.join(participants)}{tag}")

    # Per-device train share: the number the objective optimises, worth eyeballing.
    train_counts = _device_counts(list_sequence_dirs(dataset_root, "train_mixed"))
    test_counts = _device_counts(list_sequence_dirs(dataset_root, "test_mixed"))
    print("\n  current split, train share by device:")
    for dev in sorted(set(train_counts) | set(test_counts)):
        tr, te = train_counts.get(dev, 0), test_counts.get(dev, 0)
        if tr + te:
            print(f"    {dev:10s} {tr:4d} train / {te:4d} test = "
                  f"{tr / (tr + te):.1%} train")
    tr, te = sum(train_counts.values()), sum(test_counts.values())
    if tr + te:
        print(f"    {'TOTAL':10s} {tr:4d} train / {te:4d} test = "
              f"{tr / (tr + te):.1%} train (target 80.0%)")

    train_dirs, val_dirs = split_train_val(
        list_sequence_dirs(dataset_root, "train_mixed"))
    print(f"\n  train_mixed further carved into {len(train_dirs)} train / "
          f"{len(val_dirs)} val sequences")

    # Does the frozen split still match what this catalogue would derive?
    derived_train, derived_test = _derive_mixed_split(all_dirs)
    if derived_train == TRAIN_MIXED_PARTICIPANTS and \
            derived_test == TEST_MIXED_PARTICIPANTS:
        print("\n  frozen split MATCHES the split derived from this catalogue.")
    else:
        print("\n  WARNING: the frozen split does NOT match this catalogue.")
        print(f"    frozen  train {sorted(TRAIN_MIXED_PARTICIPANTS)}")
        print(f"    derived train {sorted(derived_train)}")
        print(f"    frozen  test  {sorted(TEST_MIXED_PARTICIPANTS)}")
        print(f"    derived test  {sorted(derived_test)}")
        print("    Expected if this copy of the dataset is incomplete -- the")
        print("    frozen constants still govern, which is the point of")
        print("    freezing them. Update them only if the CATALOGUE changed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _report(sys.argv[1])
    else:
        print("usage: python py/training/common/hot3d_split.py <hot3d_dataset_dir>")
        print()
        print("The device-aware report needs real metadata.json files, so it")
        print("cannot run against anything but a real dataset root.")
        print()
        print(f"frozen train participants: {sorted(TRAIN_MIXED_PARTICIPANTS)}")
        print(f"frozen test  participants: {sorted(TEST_MIXED_PARTICIPANTS)}")
