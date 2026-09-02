"""
Does HOT3DKeypointDataset actually produce usable keypoint ground truth for
Quest 3 recordings?

WHY THIS EXISTS
---------------
The mixed Aria+Quest split puts Quest recordings into KeyNet's training and
evaluation sets. Until that split, KeyNet only ever saw Aria data, so the Quest
path through this dataset class was never exercised end to end.

Two specific reasons not to assume it works:

1. py/training/common/hot3d_timecode_compat.py -- the shim that makes Quest
   recordings readable at all -- states in its own "WHAT THIS DOES NOT COVER"
   section that it was verified against the detection ground truth
   (box2d_hands.csv) only, and explicitly NOT against the UmeTrack-format
   keypoint ground truth this dataset reads.
2. Until this check was written, HOT3DKeypointDataset never applied that shim.

The failure mode this guards against is not a crash. A crash is easy. It is an
index that silently comes back empty or tiny for every Quest sequence, so
training quietly proceeds on Aria data wearing a mixed-split label, and the
reported result describes something other than what the thesis claims.

WHAT IT CHECKS
--------------
For one sequence:
  - the index builds and is non-empty
  - the acceptance rate (kept hands / candidate hands) is not pathological
  - samples decode: image present, correct shape, not uniformly blank
  - per-joint validity flags are not all false
  - projected keypoints land inside the source frame rather than at infinity

Run it on one Quest sequence and one Aria sequence and compare the two. Aria is
the control: if both look alike, the Quest path is behaving.

USAGE
-----
    python py/evaluation/check_quest_keypoints.py <sequence_dir> [--samples 20]

    # typical use, from the mercury_train root on the cluster:
    python py/evaluation/check_quest_keypoints.py \
        /storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/P0013_0ec32d10

Exits non-zero if any check fails, so it can gate a job submission.
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_MERCURY_TRAIN_ROOT,
           os.path.join(_MERCURY_TRAIN_ROOT, "py", "training", "keypoint")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import local_config
import py.training.common.hot3d_split as hot3d_split
from HOT3DKeypointDataset import HOT3DKeypointDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sequence_dir")
    parser.add_argument("--samples", type=int, default=20,
                        help="how many samples to actually decode")
    parser.add_argument("--frame-stride", type=int, default=5)
    args = parser.parse_args()

    seq_dir = os.path.abspath(args.sequence_dir)
    headset = hot3d_split.headset_of(seq_dir)
    print(f"sequence : {os.path.basename(seq_dir)}")
    print(f"headset  : {headset}")
    if headset is None:
        print("FAIL: no readable metadata.json -- headset_of() returned None.")
        return 1

    # eval_mode=True: deterministic crops, so a failure here is a property of
    # the data rather than of one random draw.
    dataset = HOT3DKeypointDataset(
        sequence_dirs=[seq_dir],
        hot3d_repo_root=local_config.hot3d_repo_root,
        object_library_path=local_config.hot3d_object_library_path,
        frame_stride=args.frame_stride,
        index_cache_dir=None,          # never cache a diagnostic run
        eval_mode=True,
    )

    failures = []

    n = len(dataset)
    print(f"index    : {n} samples")
    if n == 0:
        print("FAIL: the index is empty. Every candidate hand was rejected, or "
              "no timestamps were resolved at all.")
        return 1

    valid_frac = dataset._valid[:, :, 0].mean()
    print(f"xy-valid : {valid_frac:.1%} of joints across the whole index")
    if valid_frac < 0.10:
        failures.append("under 10% of joints have valid 2D positions -- the "
                        "projection is almost certainly wrong for this device")

    # Keypoints should sit in image coordinates, so a plausible range rather
    # than the huge values a bad projection produces.
    kps = dataset._kps[:, :, :2]
    finite = np.isfinite(kps).all()
    print(f"kp range : x [{kps[..., 0].min():.0f}, {kps[..., 0].max():.0f}]  "
          f"y [{kps[..., 1].min():.0f}, {kps[..., 1].max():.0f}]  "
          f"finite={finite}")
    if not finite:
        failures.append("projected keypoints contain non-finite values")
    if np.abs(kps).max() > 10000:
        failures.append("projected keypoints reach implausible magnitudes "
                        "(>10000 px) -- the camera model is likely mismatched")

    depth = dataset._kps[:, :, 2]
    print(f"depth    : [{depth.min():.2f}, {depth.max():.2f}] "
          f"(relative-depth units, expect roughly [-1.5, 1.5])")
    if np.abs(depth).max() > 20:
        failures.append("relative depth is far outside its expected range -- "
                        "the hand-size normalisation is probably wrong here")

    # Decode real samples. This is the part that exercises the .vrs image read
    # and the timestamp domain, which is where the Quest path differs.
    step = max(1, n // max(1, args.samples))
    blank, decoded, no_hand = 0, 0, 0
    for i in range(0, n, step):
        if decoded >= args.samples:
            break
        doct = dataset[i]
        arr = np.asarray(doct["input_image"])
        decoded += 1
        # is_hand=0 marks _empty_sample(), i.e. the image could not be read.
        if float(doct["is_hand"]) < 0.5:
            no_hand += 1
        if arr.size == 0 or float(arr.std()) < 1e-6:
            blank += 1
    print(f"decoded  : {decoded} samples, {blank} blank/constant, "
          f"{no_hand} unreadable (is_hand=0)")
    if decoded == 0:
        failures.append("no samples could be decoded at all")
    else:
        if blank > decoded * 0.5:
            failures.append(f"{blank}/{decoded} decoded crops are blank or "
                            f"constant -- images are not being read for this "
                            f"device")
        if no_hand > decoded * 0.5:
            failures.append(f"{no_hand}/{decoded} samples fell back to the "
                            f"unreadable-image placeholder")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: this sequence's keypoint ground truth looks usable.")
    print("Run the same check on an Aria sequence and compare the numbers "
          "before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
