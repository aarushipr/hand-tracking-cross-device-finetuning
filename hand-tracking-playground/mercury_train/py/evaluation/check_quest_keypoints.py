"""
Does HOT3DKeypointDataset produce usable Quest 3 keypoint ground truth? The mixed
split put Quest into KeyNet's sets for the first time, and the TimeCode shim was
only ever verified against the detection ground truth. The failure mode is a
silently empty index, not a crash, so run it on one Quest and one Aria sequence
and compare. Exits non-zero, so it can gate a job submission.
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

    # eval_mode=True: deterministic crops, so a failure is the data, not one draw.
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

    # Keypoints should sit in image coordinates, not the huge values a bad projection gives.
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

    # Decoding real samples is what exercises the .vrs read and the timestamp domain.
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
