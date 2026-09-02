"""
What fraction of candidate hands in train_mixed / test_mixed have at least one
unusable keypoint?

WHY THIS EXISTS
---------------
Chapter 4 quotes 35.9% (train) / 40.0% (test) for this statistic, but those
numbers were measured on the archived Aria-only split. HOT3DKeypointDataset's
own docstring already flags them as not re-measured on the mixed split. This
script reproduces the exact same definition -- "at least one of a hand's 21
joints failed the projection/visibility check" -- against train_mixed and
test_mixed, so the chapter can cite a real, current number instead of a
leftover one.

DEFINITION
----------
A "candidate hand" is one that cleared the box-visibility gate in Section 4.2
(box2d_hands.csv, hand_index -> slot, visibility >= 0.2) -- this is
n_candidates as HOT3DKeypointDataset already tracks it internally.

Of those candidates, HOT3DKeypointDataset keeps only the ones with at least
one usable joint (self.actual_size of them, each carrying a (21,2) boolean
xy_valid/depth_valid mask in self._valid) and rejects the rest outright
(self.n_rejected_by_projection -- by construction, EVERY joint failed for
these, so they count as "at least one unusable joint" too).

    at_least_one_unusable = n_rejected_by_projection
                           + count(kept hands where not all 21*2 flags are True)
    pct = 100 * at_least_one_unusable / n_candidates

This is not the "no joint at all usable" figure HOT3DKeypointDataset prints
automatically on construction -- that one is stricter (all 42 flags false)
and much smaller. Don't confuse the two.

USAGE
-----
    python py/evaluation/measure_unusable_joint_rate.py

Builds the index for the whole train_mixed pool (235 sequences) and the whole
test_mixed pool (59 sequences), same sequence lists the trainer and evaluator
use. Index construction reads only poses and calibration, not images, so this
is quick even on a cold cache -- and if job 1693478's train_mixed index is
already warm in hot3d_index_cache_dir, that half is close to instant.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_MERCURY_TRAIN_ROOT,
           os.path.join(_MERCURY_TRAIN_ROOT, "py", "training", "keypoint")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import local_config
import py.training.common.hot3d_split as hot3d_split
from HOT3DKeypointDataset import HOT3DKeypointDataset

FRAME_STRIDE = 5  # matches HOT3D_FRAME_STRIDE in kpest_trainer.py


def measure(split):
    seq_dirs = hot3d_split.list_sequence_dirs(local_config.hot3d_dataset_root, split)
    print(f"\n=== {split}: {len(seq_dirs)} sequences ===")

    dataset = HOT3DKeypointDataset(
        sequence_dirs=seq_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root,
        object_library_path=local_config.hot3d_object_library_path,
        frame_stride=FRAME_STRIDE,
        index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
    )

    n_candidates = dataset.n_rejected_by_projection + dataset.actual_size

    # Kept hands (dataset._valid) that still have at least one bad joint --
    # rejected hands (dataset.n_rejected_by_projection) are ALL bad by
    # construction, so they're added in separately, not double-counted.
    per_hand_all_valid = dataset._valid.all(axis=(1, 2))
    n_kept_with_some_invalid = int((~per_hand_all_valid).sum())

    n_at_least_one_unusable = dataset.n_rejected_by_projection + n_kept_with_some_invalid
    pct = 100.0 * n_at_least_one_unusable / n_candidates if n_candidates else float("nan")

    print(f"candidate hands              : {n_candidates}")
    print(f"  fully rejected (0 usable)  : {dataset.n_rejected_by_projection}")
    print(f"  kept but >=1 unusable joint: {n_kept_with_some_invalid}")
    print(f"  kept, all 21 joints usable : {int(per_hand_all_valid.sum())}")
    print(f">>> at least one unusable joint: {n_at_least_one_unusable}/{n_candidates} "
          f"= {pct:.1f}%")
    return pct


def main():
    train_pct = measure("train_mixed")
    test_pct = measure("test_mixed")
    print(f"\n=== summary (paste into §4.3) ===")
    print(f"train_mixed: {train_pct:.1f}%")
    print(f"test_mixed : {test_pct:.1f}%")


if __name__ == "__main__":
    main()
