"""
What fraction of candidate hands have at least one unusable keypoint? Chapter 4
quotes 35.9% train / 40.0% test from the archived Aria-only split; this reproduces
the same definition against the mixed split. Hands rejected outright by the
projection gate count too, since every joint failed for those.
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

    # Rejected hands are all-bad by construction, so they're added separately, not twice.
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
