"""
What fraction of DetNet's candidate frames in train_mixed / test_mixed contain no
hand at all? The negatives exist by construction, since the index walks every
timestamp rather than only those in box2d_hands.csv, but the percentage is logged
nowhere. Same sampling as the trainer: stride 5, visibility >= 0.2.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_MERCURY_TRAIN_ROOT,
           os.path.join(_MERCURY_TRAIN_ROOT, "py", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import local_config
import py.training.common.hot3d_split as hot3d_split
from HOT3DVRSDetectionDataset import HOT3DVRSDetectionDataset

FRAME_STRIDE = 5           # matches HOT3D_FRAME_STRIDE in trainer_detection.py
MIN_VISIBILITY_RATIO = 0.2  # matches the class default trainer_detection.py relies on


def measure(split):
    seq_dirs = hot3d_split.list_sequence_dirs(local_config.hot3d_dataset_root, split)
    print(f"\n=== {split}: {len(seq_dirs)} sequences ===")

    ds = HOT3DVRSDetectionDataset(
        sequence_dirs=seq_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root,
        min_visibility_ratio=MIN_VISIBILITY_RATIO,
        frame_stride=FRAME_STRIDE,
        index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        augment=False,
    )

    n = len(ds)
    both = left_only = right_only = neither = 0

    for idx in range(n):
        seq_dir = ds.sequence_dirs[int(ds._seq_idx[idx])]
        stream_str = str(ds._stream_str[idx])
        ts = int(ds._ts[idx])

        bundle = ds._providers_for(seq_dir)
        stream_id = bundle.stream_id_by_str.get(stream_str)
        exists = [False, False]

        if stream_id is not None and ds._StreamId(str(stream_id)) in bundle.box2d_provider.stream_ids:
            result = bundle.box2d_provider.get_bbox_at_timestamp(
                stream_id=stream_id,
                timestamp_ns=ts,
                time_query_options=ds._TimeQueryOptions.CLOSEST,
                time_domain=ds._TimeDomain.TIME_CODE,
            )
            if result is not None:
                for hand_index, hand_box in result.box2d_collection.box2ds.items():
                    if hand_box.box2d is None:
                        continue
                    if hand_box.visibility_ratio is None:
                        continue
                    if hand_box.visibility_ratio < MIN_VISIBILITY_RATIO:
                        continue
                    slot = 0 if hand_index == 0 else 1
                    exists[slot] = True

        if exists[0] and exists[1]:
            both += 1
        elif exists[0]:
            left_only += 1
        elif exists[1]:
            right_only += 1
        else:
            neither += 1

        if (idx + 1) % 20000 == 0:
            print(f"  ...{idx + 1}/{n} frames scanned")

    pct_neither = 100.0 * neither / n if n else float("nan")
    print(f"frames scanned          : {n}")
    print(f"  both hands present   : {both} ({100.0 * both / n:.1f}%)")
    print(f"  left only            : {left_only} ({100.0 * left_only / n:.1f}%)")
    print(f"  right only           : {right_only} ({100.0 * right_only / n:.1f}%)")
    print(f"  neither hand present : {neither} ({pct_neither:.1f}%)")
    return pct_neither


def main():
    train_pct = measure("train_mixed")
    test_pct = measure("test_mixed")
    print("\n=== summary ===")
    print(f"train_mixed: {train_pct:.1f}% of frames have no hand at all")
    print(f"test_mixed : {test_pct:.1f}% of frames have no hand at all")


if __name__ == "__main__":
    main()
