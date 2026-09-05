"""
What fraction of DetNet's candidate frames in train_mixed / test_mixed
contain no hand at all?

WHY THIS EXISTS
---------------
Your supervisor asked whether the detection dataset includes negative
frames (no hand present) so DetNet learns "sometimes there's nothing
here", and if so what fraction that is. HOT3DVRSDetectionDataset's own
docstring confirms these negatives exist by construction -- the sample
index walks every timestamp of every camera stream, not just the
timestamps present in box2d_hands.csv -- but the actual percentage isn't
computed or logged anywhere. This script measures it directly.

DEFINITION
----------
Same sampling this project's own DetNet trainer uses: train_mixed /
test_mixed sequences, frame_stride=5 (HOT3D_FRAME_STRIDE in
trainer_detection.py), min_visibility_ratio=0.2 (the class default,
which trainer_detection.py doesn't override). For each sampled frame,
a hand counts as present if box2d_hands.csv has a non-null box with a
non-null visibility_ratio >= 0.2 for that hand at that timestamp --
exactly the check HOT3DVRSDetectionDataset.__getitem__ applies before
setting exists[slot]=1. A frame is a "no hand" frame when neither hand
clears that bar.

WHY THIS IS FAST DESPITE HOT3D'S I/O BEING SLOW
------------------------------------------------
Training is I/O-bound because every sample decodes a full VRS image
frame over network storage. This script never calls get_image() --
it only resolves stream IDs and queries box2d_hands.csv-backed
providers, both of which are small in-memory lookups once a
sequence's providers are open. Opening those providers still touches
each sequence's .vrs once (to list stream IDs), so this isn't free,
but it's a per-sequence cost (~294 sequences total) instead of a
per-frame one.

USAGE
-----
    python py/evaluation/measure_hand_presence_rate.py

Builds the same frame_stride=5 index the real DetNet trainer used
(reusing its on-disk cache at hot3d_index_cache_dir if warm), then
scans train_mixed's and test_mixed's sequence pools.
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
