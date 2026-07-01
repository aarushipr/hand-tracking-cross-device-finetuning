"""
Standalone round-trip test for convert_umetrack_to_rando_csv.py's CSV format.

This does NOT require the UmeTrack dataset, hand_tracking_toolkit, Blender,
or the C++ build -- it only checks that a CSV written by
`build_rando_csv_row` is byte-for-byte compatible with the exact parsing
logic in RandoData.py's RandoDataset.__getitem__ (copied here verbatim so
this test has no import-time dependency on the rest of the training code).

Run: python verify_umetrack_csv_format.py
"""

import numpy as np
import pandas as pd

from convert_umetrack_to_rando_csv import (
    build_rando_csv_row,
    CSV_HEADER,
    NUM_PROJECT_KEYPOINTS,
)


def parse_row_like_rando_dataset(b: pd.Series):
    """Copied from RandoData.py's RandoDataset.__getitem__ parsing logic."""
    acc_idx = 0

    filename = b[acc_idx]
    acc_idx += 1

    kps = np.zeros((22, 3))
    for i in range(22):
        for j in range(3):
            kps[i][j] = b[acc_idx]
            acc_idx += 1

    gt_xy_valid = np.zeros((22))
    gt_depth_valid = np.zeros((22))
    for i in range(22):
        gt_xy_valid[i] = b[acc_idx]
        acc_idx += 2
        gt_depth_valid[i] = b[acc_idx]
        acc_idx += 1

    is_right = bool(b[acc_idx])
    acc_idx += 1

    mask_filename = None
    if len(b) == acc_idx + 1:
        mask_filename = b[acc_idx]

    return filename, kps, gt_xy_valid, gt_depth_valid, is_right, mask_filename, acc_idx


def main():
    rng = np.random.default_rng(0)
    kps_a = rng.uniform(0, 128, (NUM_PROJECT_KEYPOINTS, 3))
    kps_a[:, 2] = rng.uniform(0.1, 1.0, NUM_PROJECT_KEYPOINTS)
    kps_b = rng.uniform(0, 128, (NUM_PROJECT_KEYPOINTS, 3))
    kps_b[:, 2] = rng.uniform(0.1, 1.0, NUM_PROJECT_KEYPOINTS)

    rows = [
        build_rando_csv_row("seq0_000001_r_1201-1.jpg", kps_a, True),
        build_rando_csv_row("seq0_000002_l_1201-2.jpg", kps_b, False),
    ]

    csv_text = CSV_HEADER + "\n" + "\n".join(rows) + "\n"
    tmp_path = "/tmp/_verify_umetrack_csv.csv"
    with open(tmp_path, "w") as f:
        f.write(csv_text)

    csvframe = pd.read_csv(tmp_path, delimiter=" ", quotechar="|")

    assert csvframe.shape[0] == 2, f"expected 2 data rows, got {csvframe.shape[0]}"

    expected = [
        ("seq0_000001_r_1201-1.jpg", kps_a, True),
        ("seq0_000002_l_1201-2.jpg", kps_b, False),
    ]

    for row_idx, (exp_filename, exp_kps, exp_is_right) in enumerate(expected):
        b = csvframe.iloc[row_idx]
        filename, kps, gt_xy_valid, gt_depth_valid, is_right, mask_filename, acc_idx = (
            parse_row_like_rando_dataset(b)
        )

        assert filename == exp_filename, f"filename mismatch: {filename!r} != {exp_filename!r}"
        assert np.allclose(kps[:21], exp_kps, atol=1e-4), "keypoints did not round-trip"
        assert np.allclose(kps[21], 0), "padding joint 21 should be zero"
        assert np.all(gt_xy_valid[:21] == 1.0), "expected all real joints marked valid"
        assert gt_xy_valid[21] == 0.0, "padding joint should be marked invalid"
        assert np.all(gt_depth_valid[:21] == 1.0)
        assert is_right == exp_is_right, f"is_right mismatch on row {row_idx}"
        assert mask_filename is None, "did not expect a mask column"
        assert len(b) == acc_idx, (
            f"column count mismatch: row has {len(b)} columns, parser consumed {acc_idx}. "
            "This means RandoDataset would either crash or silently misread the next column "
            "as a mask filename."
        )

    print("OK: convert_umetrack_to_rando_csv.py's CSV format round-trips correctly")
    print(f"    through RandoData.py's exact parsing logic ({csvframe.shape[1]} columns/row).")


if __name__ == "__main__":
    main()
