"""
extract_dataset_frames.py -- pull raw example frames from the HOT3D and
Phanesim datasets, undecorated (no boxes, no model), for a "what does the
data look like" figure in Chapter 2.

Read-only. Runs no model and does not touch eval_detnet.py, eval_keynet.py,
or any training/eval code path -- it only reuses their raw-frame readers.

USAGE:
    python py/evaluation/extract_dataset_frames.py --source hot3d \
        --split test_mixed --per-device 3 --out-dir qual/ch2_hot3d

    python py/evaluation/extract_dataset_frames.py --source phanesim \
        --num-frames 3 --out-dir qual/ch2_phanesim
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _to_disp(gray):
    lo, hi = float(gray.min()), float(gray.max())
    return ((gray - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HOT3D
# ---------------------------------------------------------------------------

def extract_hot3d(args):
    from eval_detnet import Hot3dRawFrameSource, sequence_dirs_for_split
    import preprocess_baseline as pp
    import local_config_cluster as lc

    dataset_root = args.dataset_root or lc.hot3d_dataset_root
    repo_root = args.hot3d_repo_root or lc.hot3d_repo_root

    seq_dirs = sequence_dirs_for_split(dataset_root, args.split)
    if not seq_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under {dataset_root}")
    source_data = Hot3dRawFrameSource(seq_dirs, repo_root, args.min_visibility_ratio, None)
    n = len(source_data)
    print(f"[extract_dataset_frames] {n} (frame, camera-stream) samples")

    devices = args.devices
    per_device_seen_seqs = {d: set() for d in devices}
    per_device_saved = {d: 0 for d in devices}
    os.makedirs(args.out_dir, exist_ok=True)

    # Evenly-spaced stride rather than a scan from idx 0, and at most one
    # saved frame per (device, sequence), so the examples come from
    # different clips instead of all being consecutive frames of one.
    stride = max(1, n // max(1, args.per_device * len(devices) * 40))
    for idx in range(0, n, stride):
        if all(per_device_saved[d] >= args.per_device for d in devices):
            break
        seq_name, headset, stream_id, ts, image, gt_boxes = source_data.get(idx)
        if image is None or headset not in devices:
            continue
        if per_device_saved[headset] >= args.per_device:
            continue
        if seq_name in per_device_seen_seqs[headset]:
            continue
        orientation = pp.DEVICE_ORIENTATION.get(headset, 270)
        upright = pp.rotate_upright(image, orientation)
        disp = _to_disp(upright)
        fname = f"hot3d_{headset}_{seq_name}_idx{idx}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), disp)
        print(f"  wrote {fname}")
        per_device_seen_seqs[headset].add(seq_name)
        per_device_saved[headset] += 1

    for d in devices:
        if per_device_saved[d] < args.per_device:
            print(f"  WARNING: only found {per_device_saved[d]}/{args.per_device} "
                  f"frames for device {d!r} at stride {stride}. Either that device "
                  f"name doesn't appear in this split (check spelling against the "
                  f"headset values eval_detnet.py reports), or a smaller stride is "
                  f"needed -- rerun with --min-visibility-ratio 0 to widen the pool.")


# ---------------------------------------------------------------------------
# Phanesim
# ---------------------------------------------------------------------------

def discover_phanesim_clips(roots):
    clip_dirs = []
    for root in roots:
        for clip_dir in sorted(glob.glob(os.path.join(root, "clip_*"))):
            done_path = os.path.join(clip_dir, "_done.json")
            rect_path = os.path.join(clip_dir, "cam_head0", "hand_rect.csv")
            if os.path.exists(done_path) and os.path.exists(rect_path):
                clip_dirs.append(clip_dir)
    return clip_dirs


def extract_phanesim(args):
    import local_config_cluster as lc

    roots = args.phanesim_roots or lc.phanesim_dataset_roots
    clip_dirs = discover_phanesim_clips(roots)
    if not clip_dirs:
        raise SystemExit(f"No usable Phanesim clips found under {roots}")
    print(f"[extract_dataset_frames] {len(clip_dirs)} usable Phanesim clips")

    os.makedirs(args.out_dir, exist_ok=True)
    idxs = np.linspace(0, len(clip_dirs) - 1, args.num_frames, dtype=int)
    saved = 0
    for i in idxs:
        clip_dir = clip_dirs[int(i)]
        img_path = os.path.join(clip_dir, "cam_head0", f"frame_{args.frame_index:06d}.png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  skipping {img_path}: not found or unreadable")
            continue
        clip_name = os.path.basename(clip_dir)
        root_name = os.path.basename(os.path.dirname(clip_dir))
        fname = f"phanesim_{root_name}_{clip_name}_frame{args.frame_index:06d}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), image)
        print(f"  wrote {fname}")
        saved += 1
    if saved < args.num_frames:
        print(f"  WARNING: only saved {saved}/{args.num_frames} frames; try a "
              f"different --frame-index if frame 0 is missing in some clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--source", required=True, choices=["hot3d", "phanesim"])
    parser.add_argument("--out-dir", required=True)
    # HOT3D
    parser.add_argument("--split", default="test_mixed")
    parser.add_argument("--per-device", type=int, default=3)
    parser.add_argument("--devices", nargs="+", default=["Aria", "Quest3"])
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--hot3d-repo-root", default=None)
    # Phanesim
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--phanesim-roots", nargs="+", default=None)
    args = parser.parse_args()

    if args.source == "hot3d":
        extract_hot3d(args)
    else:
        extract_phanesim(args)


if __name__ == "__main__":
    main()
