"""
Pulls raw example frames from HOT3D and Phanesim, undecorated, for the "what does
the data look like" figure in Chapter 2. Read-only: runs no model and only reuses
the raw-frame readers. See --help for arguments.
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
# local_config_cluster.py lives in training/detection; extract_phanesim needs it here.
_DETECTION_DIR = os.path.join(_THIS_DIR, "..", "training", "detection")
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)


def _to_disp(gray):
    lo, hi = float(gray.min()), float(gray.max())
    return ((gray - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)


# --- HOT3D ---------------------------------------------------------------------

def extract_hot3d(args):
    import random
    from eval_detnet import Hot3dRawFrameSource, sequence_dirs_for_split
    import preprocess_baseline as pp
    import local_config_cluster as lc
    import py.training.common.hot3d_split as hot3d_split

    dataset_root = args.dataset_root or lc.hot3d_dataset_root
    repo_root = args.hot3d_repo_root or lc.hot3d_repo_root

    all_seq_dirs = sequence_dirs_for_split(dataset_root, args.split)
    if not all_seq_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under {dataset_root}")

    # Hot3dRawFrameSource opens whole .vrs files, so use a couple of sequences per device.
    devices = args.devices
    random.seed(args.seed)
    by_device = {}
    for d in all_seq_dirs:
        dev = hot3d_split.headset_of(d)
        if dev is not None:
            by_device.setdefault(dev, []).append(d)

    chosen_seq_dirs = []
    for dev in devices:
        pool = by_device.get(dev, [])
        if not pool:
            print(f"  WARNING: no sequences found for device {dev!r} in split "
                  f"{args.split!r} (check the spelling against what "
                  f"hot3d_split.headset_of returns for this dataset)")
            continue
        k = min(len(pool), max(1, args.seqs_per_device))
        chosen_seq_dirs.extend(random.sample(pool, k))

    print(f"[extract_dataset_frames] indexing {len(chosen_seq_dirs)} of "
          f"{len(all_seq_dirs)} sequences in the split")
    source_data = Hot3dRawFrameSource(chosen_seq_dirs, repo_root, args.min_visibility_ratio, None)
    n = len(source_data)
    print(f"[extract_dataset_frames] {n} (frame, camera-stream) samples in the "
          f"chosen sequences")

    per_device_saved = {d: 0 for d in devices}
    os.makedirs(args.out_dir, exist_ok=True)

    order = list(range(n))
    random.shuffle(order)
    for idx in order:
        if all(per_device_saved[d] >= args.per_device for d in devices):
            break
        seq_name, headset, stream_id, ts, image, gt_boxes = source_data.get(idx)
        if image is None or headset not in devices:
            continue
        if per_device_saved[headset] >= args.per_device:
            continue
        orientation = pp.DEVICE_ORIENTATION.get(headset, 270)
        upright = pp.rotate_upright(image, orientation)
        disp = _to_disp(upright)
        fname = f"hot3d_{headset}_{seq_name}_idx{idx}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname), disp)
        print(f"  wrote {fname}")
        per_device_saved[headset] += 1

    for d in devices:
        if per_device_saved[d] < args.per_device:
            print(f"  WARNING: only found {per_device_saved[d]}/{args.per_device} "
                  f"frames for device {d!r} among the {args.seqs_per_device} sampled "
                  f"sequences. Rerun with a larger --seqs-per-device, or "
                  f"--min-visibility-ratio 0 to widen the pool within those sequences.")


# --- Phanesim ------------------------------------------------------------------

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
    parser.add_argument("--seqs-per-device", type=int, default=2,
                        help="how many sequences per device to open .vrs files for; "
                             "keep this small, it's the slow part")
    parser.add_argument("--seed", type=int, default=0)
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
