"""
dump_qualitative_examples.py -- render ground-truth-vs-prediction overlay
images for DetNet or KeyNet, for the thesis's qualitative figures.

Does NOT modify eval_detnet.py or eval_keynet.py. It only imports their
public model-loading and decoding functions, so the numbers already
reported in Tables 5.1-5.5 are untouched by anything this script does.

SELECTION MODES
----------------
--frame-indices i j k ...   render exactly these dataset indices. Use the
                             SAME indices across three separate runs (one
                             per --weights condition) to get a same-frame,
                             cross-condition comparison figure.
--rank-worst N               score every frame in the split, then render
                             the N worst-error frames (failure cases).
--rank-best N                same, but the N best frames.

Exactly one of --frame-indices / --rank-worst / --rank-best is required per
invocation (rank-worst and rank-best may be combined in one run).

USAGE (DetNet, same frames across three conditions):
    python py/evaluation/dump_qualitative_examples.py --model detnet \
        --weights monado --split test_mixed --tag baseline \
        --frame-indices 120 900 4400 --out-dir qual/detnet

    python py/evaluation/dump_qualitative_examples.py --model detnet \
        --weights py/training/detection/checkpoints/CKPT.pth --split test_mixed \
        --tag phanesim --frame-indices 120 900 4400 --out-dir qual/detnet

USAGE (KeyNet failure cases):
    python py/evaluation/dump_qualitative_examples.py --model keynet \
        --weights py/training/keypoint/checkpoints/CKPT.pth --split test_mixed \
        --tag phanesim --rank-worst 4 --out-dir qual/keynet_failures
"""
import argparse
import os
import sys

import numpy as np
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _to_disp(gray):
    """float/uint array (H,W) -> uint8 BGR, safe on a constant frame."""
    lo, hi = float(gray.min()), float(gray.max())
    disp = ((gray - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)


def _save(img, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, img)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# DetNet
# ---------------------------------------------------------------------------

def render_detnet(args):
    from eval_detnet import (build_model, Hot3dRawFrameSource, sequence_dirs_for_split,
                             _MERCURY_TRAIN_ROOT)
    import preprocess_baseline as pp
    import local_config_cluster as lc

    dataset_root = args.dataset_root or lc.hot3d_dataset_root
    repo_root = args.hot3d_repo_root or lc.hot3d_repo_root
    # Reuses eval_detnet.py's own _MERCURY_TRAIN_ROOT and exact default formula
    # rather than re-deriving the relative path here, since a hand-counted
    # "../../.." was one level short and pointed at the wrong directory.
    models_dir = args.models_dir or os.path.join(_MERCURY_TRAIN_ROOT, "..", "..", "hand-tracking-models")

    seq_dirs = sequence_dirs_for_split(dataset_root, args.split)
    if not seq_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under {dataset_root}")
    if args.sequences:
        wanted = set(args.sequences)
        seq_dirs = [d for d in seq_dirs if os.path.basename(os.path.normpath(d)) in wanted]
        if not seq_dirs:
            raise SystemExit(f"None of {args.sequences!r} found in split {args.split!r}")
        print(f"[dump_qualitative] restricted to {len(seq_dirs)} sequence(s): "
              f"{[os.path.basename(os.path.normpath(d)) for d in seq_dirs]}")
    source_data = Hot3dRawFrameSource(seq_dirs, repo_root, args.min_visibility_ratio, None)
    print(f"[dump_qualitative] {len(source_data)} (frame, camera-stream) samples")

    model, source = build_model(args.weights, models_dir)
    print(f"[dump_qualitative] weights: {source}")

    def predict(idx):
        seq_name, headset, stream_id, ts, image, gt_boxes = source_data.get(idx)
        if image is None:
            return None
        orientation = pp.DEVICE_ORIENTATION.get(headset, 270)
        net_input, upright, go = pp.detnet_preprocess(image, orientation)
        if net_input is None:
            return None
        pred = model(net_input)
        h, w = image.shape[:2]
        for gt in gt_boxes:
            x0, y0, x1, y1 = gt["box"]
            corners = pp.rotate_points_upright([[x0, y0], [x1, y1]], orientation, w, h)
            gt["box_upright"] = (float(min(corners[:, 0])), float(min(corners[:, 1])),
                                  float(max(corners[:, 0])), float(max(corners[:, 1])))
        slots = []
        for slot in (0, 1):
            conf, box = pp.decode_detection(pred["hand_exists"], pred["cx"], pred["cy"],
                                             pred["size"], go, slot)
            gt = next((g for g in gt_boxes if g["slot"] == slot), None)
            gt_box = gt["box_upright"] if gt is not None else None
            iou = pp.box_iou(box, gt_box) if gt_box is not None else None
            slots.append({"pred_box": box, "pred_conf": conf, "gt_box": gt_box, "iou": iou})
        return {"seq_name": seq_name, "device": headset, "upright": upright, "slots": slots}

    def draw_and_save(idx, tag):
        r = predict(idx)
        if r is None:
            print(f"  idx {idx}: unusable frame, skipped")
            return
        img = _to_disp(r["upright"])
        for s in r["slots"]:
            if s["gt_box"] is not None:
                x0, y0, x1, y1 = [int(round(v)) for v in s["gt_box"]]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)   # GT: green
            x0, y0, x1, y1 = [int(round(v)) for v in s["pred_box"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)      # pred: red
            label = f"conf={s['pred_conf']:.2f}"
            if s["iou"] is not None:
                label += f" IoU={s['iou']:.2f}"
            cv2.putText(img, label, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 255), 1, cv2.LINE_AA)
        fname = f"{args.tag}_{tag}_{r['seq_name']}_{r['device']}_idx{idx}.png"
        _save(img, args.out_dir, fname)

    if args.frame_indices:
        for idx in args.frame_indices:
            draw_and_save(idx, "frame")
        return

    n = len(source_data) if args.limit is None else min(args.limit, len(source_data))
    scored = []
    for idx in range(n):
        r = predict(idx)
        if r is None:
            continue
        present_ious = [s["iou"] for s in r["slots"] if s["iou"] is not None]
        if present_ious:
            scored.append((idx, min(present_ious)))
        if idx % 500 == 0:
            print(f"  scoring {idx}/{n}", flush=True)

    scored.sort(key=lambda t: t[1])
    if args.rank_worst:
        print(f"[dump_qualitative] rendering {args.rank_worst} worst frames")
        for idx, iou in scored[:args.rank_worst]:
            draw_and_save(idx, "worst")
    if args.rank_best:
        print(f"[dump_qualitative] rendering {args.rank_best} best frames")
        for idx, iou in scored[-args.rank_best:][::-1]:
            draw_and_save(idx, "best")


# ---------------------------------------------------------------------------
# KeyNet
# ---------------------------------------------------------------------------

SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4),
           (0, 5), (5, 6), (6, 7), (7, 8),
           (0, 9), (9, 10), (10, 11), (11, 12),
           (0, 13), (13, 14), (14, 15), (15, 16),
           (0, 17), (17, 18), (18, 19), (19, 20)]


def render_keynet(args):
    import torch
    from eval_keynet import (build_model, sequence_dirs_for_split, decode_xy, CROP_SIDE_PX)
    import local_config
    from HOT3DKeypointDataset import HOT3DKeypointDataset

    sequence_dirs = sequence_dirs_for_split(args.split)
    if not sequence_dirs:
        raise SystemExit(f"No sequences for split {args.split!r} under "
                         f"{local_config.hot3d_dataset_root}")
    dataset = HOT3DKeypointDataset(
        sequence_dirs=sequence_dirs,
        hot3d_repo_root=local_config.hot3d_repo_root,
        object_library_path=local_config.hot3d_object_library_path,
        frame_stride=args.frame_stride,
        index_cache_dir=getattr(local_config, "hot3d_index_cache_dir", None),
        eval_mode=True,   # deterministic crop, same convention as Tables 5.4/5.5
    )
    print(f"[dump_qualitative] {len(dataset)} keypoint samples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, source = build_model(args.weights, device)
    print(f"[dump_qualitative] weights: {source}")

    def predict(idx):
        doct = dataset[idx]
        image = doct["input_image"].unsqueeze(0).to(device)
        pred_kp = torch.zeros_like(doct["input_predicted_keypoints"]).unsqueeze(0).to(device)
        pred_valid = torch.zeros(1, device=device).float()
        with torch.no_grad():
            model_xy, _model_depth, _extras, _curls = model(
                image, torch.flatten(pred_kp, start_dim=1), pred_valid)
        pred_xy = decode_xy(model_xy).squeeze(0).cpu().numpy()          # (21,2) crop px
        gt_xy = doct["gt_joint_locs"][:, :2].cpu().numpy()              # (21,2) crop px
        # dataset[idx] returns some fields as plain numpy arrays that only
        # become tensors once a DataLoader's default_collate touches them
        # (which is how eval_keynet.py's own evaluate() gets away with
        # .to(device).bool() on this same field) -- indexing the dataset
        # directly here skips that, so xy_valid_per_joint arrives as a
        # numpy array with no .bool() method. np.asarray().astype(bool)
        # works whichever type it actually is.
        valid = np.asarray(doct["xy_valid_per_joint"]).astype(bool)      # (21,)
        crop_img = doct["input_image"].squeeze(0).cpu().numpy()          # (128,128)
        return crop_img, gt_xy, pred_xy, valid

    def draw_and_save(idx, tag):
        crop_img, gt_xy, pred_xy, valid = predict(idx)
        img = _to_disp(crop_img)
        err = np.linalg.norm(pred_xy - gt_xy, axis=-1)

        def draw(xy, color):
            pts = [(int(round(x)), int(round(y))) for x, y in xy]
            for a, b in SKELETON:
                if valid[a] and valid[b]:
                    cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
            for j, (x, y) in enumerate(pts):
                if valid[j]:
                    cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)

        draw(gt_xy, (0, 200, 0))     # GT: green
        draw(pred_xy, (0, 0, 255))   # pred: red
        mean_err = float(err[valid].mean()) if valid.any() else float("nan")
        cv2.putText(img, f"mean err={mean_err:.1f}px", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (0, 0, 255), 1, cv2.LINE_AA)
        fname = f"{args.tag}_{tag}_idx{idx}.png"
        _save(img, args.out_dir, fname)
        return mean_err

    if args.frame_indices:
        for idx in args.frame_indices:
            draw_and_save(idx, "frame")
        return

    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    scored = []
    for idx in range(n):
        crop_img, gt_xy, pred_xy, valid = predict(idx)
        if valid.sum() < args.min_valid_joints:
            continue
        err = np.linalg.norm(pred_xy - gt_xy, axis=-1)
        scored.append((idx, float(err[valid].mean())))
        if idx % 200 == 0:
            print(f"  scoring {idx}/{n}", flush=True)

    # Ascending by pixel error, so index 0 is the LOWEST error (best), unlike
    # DetNet's IoU-based ranking above where ascending-first is worst -- error
    # and IoU point opposite directions, so this can't reuse that slicing.
    scored.sort(key=lambda t: t[1])
    if args.rank_worst:
        print(f"[dump_qualitative] rendering {args.rank_worst} worst samples")
        for idx, e in scored[-args.rank_worst:][::-1]:
            draw_and_save(idx, "worst")
    if args.rank_best:
        print(f"[dump_qualitative] rendering {args.rank_best} best samples")
        for idx, e in scored[:args.rank_best]:
            draw_and_save(idx, "best")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["detnet", "keynet"])
    parser.add_argument("--weights", required=True,
                        help="'monado' for the shipped zero-shot weights, or a .pth path")
    parser.add_argument("--split", required=True)
    parser.add_argument("--tag", required=True,
                        help="short label for this condition, used as a filename prefix "
                             "(e.g. baseline / phanesim / hot3d)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None)
    parser.add_argument("--rank-worst", type=int, default=None)
    parser.add_argument("--rank-best", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the ranking pass to the first N samples, for a smoke test")
    # DetNet-only
    parser.add_argument("--min-visibility-ratio", type=float, default=0.2)
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="restrict to these sequence names (e.g. P0001_10a27bf7), "
                             "skipping the VRS-open cost for the rest of the split")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--hot3d-repo-root", default=None)
    # KeyNet-only
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-valid-joints", type=int, default=15,
                        help="skip frames with fewer than this many visible GT joints "
                             "when ranking (avoids near-empty crops dominating worst/best)")
    args = parser.parse_args()

    if not (args.frame_indices or args.rank_worst or args.rank_best):
        raise SystemExit("Give at least one of --frame-indices / --rank-worst / --rank-best")

    if args.model == "detnet":
        render_detnet(args)
    else:
        render_keynet(args)


if __name__ == "__main__":
    main()
