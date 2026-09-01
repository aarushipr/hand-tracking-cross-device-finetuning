"""
The regression test for the input convention. Run it before trusting any
evaluation result, and in CI if there ever is one.

test_train_encode_eval_decode_agree is the one that matters. It takes a box
of known position and size, encodes it with the TRAINING pipeline's own
target encoder (augmentation.imgwithboundingboxes320_to_heatmaps_2hand) and
decodes it with the EVALUATION pipeline's decoder
(preprocess_baseline.decode_detection), and asserts the box comes back.

That property is what was broken: the evaluator's decode came from
hg_model.cpp and multiplied `size` by an extra factor of two relative to what
the training encoder writes, so every predicted box was twice as wide as
intended. It produced no error, only a quietly halved IoU. Nothing except a
test like this one distinguishes that from a genuinely weak model.

    python py/evaluation/test_convention.py
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..", "..")
for _p in (_ROOT, _THIS_DIR, os.path.join(_ROOT, "py", "training", "detection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocess_baseline as pp


def test_self_check():
    pp.self_check(verbose=True)


def test_train_encode_eval_decode_agree():
    import augmentation
    from a_structs import ImageWithBoundingBoxes, bbox as Bbox

    s = pp.DETECTION_INPUT_SIZE
    go = pp.compute_blackbar_transform(s, s, s, s)   # identity letterbox

    for cx, cy, side in [(80.0, 80.0, 40.0), (50.0, 110.0, 25.0), (110.0, 60.0, 70.0)]:
        thing = ImageWithBoundingBoxes(image=np.zeros((s, s), np.uint8),
                                       bboxes=[Bbox(cx, cy, side, side), None])
        target = augmentation.imgwithboundingboxes320_to_heatmaps_2hand(thing)
        _conf, box = pp.decode_detection([1.0, 0.0], target["center_x"],
                                         target["center_y"], target["size"], go, 0)
        got_cx, got_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        got_side = box[2] - box[0]
        assert abs(got_cx - cx) < 0.6 and abs(got_cy - cy) < 0.6, \
            f"centre drifted: put in ({cx}, {cy}), got ({got_cx:.2f}, {got_cy:.2f})"
        assert abs(got_side - side) < 0.6, (
            f"the training encoder and the evaluation decoder disagree on `size`: "
            f"encoded a {side:.1f}px box, decoded {got_side:.1f}px "
            f"(ratio {got_side / side:.3f}). A ratio near 2.0 means "
            f"SIZE_DECODE_FACTOR has been reverted to the hg_model.cpp value.")
        print(f"  box c=({cx:5.1f},{cy:5.1f}) side={side:5.1f} survives "
              f"encode->decode  ok")


def test_keypoint_crop_is_deterministic():
    rng = np.random.default_rng(1234)
    kps = rng.uniform(80.0, 420.0, (21, 2))
    mats = [pp.keynet_crop_matrix(kps, is_right=False) for _ in range(5)]
    for m in mats[1:]:
        assert np.allclose(mats[0], m), \
            "the evaluation crop is not deterministic; two models would be " \
            "scored on different pixels"
    print("  keypoint evaluation crop is deterministic across repeats  ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall convention tests passed")
