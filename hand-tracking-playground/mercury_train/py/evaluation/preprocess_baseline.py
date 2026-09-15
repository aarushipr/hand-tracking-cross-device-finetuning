"""
The baseline input convention, defined exactly once, because every consumer of the
shipped weights must reproduce it or the comparison measures format mismatch.
Where hg_model.cpp and the training code disagree, the training code wins.
Photometric: stddev 0.25 then mean 0.5, not /255. DetNet letterboxes to 160x160
and `size` is a fraction of frame width, one multiply by 160, not the runtime's
extra *2. KeyNet crops 128x128, right hands mirrored, sRGB EOTF applied. HOT3D
frames need rotate_upright(270) first.
"""
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MERCURY_TRAIN_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _MERCURY_TRAIN_ROOT not in sys.path:
    sys.path.insert(0, _MERCURY_TRAIN_ROOT)

from py.training.common.a_geometry import normalize_grayscale_exact

# --- Constants -------------------------------------------------------------
# These are the convention; changing one invalidates every number produced before.

DETECTION_INPUT_SIZE = 160          # detection/header.py model_input_{width,height}
KEYPOINT_CROP_SIZE = 128            # a_aug_config output_size
HEATMAP_SIDE = 22                   # maker_of_augmentations.make_heatmap_output
DEPTH_HALF_RANGE = 1.5              # z expected in [-1.5, 1.5]

# See SIZE_DECODE in the module docstring. 1.0, not 2.0, and measured.
SIZE_DECODE_FACTOR = 1.0

# hg_sync.cpp: mercury_min_detection_confidence, default 0.3, applied with strict >.
MIN_DETECTION_CONFIDENCE = 0.3

# Confirmed per device at all four rotations (dump_orientation_check.py), Aug 2026.
DEVICE_ORIENTATION = {"Aria": 270, "Quest": 270, "Quest3": 270}

# Held at the centre of the training distributions, so a crop depends on the hand only.
CROP_ROTATION_EVAL = 0.0
CROP_RADIUS_SCALE_EVAL = 1.0        # RandoData: uniformcr(1.65, 0.2) / 1.65, mean 1.0


# --- Shared photometric step ---------------------------------------------------

def normalize_grayscale(img_uint8):
    """normalizeGrayscaleImage on a uint8 frame. Returns float32, or None on
    an image of exactly zero variance; the C++ bails out there and so does
    evaluation, which can afford to drop one degenerate frame. Training
    cannot drop a sample mid-batch and substitutes noise instead; that
    divergence is deliberate and lives in a_geometry.normalizeGrayscaleImage.
    """
    return normalize_grayscale_exact(np.asarray(img_uint8, np.float32) / 255.0)


# --- DetNet geometry -----------------------------------------------------------

def rotate_upright(image, orientation):
    """Rotate a raw sensor frame into the upright orientation the weights
    expect. Used by BOTH the detection training dataset and the detection
    evaluator, so the two cannot drift apart.

    orientation is the camera mount rotation in degrees (0/90/180/270), i.e.
    what DEVICE_ORIENTATION reports for the capturing device.
    """
    if orientation == 0:
        return image
    if orientation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"orientation must be 0/90/180/270, got {orientation}")


def rotate_points_upright(points_xy, orientation, in_w, in_h):
    """Apply rotate_upright's coordinate mapping to (N,2) points, so ground
    truth follows the image. in_w/in_h are the ORIGINAL frame's dimensions.
    """
    p = np.asarray(points_xy, np.float64).reshape(-1, 2)
    if orientation == 0:
        return p.copy()
    if orientation == 90:      # ROTATE_90_COUNTERCLOCKWISE
        return np.stack([p[:, 1], (in_w - 1) - p[:, 0]], axis=-1)
    if orientation == 180:
        return np.stack([(in_w - 1) - p[:, 0], (in_h - 1) - p[:, 1]], axis=-1)
    if orientation == 270:     # ROTATE_90_CLOCKWISE
        return np.stack([(in_h - 1) - p[:, 1], p[:, 0]], axis=-1)
    raise ValueError(f"orientation must be 0/90/180/270, got {orientation}")


def compute_blackbar_transform(in_w, in_h, out_w, out_h):
    """Monado `blackbar()`: scale-to-fit plus centre padding, as a 2x3 matrix
    in cv2.warpAffine's convention (original pixel -> letterboxed pixel).

    The camera-mount rotation that the C++ folds into this matrix is NOT
    included here. It is applied separately by rotate_upright() so that the
    training pipeline, which builds its own randomised affine and cannot
    use this matrix, can still apply the identical rotation.
    """
    s = min(out_w / in_w, out_h / in_h)
    go = np.zeros((2, 3), dtype=np.float32)
    go[0] = [s, 0.0, (out_w - in_w * s) / 2.0]
    go[1] = [0.0, s, (out_h - in_h * s) / 2.0]
    return go


def letterbox(image, out_size=DETECTION_INPUT_SIZE):
    """Returns (letterboxed_uint8, go). Pass `image` already upright."""
    h, w = image.shape[:2]
    go = compute_blackbar_transform(w, h, out_size, out_size)
    return cv2.warpAffine(image, go, (out_size, out_size)), go


def detnet_preprocess(raw_image, orientation, out_size=DETECTION_INPUT_SIZE):
    """Raw sensor frame -> (network input float32 [1,1,S,S], upright image, go).

    `go` maps upright-image pixels to letterboxed pixels; invert it to bring
    predictions back. Returns (None, upright, go) on a zero-variance frame.
    """
    upright = rotate_upright(raw_image, orientation)
    lb, go = letterbox(upright, out_size)
    norm = normalize_grayscale(lb)
    if norm is None:
        return None, upright, go
    return norm.reshape(1, 1, out_size, out_size).astype(np.float32), upright, go


def decode_detection(hand_exists, cx, cy, size, go, slot,
                     input_size=DETECTION_INPUT_SIZE):
    """One hand slot's raw network outputs -> (confidence, box in UPRIGHT
    image pixels as (left, top, right, bottom)).

    cx, cy are in [-1, 1] across the letterboxed frame; `size` is the box's
    larger side as a fraction of frame width. See SIZE_DECODE.
    """
    conf = float(np.asarray(hand_exists).reshape(-1)[slot])
    cx_raw = float(np.asarray(cx).reshape(-1)[slot])
    cy_raw = float(np.asarray(cy).reshape(-1)[slot])
    size_raw = float(np.asarray(size).reshape(-1)[slot])

    cx_lb = (cx_raw + 1.0) / 2.0 * input_size
    cy_lb = (cy_raw + 1.0) / 2.0 * input_size

    inv = cv2.invertAffineTransform(go)
    centre = inv @ np.array([cx_lb, cy_lb, 1.0])
    px, py = float(centre[0]), float(centre[1])

    # go's linear part is an isotropic scale; undo it to get original pixels.
    scale_back = 1.0 / float(np.hypot(go[0, 0], go[1, 0]))
    side = size_raw * input_size * SIZE_DECODE_FACTOR * scale_back

    return conf, (px - side / 2, py - side / 2, px + side / 2, py + side / 2)


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
             + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter)
    return inter / union if union > 0 else 0.0


# --- KeyNet geometry -----------------------------------------------------------

def _bounding_square(kps):
    lo = kps.min(axis=0)
    hi = kps.max(axis=0)
    centre = (lo + hi) / 2.0
    return centre, float(np.max(hi - lo))


def _palm_length(kps):
    return float(np.linalg.norm(kps[9][:2] - kps[0][:2]))


def keynet_crop_matrix(keypoints_px, is_right,
                       rotation=CROP_ROTATION_EVAL,
                       radius_scale=CROP_RADIUS_SCALE_EVAL,
                       out_size=KEYPOINT_CROP_SIZE):
    """The 2x3 crop matrix RandoData.crop builds, with its two random draws
    exposed as arguments instead of sampled internally.

    Training passes rotation ~ U(0, 2pi) and radius_scale ~ U(0.81, 1.18),
    which is augmentation and belongs there. Evaluation uses the defaults;
    the centre of both distributions; so that a crop is a deterministic
    function of the hand, and the baseline and the fine-tuned model are
    scored on pixel-identical inputs. With unseeded random crops they are
    not, and the difference between two models is then partly the difference
    between two random draws.

    `is_right` mirrors the crop, so the network only ever sees left-hand
    geometry. That mirroring is not augmentation and is never disabled.
    """
    kps = np.asarray(keypoints_px, np.float64)[:, :2]

    c, s = np.cos(rotation), np.sin(rotation)
    fwd = np.array([[c, -s, 0.0], [s, c, 0.0]])
    back = np.array([[c, s, 0.0], [-s, c, 0.0]])

    rotated = np.stack([fwd[0, 0] * kps[:, 0] + fwd[0, 1] * kps[:, 1],
                        fwd[1, 0] * kps[:, 0] + fwd[1, 1] * kps[:, 1]], axis=-1)

    centre, extent = _bounding_square(rotated)
    radius = max(extent * 1.65, _palm_length(rotated) * 2.2) * radius_scale

    half = radius / 2.0
    down = np.array([0.0, half])
    right = np.array([half, 0.0])
    if is_right:
        right = -right

    corners = np.stack([centre - down - right,
                        centre - down + right,
                        centre + down - right])
    corners = np.stack([back[0, 0] * corners[:, 0] + back[0, 1] * corners[:, 1],
                        back[1, 0] * corners[:, 0] + back[1, 1] * corners[:, 1]], axis=-1)

    dst = np.float32([(0, 0), (out_size, 0), (0, out_size)])
    return cv2.getAffineTransform(np.float32(corners), dst)


# --- Self-check ----------------------------------------------------------------

def self_check(verbose=True):
    """Round-trips known quantities through the transforms above and asserts
    they come back. Exists because the two-times size error produced correct
    box CENTRES and plausible IoU numbers, so nothing in a normal run looked
    wrong. Run it in CI, and before trusting any evaluation result.
    """
    def say(msg):
        if verbose:
            print(msg)

    # 1. A box of known width survives letterbox -> encode -> decode.
    in_w, in_h = 1280, 720
    go = compute_blackbar_transform(in_w, in_h, DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE)
    for true_side in (60.0, 180.0, 420.0):
        cx_o, cy_o = 640.0, 360.0
        p = go @ np.array([cx_o, cy_o, 1.0])
        cx_enc = p[0] / DETECTION_INPUT_SIZE * 2.0 - 1.0
        cy_enc = p[1] / DETECTION_INPUT_SIZE * 2.0 - 1.0
        # The training target: side, letterboxed, over frame width.
        side_lb = true_side * float(np.hypot(go[0, 0], go[1, 0]))
        size_enc = side_lb / DETECTION_INPUT_SIZE

        conf, box = decode_detection([1.0, 0.0], [cx_enc, 0], [cy_enc, 0],
                                     [size_enc, 0], go, 0)
        got_side = box[2] - box[0]
        assert abs(got_side - true_side) < 0.5, (
            f"size decode round-trip failed: put in {true_side:.1f}px, "
            f"got back {got_side:.1f}px (ratio {got_side / true_side:.3f}). "
            f"If the ratio is ~2.0, SIZE_DECODE_FACTOR has been reverted to "
            f"the hg_model.cpp value -- read SIZE_DECODE in the docstring.")
        assert abs((box[0] + box[2]) / 2 - cx_o) < 0.5
        say(f"  size round-trip {true_side:6.1f}px -> {got_side:6.1f}px  ok")

    # 2. A marked pixel must land where rotate_points_upright says it will.
    for orientation in (0, 90, 180, 270):
        img = np.zeros((60, 100), np.uint8)
        py, px = 12, 77
        img[py, px] = 255
        rot = rotate_upright(img, orientation)
        want = rotate_points_upright([[px, py]], orientation, 100, 60)[0]
        got = np.argwhere(rot == 255)[0][::-1]
        assert np.allclose(want, got, atol=1.0), (
            f"orientation {orientation}: image puts the pixel at {got}, "
            f"rotate_points_upright predicts {want} -- ground truth would be "
            f"misaligned from the image by that amount.")
        say(f"  orientation {orientation:3d} image/points agree  ok")

    # 3. The evaluation crop is deterministic and mirrors right hands.
    rng = np.random.default_rng(0)
    kps = rng.uniform(100, 400, (21, 2))
    a = keynet_crop_matrix(kps, is_right=False)
    b = keynet_crop_matrix(kps, is_right=False)
    assert np.allclose(a, b), "evaluation crop is not deterministic"
    r = keynet_crop_matrix(kps, is_right=True)
    assert not np.allclose(a, r), "is_right did not mirror the crop"
    say("  keypoint crop deterministic, right-hand mirrored  ok")

    say("preprocess_baseline.self_check: all checks passed")
    return True


if __name__ == "__main__":
    self_check()
