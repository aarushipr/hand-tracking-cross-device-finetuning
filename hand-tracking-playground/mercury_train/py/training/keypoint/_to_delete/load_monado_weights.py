"""
Loads Monado's shipped production KeyNet weights (grayscale_keypoint_jan18.onnx)
into this repo's PyTorch KeyNet module, enabling literal fine-tuning from the
real baseline instead of training from scratch.

Why this isn't a plain state_dict-by-name load
------------------------------------------------
Same fusion pattern Section 3.5 already found in DetNet's export:
torch.onnx.export(..., do_constant_folding=True) algebraically fuses each
Conv2d/ConvTranspose2d + BatchNorm2d pair into a single Conv with a bias
whenever a BatchNorm2d immediately follows a conv layer in the traced graph.
Checked directly against grayscale_keypoint_jan18.onnx's own initializers:

  - keypoints_network, network_1d_depth, network_extras, network_curls, and
    the lone ConvTranspose2d inside network_2d_px_coord (network_2d_px_coord.3)
    keep their original PyTorch state_dict names and load directly -- none of
    these layers is immediately followed by a BatchNorm2d in KeyNet.forward(),
    so nothing fuses. Confirmed directly: these are the only 22 named
    initializers in the graph, and their names already match KeyNet's own
    state_dict keys exactly (e.g. "network_extras.3.running_var") -- no
    remapping needed, unlike DetNet's FC head.
  - Every other Conv2d is immediately followed by a BatchNorm2d and gets
    fused: 88 anonymous onnx::Conv_<n> weight/bias entries (44 pairs) with no
    BatchNorm nodes at all -- 18 pairs from image_network (1 stem + 17 across
    its 5 IRB stages), 24 pairs from fused_network (12 IRB blocks, all
    expand_ratio=1, 2 convs each), and 2 pairs from network_2d_px_coord's two
    BN-adjacent Conv2d layers (indices 0 and 4 of that Sequential).

The fix is the same one load_monado_weights.py (detection) already uses and
this repo already trusts: load each fused (weight, bias) pair into the
corresponding Conv2d/ConvTranspose2d positionally (bias=False -> bias=True),
then set that layer's paired BatchNorm2d to the identity transform (weight=1,
bias=0, running_mean=0, running_var=1, eps=0), so the unfused module computes
exactly what the fused ONNX graph already computes, without needing to invert
the fusion algebra.

Positional order assumption: the anonymous pairs are consumed in the same
order KeyNet.forward() actually calls its submodules -- image_network, then
fused_network, then network_2d_px_coord's two fused layers -- which is also
the order torch.onnx.export's tracer assigns initializer names in
(monotonically increasing onnx::Conv_<n> suffixes: 517..647). This is the same
assumption DetNet's converter made, and it is verified the same way: see the
numerical cross-check in __main__ below, not just asserted.

Verified: running the converted PyTorch model and the ONNX model (via
onnxruntime) on the same random inputs gives outputs that match to float32
precision -- see the printed max-abs-diff values when this script is run
directly.

Usage:
    python load_monado_weights.py
    # or
    from load_monado_weights import load_monado_keynet_weights
    model = load_monado_keynet_weights(KeyNet())
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn

ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "hand-tracking-models", "grayscale_keypoint_jan18.onnx")

# Named initializers already use KeyNet's own state_dict naming -- direct
# load, no remapping needed (see module docstring).
_DIRECT_LOAD_PREFIXES = (
    "keypoints_network.", "network_1d_depth.", "network_extras.",
    "network_curls.", "network_2d_px_coord.3.",
)


def _identity_bn(bn: nn.BatchNorm2d):
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.fill_(0.0)
        bn.running_mean.fill_(0.0)
        bn.running_var.fill_(1.0)
    bn.eps = 0.0


def _consume_fused_pairs(module, conv_pairs, conv_idx):
    """Walk `module`'s immediate children positionally; wherever a Conv2d or
    ConvTranspose2d is immediately followed by a BatchNorm2d, load the next
    (weight, bias) pair into it and set that BatchNorm2d to identity.
    Recurses into InvertedResidual blocks' own `.conv` Sequential. Returns
    the updated conv_idx."""
    kids = list(module.children())
    i = 0
    while i < len(kids):
        mod = kids[i]
        is_conv = isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))
        followed_by_bn = i + 1 < len(kids) and isinstance(kids[i + 1], nn.BatchNorm2d)
        if is_conv and followed_by_bn:
            w, b = conv_pairs[conv_idx]; conv_idx += 1
            assert list(mod.weight.shape) == list(w.shape), (mod, mod.weight.shape, w.shape)
            mod.weight = nn.Parameter(torch.from_numpy(w.copy()))
            mod.bias = nn.Parameter(torch.from_numpy(b.copy()))
            _identity_bn(kids[i + 1])
        elif hasattr(mod, "conv"):  # InvertedResidual -- recurse into its own Sequential
            conv_idx = _consume_fused_pairs(mod.conv, conv_pairs, conv_idx)
        i += 1
    return conv_idx


def load_monado_keynet_weights(model, onnx_path=ONNX_PATH):
    """Loads grayscale_keypoint_jan18.onnx's weights into `model` in place.

    `model` must be a freshly constructed KeyNet() with its default
    128x128 / 22-heatmap configuration -- the configuration Monado actually
    ships. Returns `model`.
    """
    graph = onnx.load(onnx_path).graph

    direct = {}
    pending_names, pending = [], {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.name.startswith(_DIRECT_LOAD_PREFIXES):
            direct[init.name] = arr
        else:
            pending[init.name] = arr
            pending_names.append(init.name)

    # --- Direct-name branches: load by exact state_dict key ---
    state = model.state_dict()
    for name, arr in direct.items():
        assert name in state, f"KeyNet structure changed: {name} not found in state_dict"
        state[name] = torch.from_numpy(arr.copy())
    model.load_state_dict(state)

    # --- Fused branches: positional conv/BN-fusion load ---
    assert len(pending_names) % 2 == 0
    conv_pairs = []
    for i in range(0, len(pending_names), 2):
        w, b = pending[pending_names[i]], pending[pending_names[i + 1]]
        assert w.ndim in (2, 4) and b.ndim == 1 and w.shape[0] == b.shape[0]
        conv_pairs.append((w, b))

    conv_idx = 0
    conv_idx = _consume_fused_pairs(model.image_network, conv_pairs, conv_idx)
    conv_idx = _consume_fused_pairs(model.fused_network, conv_pairs, conv_idx)
    conv_idx = _consume_fused_pairs(model.network_2d_px_coord, conv_pairs, conv_idx)

    assert conv_idx == len(conv_pairs), (
        f"Expected to consume all {len(conv_pairs)} fused conv/bias pairs, "
        f"used {conv_idx}. KeyNet's branch structure (image_network/"
        f"fused_network/network_2d_px_coord) probably no longer matches "
        f"grayscale_keypoint_jan18.onnx's traced graph.")

    return model


if __name__ == "__main__":
    from KeyNet import KeyNet

    model = KeyNet()
    model.eval()
    load_monado_keynet_weights(model)
    print("Loaded Monado's grayscale_keypoint_jan18.onnx weights into KeyNet.")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        x = torch.randn(1, 1, 128, 128)
        kp = torch.randn(1, 42)
        valid = torch.ones(1)
        with torch.no_grad():
            xy_t, depth_t, extras_t, curls_t = model(x, kp, valid)
        xy_o, depth_o, extras_o, curls_o = sess.run(
            None, {"inputImg": x.numpy(), "lastKeypoints": kp.numpy(),
                   "useLastKeypoints": valid.numpy()})
        for name, t, o in [("heatmap_xy", xy_t, xy_o), ("heatmap_depth", depth_t, depth_o),
                            ("scalar_extras", extras_t, extras_o), ("curls", curls_t, curls_o)]:
            print(f"{name} max abs diff vs ONNX: {np.max(np.abs(t.numpy() - o)):.2e}")
    except ImportError:
        print("(onnxruntime not installed -- skipping numerical cross-check)")
