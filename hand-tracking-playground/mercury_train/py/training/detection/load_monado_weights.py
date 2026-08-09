"""
Loads Monado's shipped production DetNet weights (grayscale_detection_160x160.onnx)
into this repo's PyTorch DetNet module, enabling literal fine-tuning from the
real baseline instead of training from scratch.

Why this isn't a plain state_dict-by-name load
------------------------------------------------
DetNet.py's backbone is written as separate nn.Conv2d(bias=False) + nn.BatchNorm2d
pairs. When the shipped ONNX model was exported (torch.onnx.export(..., do_constant_folding=True),
see DetNet.py's __main__), the exporter algebraically fused each Conv+BatchNorm pair
into a single Conv with a bias. The result: the ONNX graph's backbone initializers
show up as anonymous "onnx::Conv_<n>" weight/bias pairs (51 of them — 1 stem conv +
50 across the 12 IRB stages) with no BatchNorm nodes at all, so there's nothing to
match by name. The FC head (fc.1 / fc.2 / fc.4 / fc.5 / fc.7) was *not* fused —
those initializers keep their original state_dict names and load directly.

The fix used here: load each fused (weight, bias) pair into the corresponding
Conv2d (converting it from bias=False to bias=True), then set the Conv's paired
BatchNorm2d to the identity transform (weight=1, bias=0, running_mean=0,
running_var=1, eps=0). In eval mode, BatchNorm2d then computes
y = (x - 0) / sqrt(1 + 0) * 1 + 0 = x, i.e. a no-op — reproducing exactly the
function the fused ONNX conv already computes, without needing to invert the
fusion algebra (which would require the original BN running stats that the
fused graph no longer contains).

Verified: running the converted PyTorch model and the ONNX model (via
onnxruntime) on the same random [1, 1, 160, 160] input gives outputs that
match to ~1e-7 (float32 precision), i.e. numerically identical.

Usage:
    python load_monado_weights.py
    # or
    from load_monado_weights import load_monado_detnet_weights
    model = load_monado_detnet_weights(DetNet())
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn

ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "hand-tracking-models", "grayscale_detection_160x160.onnx")

_FC_NAME_MAP = {
    "1.weight": "fc.1.weight", "1.bias": "fc.1.bias",
    "2.weight": "fc.2.weight", "2.bias": "fc.2.bias",
    "2.running_mean": "fc.2.running_mean", "2.running_var": "fc.2.running_var",
    "4.weight": "fc.4.weight", "4.bias": "fc.4.bias",
    "5.weight": "fc.5.weight", "5.bias": "fc.5.bias",
    "5.running_mean": "fc.5.running_mean", "5.running_var": "fc.5.running_var",
    "7.weight": "fc.7.weight", "7.bias": "fc.7.bias",
}


def _identity_bn(bn: nn.BatchNorm2d):
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.fill_(0.0)
        bn.running_mean.fill_(0.0)
        bn.running_var.fill_(1.0)
    bn.eps = 0.0


def load_monado_detnet_weights(model, onnx_path=ONNX_PATH):
    """Loads grayscale_detection_160x160.onnx's weights into `model` in place.

    `model` must be a freshly constructed DetNet() with the 160x160 / 1440-wide
    FC head (i.e. header.py's model_input_height = 160). Returns `model`.
    """
    graph = onnx.load(onnx_path).graph

    fc_inits = {}
    conv_pairs = []
    pending_names, pending = [], {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.name.startswith("fc."):
            fc_inits[init.name] = arr
        else:
            pending[init.name] = arr
            pending_names.append(init.name)

    assert len(pending_names) % 2 == 0
    for i in range(0, len(pending_names), 2):
        w, b = pending[pending_names[i]], pending[pending_names[i + 1]]
        assert w.ndim == 4 and b.ndim == 1 and w.shape[0] == b.shape[0]
        conv_pairs.append((w, b))

    # --- FC head: load by name, unchanged structure ---
    fc_state = model.fc.state_dict()
    for local_key, onnx_key in _FC_NAME_MAP.items():
        assert local_key in fc_state, f"DetNet.fc structure changed: {local_key} not found"
        fc_state[local_key] = torch.from_numpy(fc_inits[onnx_key].copy())
    model.fc.load_state_dict(fc_state)

    # --- Backbone: positional conv/BN-fusion load ---
    conv_idx = 0

    def assign(conv: nn.Conv2d, w: np.ndarray, b: np.ndarray):
        assert list(conv.weight.shape) == list(w.shape), (conv.weight.shape, w.shape)
        conv.weight = nn.Parameter(torch.from_numpy(w.copy()))
        conv.bias = nn.Parameter(torch.from_numpy(b.copy()))

    modules = list(model.backbone.children())
    i = 0
    while i < len(modules):
        mod = modules[i]
        if isinstance(mod, nn.Conv2d):
            # stem: Conv2d, BatchNorm2d, ReLU6
            w, b = conv_pairs[conv_idx]; conv_idx += 1
            assign(mod, w, b)
            bn = modules[i + 1]
            assert isinstance(bn, nn.BatchNorm2d)
            _identity_bn(bn)
        elif hasattr(mod, "conv"):  # InvertedResidual block
            for sub in mod.conv:
                if isinstance(sub, nn.Conv2d):
                    w, b = conv_pairs[conv_idx]; conv_idx += 1
                    assign(sub, w, b)
                elif isinstance(sub, nn.BatchNorm2d):
                    _identity_bn(sub)
        i += 1

    assert conv_idx == len(conv_pairs), (
        f"Expected to consume all {len(conv_pairs)} fused conv/bias pairs, "
        f"used {conv_idx}. header.py's model_input_height is probably not 160 "
        f"(FC layer width won't be 1440, backbone shapes will mismatch first).")

    return model


if __name__ == "__main__":
    from DetNet import DetNet

    model = DetNet()
    model.eval()
    load_monado_detnet_weights(model)
    print("Loaded Monado's grayscale_detection_160x160.onnx weights into DetNet.")

    # Optional numerical cross-check against the ONNX model, if onnxruntime is installed.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        x = torch.randn(1, 1, 160, 160)
        with torch.no_grad():
            exists_t, cx_t, cy_t, size_t = model(x)
        exists_o, cx_o, cy_o, size_o = sess.run(None, {"inputImg": x.numpy()})
        for name, t, o in [("exists", exists_t, exists_o), ("cx", cx_t, cx_o),
                            ("cy", cy_t, cy_o), ("size", size_t, size_o)]:
            print(f"{name} max abs diff vs ONNX: {np.max(np.abs(t.numpy() - o)):.2e}")
    except ImportError:
        print("(onnxruntime not installed — skipping numerical cross-check)")
