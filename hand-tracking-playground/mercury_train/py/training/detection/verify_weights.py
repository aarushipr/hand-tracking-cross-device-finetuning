"""
Numerical cross-check of load_detnet_weights() against Monado's shipped
grayscale_detection_160x160.onnx.

Loads the ONNX weights into a fresh DetNet via load_detnet_weights(), then
runs one shared random input through both the ONNX graph (via onnxruntime)
and the loaded PyTorch module, and reports the max absolute difference per
output. Values at ~1e-6 or smaller are ordinary float32 rounding, not a
real discrepancy -- this is what licenses treating the loaded PyTorch
model as the same network Monado ships, not merely one built to resemble
it (see load_weights.py / DetNet.py for why this isn't a plain
state-dict-by-name load: the ONNX export fused each Conv+BatchNorm pair,
discarding the original BatchNorm statistics).

Usage:
    python verify_weights.py
"""
import numpy as np
import torch
import onnxruntime as ort

import DetNet
from load_weights import load_detnet_weights, ONNX_PATH

OUTPUT_NAMES = ["exists", "cx", "cy", "size"]


def main():
    model = DetNet.DetNet()
    model.eval()
    load_detnet_weights(model)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 160)

    with torch.no_grad():
        out_t = model(x)
    out_o = sess.run(None, {"inputImg": x.numpy()})

    for name, t, o in zip(OUTPUT_NAMES, out_t, out_o):
        diff = np.max(np.abs(t.numpy() - o))
        print(f"{name} max abs diff vs ONNX: {diff:.2e}")


if __name__ == "__main__":
    main()
