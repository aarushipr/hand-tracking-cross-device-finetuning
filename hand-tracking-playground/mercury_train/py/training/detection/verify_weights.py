"""
Numerical cross-check of load_detnet_weights() against Monado's shipped
grayscale_detection_160x160.onnx: one random input through both, reporting the max
absolute difference per output. Around 1e-6 is float32 rounding. Needed because
the ONNX export fused every Conv+BatchNorm pair, so this is not a
state-dict-by-name load. Results are recorded in CONVERSION_VERIFICATION.md.
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
