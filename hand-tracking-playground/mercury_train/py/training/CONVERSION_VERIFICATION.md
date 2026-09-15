# ONNX -> PyTorch conversion: numerical verification

Verified 2026-08-27 against the `load_detnet_weights` /
`load_keynet_weights` functions that `trainer_detection.py` and
`kpest_trainer.py` actually call.

## What was run

- DetNet: `python detection/verify_weights.py`
- KeyNet: `python keypoint/load_weights.py` (the numerical cross-check is
  built into that file's own `__main__` block)

Both scripts do the same thing: load the shipped ONNX weights into a
freshly constructed model via the project's own loader function, then run
one shared random input through both the original ONNX graph (executed
via `onnxruntime`) and the freshly loaded PyTorch module, and report the
max absolute difference per output.

## Results (2026-08-27)

DetNet (`load_detnet_weights`, `grayscale_detection_160x160.onnx`,
input `torch.randn(1, 1, 160, 160)`, seed 0):

| output | max abs diff vs ONNX |
|---|---|
| exists | 6.64e-08 |
| cx     | 2.98e-07 |
| cy     | 7.15e-07 |
| size   | 1.19e-07 |

KeyNet (`load_keynet_weights`, `grayscale_keypoint_jan18.onnx`,
inputs `torch.randn(1, 1, 128, 128)` image + `torch.randn(1, 42)`
keypoints, seed 0):

| output | max abs diff vs ONNX |
|---|---|
| heatmap_xy    | 3.25e-07 |
| heatmap_depth | 5.96e-07 |
| scalar_extras | 2.86e-06 |
| curls         | 9.54e-07 |

Every value is consistent with ordinary float32 rounding, not a real
discrepancy between the two implementations.

## Re-running this

Both scripts are self-contained given the repo's own dependencies (`torch`,
`onnx`, `onnxruntime`, `numpy`). No extra setup beyond whatever environment
already runs `trainer_detection.py` / `kpest_trainer.py`.
