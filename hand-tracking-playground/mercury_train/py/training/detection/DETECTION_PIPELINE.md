# Hand Detection Pipeline

## What is Hand Detection?

Before the keypoint estimator can run, the system needs to know *where* the hand is in the image. Hand detection is that first step: given a full camera frame, find the bounding box around each hand (if any). The result — center x, center y, and size of the box — is then used to crop the image and pass it to the keypoint estimator.

This is a two-stage pipeline: **detect → crop → estimate**. This document covers the detection stage.

---

## File Overview

| File | Role |
|---|---|
| `trainer_detection.py` | Entry point — training and validation loop |
| `DetNet.py` | The neural network model |
| `CombinedDataset.py` | Combines training datasets with weighted sampling |
| `HMDHandRectsDataset.py` | Loads egocentric XR hand bounding-box sequences |
| `EpicKitchensDataset.py` | Loads Epic Kitchens GoPro hand bounding-box data |
| `DarknetDataset.py` | Loads EgoHands in darknet annotation format |
| `header.py` | Model input dimensions (`160×160`) |
| `local_config.py` | Disk paths to datasets |

---

## Data Split

The three splits are kept completely separate so that validation measures generalisation to unseen subjects, not just unseen frames.

| Split | Dataset | Purpose |
|---|---|---|
| **Train** | HMDHandRects subj00+01, EgoHands, EpicKitchens | What the model learns from |
| **Val** | HMDHandRects subject02 (seq00 + seq01) | Different subject, same device — checked every epoch |
| **Test** | HOT3D *(placeholder — dataset not yet obtained)* | Different XR device — true cross-device test |

Subject02 was chosen as the validation split because holding out an entire subject is a stronger test than holding out random frames from the same subjects. Random-split validation only checks that the model didn't memorise specific frames; subject-split validation checks whether it generalises across people.

HOT3D (Project Aria) is the intended test set because it represents a genuinely different XR capture device from the HMDHandRects training data. Once obtained, wire it in at the `# test_dataloader` placeholder in `trainer_detection.py`.

---

## Data Flow

```
Train: CombinedDataset (HMDHandRects subj00+01 + EgoHands + EpicKitchens)
Val:   HMDHandRects subject02 (seq00 + seq01)
Test:  HOT3D [placeholder]
        ↓
DataLoader  (batching, shuffling, parallel loading)
        ↓
train_batch / validate_epoch
        ↓
DetNet (forward pass)
        ↓
Loss computation (exists + center_x + center_y + size)
        ↓
loss.backward() → optimizer.step()  [training only]
        ↓
wandb logging + checkpoint saving
```

---

## The Datasets

### HMDHandRectsDataset
Egocentric captures from an XR headset with bounding-box annotations for each hand. This is the most device-relevant dataset — it matches the target deployment hardware. Split by subject:

| Subject | Sequences | Split |
|---|---|---|
| subject00 | seq00–seq03 | Training |
| subject01 | seq00–seq01 | Training |
| subject02 | seq00–seq01 | **Validation** |

### DarknetDataset / EgoHands
Egocentric hand data annotated in darknet format. Not from XR hardware, but still egocentric — camera geometry is similar. Weight: 1.0.

### EpicKitchensDataset
Hand bounding boxes from the Epic Kitchens dataset, captured with a GoPro chest-mounted camera. Camera geometry is different from a headworn XR device (chest position vs. eye position, different field of view). Included for detection diversity but given the lowest weight (2.0 compared to HMDHandRects' 3.0) because it is the least representative of the deployment scenario.

### Dataset Weights

| Dataset | Weight | Rationale |
|---|---|---|
| HMDHandRects (subj00+01) | 3.0 | Egocentric XR — matches target device |
| EgoHands | 1.0 | Egocentric non-XR — useful for diversity |
| EpicKitchens | 2.0 | Chest-mounted GoPro — wrong geometry, reduced weight |

Weights control how often each dataset appears during training relative to the others. The `RepeatDataset` wrapper repeats smaller datasets so they sample at approximately the right proportion.

---

## The Model: DetNet

DetNet is a compact MobileNetV2-style convolutional network. It takes a grayscale image and directly regresses four bounding-box values per hand.

### Input
A single `160×160` grayscale image: `[B, 1, 160, 160]` — matches Monado's shipped
production baseline (`grayscale_detection_160x160.onnx`) exactly, which is what
makes literal ONNX→PyTorch weight loading possible (see `load_monado_weights.py`).

### Architecture

```
input [B, 1, 160, 160]
        ↓ backbone (Conv2d + 12 InvertedResidual blocks)
        → feature map [B, 160, ...]
        ↓ Flatten
        → [B, 1440]
        ↓ FC layers: 1440 → 256 → 8 → 8
        ↓ split across second dimension:
        ├── exists     = sigmoid(x[:, 0:2])   [B, 2]  — left/right hand confidence
        ├── center_x   = x[:, 2:4]            [B, 2]  — horizontal center
        ├── center_y   = x[:, 4:6]            [B, 2]  — vertical center
        └── size       = x[:, 6:8]            [B, 2]  — bounding box size
```

The model predicts two sets of values (index 0 and 1) — one for each hand (left and right). The `exists` output passes through sigmoid to produce a probability in [0, 1].

The fully connected layers are noted in the code as a possible area for improvement — large linear layers are less efficient than using a small convolutional head, and this is worth revisiting if training is slow.

---

## The Loss Function

Four MSE losses, summed:

```python
loss_exists   = mse(exists_gt, exists_pred)
loss_center_x = mse(center_x_gt * exists_gt, center_x_pred * exists_gt)
loss_center_y = mse(center_y_gt * exists_gt, center_y_pred * exists_gt)
loss_size     = mse(size_gt * exists_gt,     size_pred * exists_gt)

loss = loss_exists + loss_center_x + loss_center_y + loss_size
```

The center and size losses are masked by `exists_gt`. This means: if there is no hand in the image (`exists_gt = 0`), the model is only penalised for the existence prediction, not for whatever it guesses for center and size. Without this mask, background images would push center and size predictions toward zero, which is meaningless.

All four losses are unweighted — unlike the keypoint pipeline they are treated equally.

---

## The Training Loop

`train_batch` in `trainer_detection.py` runs one batch:

1. Move image and labels to GPU
2. Forward pass through DetNet
3. Compute four losses (center/size masked by exists)
4. `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`
5. Log combined loss to wandb

`validate_epoch` runs the full val dataloader under `torch.no_grad()` and returns the mean loss.

---

## Checkpointing

Same structure as the keypoint pipeline:

- `checkpoint.pth` — latest epoch, overwritten every epoch
- `checkpoint_{N}.pth` — periodic copy saved every 10 epochs
- `checkpoint_best.pth` — copy of the epoch with lowest validation loss

`best_validation_loss` is stored inside the checkpoint so best-model tracking survives training restarts.

---

## Environment Configuration

The detection pipeline uses `header.py` for model dimensions and reads wandb/SLURM configuration directly from environment variables.

| Variable | Purpose |
|---|---|
| `SLURM_CPUS_PER_TASK` | Number of DataLoader workers (falls back to `cpu_count()` off-cluster) |
| `WANDB_MODE=disabled` | Disable wandb logging without code changes |
| `WANDB_API_KEY` | Authenticate wandb without interactive login |

---

## Running on SLURM

```bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=your_key_here
# export WANDB_MODE=disabled  # uncomment to disable wandb

python trainer_detection.py
```

The trainer reads `SLURM_CPUS_PER_TASK` automatically. The detection pipeline has no GUI display — there is no visualizer equivalent, so no `AD4_ENABLEGUI` flag is needed.

---

## What's Missing / Future Work

- **Test set**: HOT3D needs to be downloaded and wired in at the `# test_dataloader` placeholder. This gives a true cross-device generalisation number on hardware different from training.
- **FC layer architecture**: The large linear layers (1440→256) are noted in `DetNet.py` as a candidate for replacement with a convolutional head. If training is slow or the model overfits, this is the first thing to try.
- **Weight initialization**: `load_monado_weights.py` loads Monado's shipped `grayscale_detection_160x160.onnx` weights into this module (verified against the ONNX model to ~1e-7). `trainer_detection.py` still initializes from scratch (`init_weights()`) — wiring in `load_monado_weights.py` before training is what turns this into literal fine-tuning rather than training from scratch.
- **No visualizer**: The detection pipeline doesn't have an equivalent of the keypoint visualizer. Adding wandb image logging (drawing predicted boxes on the input image) would make it easier to see whether the model is finding hands correctly, not just what its loss number is.
