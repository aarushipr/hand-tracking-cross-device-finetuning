# Keypoint Estimation Pipeline

## What is Keypoint Estimation?

Keypoint estimation is the task of taking an image of a hand and predicting the 2D and 3D positions of 21 anatomical landmarks (joints) — the wrist, knuckles, and fingertips. This is the core problem in hand tracking: once you know where the joints are, you can reconstruct the full hand pose.

The pipeline here is inspired by the MegaTrack and UmeTrack papers. It takes a grayscale crop of a hand (128×128 pixels) and outputs heatmaps indicating where each joint is, along with depth estimates and several auxiliary predictions.

---

## File Overview

| File | Role |
|---|---|
| `kpest_trainer.py` | Entry point — orchestrates training and validation |
| `validatoor.py` | Validation loop logic |
| `KeyNet.py` | The neural network model |
| `HOT3DKeypointDataset.py` | Loads HOT3D hand crops and keypoint ground truth |
| `../common/hot3d_split.py` | Defines the train/test partition |
| `CombinedDataset.py` | Legacy multi-source mixer — not used by this fine-tuning |
| `ArtificialData.py` | Legacy synthetic loader — not used by this fine-tuning |
| `RandoData.py` | Legacy real-dataset loader; its crop helpers are still reused |
| `settings.py` | Loss weight hyperparameters |
| `kpest_header.py` | Environment configuration (wandb, GUI, fast mode) |
| `local_config.py` | Disk paths to datasets |
| `visualizer.py` | Renders prediction vs ground truth images for wandb |
| `maker_of_augmentations.py` | Data augmentation pipeline |
| `a_aug_config.py` | Augmentation configuration |

---

## Data Split

Fine-tuning trains on HOT3D only. The split is defined in
`py/training/common/hot3d_split.py`, which is the single source of truth for it.

| Split | Source | Purpose |
|---|---|---|
| **Train** | `train_mixed` minus a 10% validation carve-out | What the model learns from |
| **Val** | 10% of `train_mixed`, carved at sequence level | Checked every epoch — convergence and checkpoint selection |
| **Test** | `test_mixed` | Held-out participants, scored only by `py/evaluation/eval_keynet.py` |

`train_mixed` and `test_mixed` pool Aria and Quest recordings and partition
**participants** 80/20, so no subject's recordings appear on both sides. The
partition is chosen so each device individually sits near 80/20, not just the
combined total, and it is frozen as a constant in `hot3d_split.py` rather than
recomputed from disk. Validation is carved out at sequence level within the
training pool, so a participant may appear in both train and val — that is
intentional, since val only monitors convergence. Generalisation is measured on
`test_mixed` alone.

The archived Aria-only design (`train`, `test_aria`, `test_quest`,
`device_shift_quest`) is still present in `hot3d_split.py` so earlier
checkpoints stay scorable. It is not on this pipeline's path.

The trainer holds no test set at all. All evaluation lives in
`py/evaluation/eval_keynet.py`, so the zero-shot Monado baseline and the
fine-tuned model are scored by exactly the same code.

---

## Data Flow

```
hot3d_split.list_sequence_dirs(root, "train_mixed")
        ↓ split_train_val  →  train sequences / val sequences
HOT3DKeypointDataset  (one sample = one hand crop, frame_stride=5)
        ↓
DataLoader  (batching, shuffling, parallel loading)
        ↓
train_loop / validation_loop_just_one
        ↓
KeyNet (forward pass)
        ↓
Loss computation
        ↓
loss.backward() → optimizer.step()
        ↓
wandb logging + checkpoint saving
```

---

## The Dataset

### HOT3DKeypointDataset (`HOT3DKeypointDataset.py`)

The only data source for this fine-tuning. Reads HOT3D recordings directly:
hand landmarks from `umetrack_hand_data_provider.get_hand_landmarks()`,
remapped by `../common/hot3d_keypoint_mapping.py` and projected into 2D through
each camera's real calibration. One sample is one hand crop, so a frame with
both hands visible contributes two samples. Depth follows the project's own
relative-depth convention, not metric distance. See that file's docstring for
the sample index, its on-disk cache, and per-joint validity handling.

Cropping reuses `RandoData.py`'s `crop()` / `add_2d_noise_to_keypoints()`
helpers directly, so crops match what the network already expects. That is the
only remaining dependency on the legacy loaders.

### Legacy loaders

`ArtificialData.py`, `RandoData.py` and `CombinedDataset.py` implement the
upstream Monado training pipeline, which mixed Blender-rendered synthetic hands
with several public datasets. They are retained because they are upstream code
and because `RandoData.py`'s crop helpers are still used, but **no part of this
fine-tuning trains on them**.

---

## The Model: KeyNet

KeyNet is a lightweight MobileNetV2-style convolutional neural network. It takes three inputs and produces four outputs.

### Inputs
1. **`input_image`** — a 128×128 grayscale image of the hand crop `[B, 1, 128, 128]`
2. **`input_predicted_keypoints`** — a flattened vector of 42 numbers (21 joints × 2D coordinates) from the previous frame's prediction `[B, 42]`
3. **`input_predicted_keypoints_valid`** — a scalar flag per sample indicating whether the prior prediction should be trusted `[B]`

The prior pose input is the temporal feedback loop from MegaTrack/UmeTrack: in real-time tracking, knowing where the joints were a moment ago helps find them now. During training this can be zeroed out (controlled by `settings.using_pose_predicted_input`) to train the model to work from image alone.

### Architecture

```
input_image [B, 1, 128, 128]
        ↓ image_network (MobileNetV2 blocks)
        → [B, 64, 16, 16]

input_predicted_keypoints [B, 42]
        ↓ keypoints_network (Linear layer)
        → reshape to [B, 32, 16, 16]
        → scaled by use_addon flag

concatenate → [B, 96, 16, 16]
        ↓ fused_network (deeper MobileNetV2 blocks)
        → [B, 160, H, W]

        ├── px_coord_regression_2d → out_xy [B, 21, H, W]  (2D heatmaps)
        ├── depth_regression_1d   → out_depth [B, 21, 22]  (1D depth heatmaps)
        ├── extras_regression     → out_extras [B, 8]       (is_hand + elbow)
        └── curls_regression      → out_curls [B, 10]       (5 curls + 5 variances)
```

### Outputs
| Output | Shape | What it represents |
|---|---|---|
| `out_xy` | `[B, 21, 22, 22]` | 2D spatial heatmap per joint — where the joint is in the image |
| `out_depth` | `[B, 21, 22]` | 1D heatmap per joint — how far the joint is from the camera |
| `out_extras[:, 0]` | `[B]` | Hand existence confidence (passed through sigmoid → probability) |
| `out_extras[:, 1:4]` | `[B, 3]` | Elbow direction as a 3D unit vector |
| `out_curls[:, 0:5]` | `[B, 5]` | Curl angle per finger |
| `out_curls[:, 5:10]` | `[B, 5]` | Predicted uncertainty (variance) per finger curl |

---

## The Loss Function

Training is supervised by five simultaneous losses, all summed into one number that drives backpropagation.

### Masking
A critical design pattern: every loss is multiplied by an availability mask before being computed. This is because different datasets provide different labels. Real datasets have 2D positions but no depth. Images with no hand at all have no pose labels at all.

Without masking, samples without labels would push the model's predictions toward zero, corrupting training. With masking, those samples simply contribute zero gradient.

```python
loss_xy = mse(pred_xy * has_xy_mask, gt_xy * has_xy_mask)
```

### The Five Loss Components

| Loss | Formula | Weight | Purpose |
|---|---|---|---|
| `loss_xy` | MSE on 2D heatmaps | 1.0 (unscaled) | Primary task — joint locations in the image |
| `loss_depth` | MSE on 1D depth heatmaps | `depth_loss_mul = 0.03` | Joint depth from camera |
| `loss_existence` | MSE on is_hand probability | `existence_loss_mul = 0.0` | Disabled — KeyNet only ever sees regions DetNet already accepted |
| `loss_elbow` | MSE on elbow direction vector | `elbow_loss_mul = 0.0` | Disabled — HOT3D has no body pose to derive elbows from |
| `loss_curls` | Gaussian NLL on curl angles | `curls_loss_mul = 0.0` | Disabled — same reason as elbow |

For this fine-tuning only the 2D heatmap and depth terms are active. The
existence, elbow and curl multipliers are set to zero in `settings.py`: HOT3D
carries no body pose, so elbow and curl have no ground truth to learn from, and
hand existence is DetNet's task. Those three heads keep their Monado
initialisation. See `settings.py` for the per-term reasoning.

**Gaussian NLL** is used for curls instead of MSE because the model also predicts its own uncertainty (variance). This lets the model say "I'm not sure about this curl" rather than being equally penalised for all errors. The variance is clamped above `curl_min_variance = 0.01` to prevent numerical instability.

### Backpropagation

```python
loss.backward()     # compute gradients for all weights
optimizer.step()    # AdamW nudges all weights to reduce loss
optimizer.zero_grad()  # clear gradients before next batch
```

---

## The Training Loop

`train_loop` in `kpest_trainer.py` runs one full pass through the training dataset. For each batch:

1. Load the batch dictionary (`doct`) from the DataLoader
2. Extract all inputs and ground truths, move to GPU
3. Apply availability masks
4. Run forward pass through KeyNet
5. Unpack outputs and compute five losses
6. Sum losses, call `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`
7. Log all five losses to wandb
8. Every 300 batches, send a visualisation image to wandb

At the end of the loop, return the average loss across all batches.

---

## The Validation Loop

`validation_loop` in `validatoor.py` runs after every training epoch, on the
validation sequences carved out of `train_mixed`.

`validation_loop_just_one` does the actual work:
- Runs inside `torch.no_grad()` — no gradient tracking, saves memory and time
- Applies the same availability masks as training so the numbers are comparable
- Computes loss per batch, stores in arrays, returns the mean

HOT3D validation runs once, with the predicted-keypoint input withheld
(`use_prediction=False`), matching how `eval_keynet.py` scores both models.

---

## Checkpointing

After every training epoch, the trainer saves a checkpoint containing:
- `epoch` — which epoch just finished
- `state_dict` — all model weights
- `optimizer` — AdamW momentum statistics
- `best_validation_loss` — the lowest validation loss seen so far

Checkpoints are written to `checkpoints_<split>/` (so `checkpoints_train_mixed/`
for the current design). Scoping the directory by split is what stops a new run
resuming from a checkpoint trained on different data. Smoke-test runs
(`AD4_LOADFAST=1`) get `checkpoints_loadfast/` instead.

Three checkpoint files are maintained:
- `checkpoint.pth` — always the latest epoch, overwritten each time
- `checkpoint_{N}.pth` — a permanent copy saved every 10 epochs
- `checkpoint_best.pth` — a permanent copy of the epoch with the lowest validation loss

When training is resumed, the checkpoint is loaded and training picks up from `start_epoch` with the optimizer in exactly the same state as when it stopped.

Training stops at `EARLY_STOPPING_PATIENCE = 8` consecutive epochs without a
validation-loss improvement, or at the `MAX_EPOCHS = 120` ceiling, whichever
comes first. `trainer_detection.py` uses the same two values, so neither
network is given more training budget than the other.

---

## Environment Configuration

`kpest_header.py` reads three environment variables to control behaviour:

| Variable | Default | Purpose |
|---|---|---|
| `AD4_LOADFAST` | `0` | Load only a tiny slice of data for quick debugging |
| `AD4_ENABLEWANDB` | `0` | Enable wandb logging |
| `AD4_ENABLEGUI` | `1` | Enable OpenCV display windows |

On SLURM, always set `AD4_ENABLEGUI=0` (no display available) and `AD4_ENABLEWANDB=1`.

---

## Running on SLURM

A minimal SLURM script should set:

```bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export AD4_ENABLEGUI=0
export AD4_ENABLEWANDB=1
export AD4_LOADFAST=0

python kpest_trainer.py
```

The trainer automatically reads `SLURM_CPUS_PER_TASK` to set the number of DataLoader workers, so it will never spawn more workers than your job is allocated.

---

## The Visualizer and SLURM

`visualizer.py` serves two purposes that work very differently on a cluster:

**1. OpenCV display window (`cv2.imshow`)**
This requires a physical display or X11 forwarding — neither of which exists on a headless SLURM node. If `AD4_ENABLEGUI=1` on SLURM, the code will crash with an error like `cannot connect to X server`. Set `AD4_ENABLEGUI=0` and this path is skipped entirely via `if header.env_settings.gui_enabled:`.

**2. wandb image logging (`wandb.log({name: wandb.Image(...)})`)**
This works fine on SLURM. The visualizer builds a canvas in memory using OpenCV drawing functions (no display needed), converts it to a uint8 image, and sends it to wandb over the network. As long as `AD4_ENABLEWANDB=1` and you've logged in via `wandb login` (or set `WANDB_API_KEY`), you'll see visualisations appear in the wandb dashboard in real time while the job runs.

**In short:** set `AD4_ENABLEGUI=0` and `AD4_ENABLEWANDB=1` on SLURM. The window display is disabled; the wandb images keep working. You watch training progress in the browser instead of on a local display.
