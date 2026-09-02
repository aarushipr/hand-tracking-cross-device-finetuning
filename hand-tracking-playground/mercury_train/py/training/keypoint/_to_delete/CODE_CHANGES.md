# Code Changes to the Keypoint Folder

This document records every change made to the keypoint training pipeline, what was changed, and why.

---

## `settings.py`

### Added `depth_loss_mul`
```python
# Before
# (not present — 0.03 was hardcoded inside kpest_trainer.py and validatoor.py)

# After
depth_loss_mul = 0.03
```
**Why:** `0.03` was a magic number buried inside the training and validation loops with a `# ??` comment suggesting even the original author was unsure of the value. All other loss weights lived in `settings.py`, so this one should too. Now if you want to tune depth weighting, there is one place to change it and it applies consistently to both training and validation.

### Added comments to all settings
```python
# Before
using_pose_predicted_input = True
existence_loss_mul = 0.0005

# After
# hand pose predicted from previous frame
using_pose_predicted_input = True

# confidence yes/no if there is a hand in the image
existence_loss_mul = 0.0005
```
**Why:** The settings file is the first place someone new to the code looks to understand what the model is doing. Without comments, the variable names alone don't explain what each loss controls.

---

## `kpest_trainer.py`

### Added `import shutil`
**Why:** Required for the `shutil.copy` calls that replace `os.system("cp ...")`. See checkpoint saving section below.

### Removed duplicate `batch_size_per_device`
```python
# Before
batch_size_per_device = 64
batch_size_per_device = 256

# After
batch_size_per_device = 256  # Warning: high values can OOM RAM, be careful
```
**Why:** The first line was immediately overwritten by the second, so it did nothing. It looked like a meaningful value when it wasn't. The warning comment was moved inline where it's actually visible.

### Renamed wandb project
```python
# Before
wandb_name = "free_scans_2d_input_jan17"

# After
wandb_name = "keypoint_estimator_training"
```
**Why:** The original name was a date-stamped personal label from the original developer's session. It means nothing to anyone else and would be confusing when looking at wandb runs.

### Removed stale `if True:` block and commented-out datasets
```python
# Before
if True:
    # val_datasets.append(val_dataset_and_name(ArtificialDataset(
    #     "/media/moses/raid/...
    val_datasets.append(...)
    val_datasets.append(...)
    val_datasets.append(...)

# After
val_datasets = [
    val_dataset_and_name(..., "panoptic_manual"),
    val_dataset_and_name(..., "freihand"),
    val_dataset_and_name(..., "tom_openhands"),
]
```
**Why:** `if True:` is a no-op wrapper that only exists as a toggle placeholder — it does nothing but add indentation and confusion. The commented-out datasets referenced absolute paths on a specific machine (`/media/moses/raid/...`) that don't exist anywhere else. Removed both and inlined the active datasets into a clean list.

### Added SLURM-aware `num_workers`
```python
# Before
num_workers=multiprocessing.cpu_count()

# After
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
```
**Why:** On a SLURM cluster, `multiprocessing.cpu_count()` returns the total number of CPUs on the node, not the number allocated to your job. If your job requests 8 CPUs but the node has 64, this would spawn 64 DataLoader workers — far more than your allocation — causing resource contention and potential job termination. `SLURM_CPUS_PER_TASK` is the SLURM environment variable that holds your actual allocation. On a non-SLURM machine it falls back to `cpu_count()` as before. Both the training and validation dataloaders now share the same `num_workers` variable for consistency.

### Removed unused `avg_loss_train` assignment
```python
# Before
avg_loss_train = train_loop(device, dataloader_train, model, optimizer)

# After
train_loop(device, dataloader_train, model, optimizer)
```
**Why:** `avg_loss_train` was stored but never referenced again — the best-model logic that used it was commented out. Storing it implied it was being used when it wasn't.

### Fixed validation loss averaging across datasets
```python
# Before
for vn in val_datasets:
    a = validatoor.validation_loop(...)
    validation_loss = a.mean_loss_no_pred + a.mean_loss_pred
# validation_loss only holds the last dataset's result

# After
val_losses = []
for vn in val_datasets:
    a = validatoor.validation_loop(...)
    val_losses.append(a.mean_loss_no_pred + a.mean_loss_pred)
mean_validation_loss = sum(val_losses) / len(val_losses)
```
**Why:** The old code overwrote `validation_loss` on every loop iteration, so only the last dataset (Tom) was retained. This made best-model tracking meaningless. Now all three validation losses are collected and averaged into a single honest number representing overall generalisation performance.

### Implemented best-model checkpoint saving
```python
# Before (commented out, broken)
# best_model = validation_loss < last_validation_loss
# if avg_loss_train < validation_loss:
#     best_model = True
# if best_model:
#     os.system(f"cp checkpoint.pth checkpoint_best.pth")

# After (working)
is_best = mean_validation_loss < best_validation_loss
if is_best:
    best_validation_loss = mean_validation_loss
...
if is_best:
    shutil.copy(
        os.path.join(checkpoint_dir, "checkpoint.pth"),
        os.path.join(checkpoint_dir, "checkpoint_best.pth"))
```
**Why:** The original logic was half-finished and commented out. The condition `if avg_loss_train < validation_loss: best_model = True` was also logically wrong — it was checking whether training loss was lower than validation loss, which is almost always true and has nothing to do with whether the model improved. The corrected version compares the current epoch's mean validation loss against the best seen so far and saves a permanent copy when it improves. `best_validation_loss` is also saved into the checkpoint so it survives training restarts.

### Replaced `os.system("cp ...")` with `shutil.copy`
```python
# Before
os.system(f"cp {checkpoint_dir}/checkpoint.pth {checkpoint_dir}/checkpoint_{epoch}.pth")

# After
shutil.copy(
    os.path.join(checkpoint_dir, "checkpoint.pth"),
    os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))
```
**Why:** `os.system` spawns a shell subprocess to run the `cp` command. This is fragile — it only works on Unix systems, it breaks if paths contain spaces, and it's slower than a direct Python call. `shutil.copy` is the standard Python library function for copying files, works on all platforms, handles edge cases correctly, and doesn't spawn a subprocess.

### Changed checkpoint path to absolute
```python
# Before
checkpoint_dir = "checkpoints"

# After
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
```
**Why:** A relative path like `"checkpoints"` is resolved relative to whatever directory the process was launched from. On SLURM, jobs are typically launched from a scratch or home directory, not the script's directory. This means checkpoints could be written to an unexpected location or a temporary folder that gets wiped. `os.path.abspath(__file__)` gives the absolute path of `kpest_trainer.py` itself, so the `checkpoints` folder is always created next to the script regardless of where SLURM launches the job from.

### Removed misleading `else: print("Not best performance!")`
```python
# Before
if epoch % 10 == 0:
    os.system(f"cp ...")
else:
    print(f"Not best performance!")

# After
if epoch % 10 == 0:
    shutil.copy(...)
```
**Why:** This `else` belonged to the every-10-epochs check, not to any performance comparison. It printed "Not best performance!" on every epoch that wasn't a multiple of 10, which had nothing to do with performance — it was a leftover from when best-model tracking was active. It would have printed on epochs 1–9, 11–19, etc., completely misleading.

### Removed debug print statements
```python
# Removed:
print(gt_curls.shape)
print(input_predicted_keypoints.shape)
```
**Why:** These were debug lines from development. On SLURM they would appear in the log file thousands of times per epoch, making it impossible to find real log output.

### Removed unused variable `is_hand_expanded_for_scalars`
```python
# Removed:
is_hand_expanded_for_scalars = gt_is_hand[:, None]
```
**Why:** This variable was created but never referenced anywhere in the function. Dead code that implies something is happening when nothing is.

### Replaced `* 0.03` magic number with `settings.depth_loss_mul`
```python
# Before
loss_depth = mse(...) * 0.03  # ??

# After
loss_depth = mse(...) * settings.depth_loss_mul
```
**Why:** This was the only loss weight not in `settings.py`. The `# ??` comment showed the author wasn't even certain about the value. Moved to settings for consistency and tunability.

### Removed stale commented-out duplicate validation block
Lines 303–309 in the original were a complete copy of the validation loop that had been left in as comments after being moved earlier in the function. Removed entirely — dead code that adds noise.

---

## `validatoor.py`

This file was significantly rewritten. Below are the specific changes.

### Removed unused imports
```python
# Removed:
import typing
from ArtificialData import ArtificialDataset
import KeyNet
import select
import sys
import argparse
import torch.nn as nn
import py.training.common.a_geometry as geo
import cv2
import visualizer
from torch.utils.data import DataLoader
```
**Why:** None of these were used in the actual validation logic. They were leftovers from earlier versions or from the now-removed `main()`. Keeping unused imports misleads readers into thinking those things matter.

### Added `torch.no_grad()`
```python
# Before
for batch, doct in enumerate(dataloader):
    ...
    model_pred_xy, ... = model(...)

# After
with torch.no_grad():
    for batch, doct in enumerate(dataloader):
        ...
        model_pred_xy, ... = model(...)
```
**Why:** During validation, `loss.backward()` is never called, so PyTorch's computation graph — which tracks every operation to enable gradient calculation — is pure waste. Without `torch.no_grad()`, PyTorch still builds that graph, consuming significant GPU memory and slowing inference. Wrapping the entire loop in `no_grad()` tells PyTorch to skip this overhead entirely. This is standard practice for any evaluation loop.

### Added `has_depth` and `has_xy` masking to match the training loop
```python
# Before
loss_hmap = loss_fn(model_pred_xy, gt_xy)
loss_depth = loss_fn(model_pred_depth, gt_depth) * 0.03  # XXX: Rethink this!

# After
has_depth = has_depth * gt_is_hand
has_depth_expanded = has_depth[:, None, None]
has_xy = doct['has_xy'].to(device)
has_xy = has_xy * gt_is_hand
has_xy_expanded = has_xy[:, None, None, None]

loss_hmap = loss_fn(model_pred_xy * has_xy_expanded, gt_xy * has_xy_expanded)
loss_depth = loss_fn(model_pred_depth * has_depth_expanded, gt_depth * has_depth_expanded) * settings.depth_loss_mul
```
**Why:** This fixes the bug flagged by the `# XXX: Rethink this!` comment. Real datasets like FreiHand and Panoptic do not provide depth labels — their `gt_depth` is all zeros. Without the `has_depth` mask, the model is penalised for its depth predictions against those zero values even though there is no real ground truth to compare against. This makes the validation depth loss meaningless and artificially inflated for real datasets. The same issue applied to `loss_hmap` for images without a hand. The fix brings validation masking into exact alignment with the training loop, so the numbers are directly comparable.

### Fixed `loss_existence` duplicate
```python
# Before
loss_is_hand = loss_fn(gt_is_hand, model_is_hand) * settings.existence_loss_mul
loss_hmap = loss_fn(model_pred_xy, gt_xy)
loss_depth = loss_fn(...) * 0.03
loss_existence = loss_fn(model_is_hand, gt_is_hand)  # identical to loss_is_hand, never stored

loss_array_existence[batch] = loss_is_hand  # loss_existence is never used

# After
loss_existence = loss_fn(model_pred_is_hand, gt_is_hand) * settings.existence_loss_mul
loss_array_existence[batch] = float(loss_existence)
```
**Why:** The old code computed `loss_is_hand` and `loss_existence` with the same formula (just with arguments in different order — MSE is symmetric so the result is identical). Only `loss_is_hand` was stored. `loss_existence` was dead code that existed purely to confuse. Consolidated into one clearly named variable consistent with the training loop. Also added `float()` conversion before storing, consistent with how losses are handled elsewhere.

### Replaced `* 0.03` with `settings.depth_loss_mul`
Same reasoning as in `kpest_trainer.py` — the magic number is now in one place in `settings.py`.

### Removed unused variable `name`
```python
# Removed:
name = f"validation_{pstring}"
```
**Why:** Defined but never referenced. Dead code.

### Removed commented-out visualization block
Lines 106–125 in the original contained a large commented-out block attempting to save validation images to disk per epoch. It referenced `visualizer.make_visualization_images` which does not exist in the current visualizer. Removed rather than leave broken dead code.

### Added depth logging to non-artificial branch
```python
# Before
wandb.log({
    f"{output_folder}_validation_loss_xy": mean_loss_no_pred,
    # depth was commented out
})

# After
wandb.log({
    f"{output_folder}_validation_loss_xy": mean_loss_no_pred,
    f"{output_folder}_validation_loss_depth": mean_loss_no_pred_depth,
    f"{output_folder}_validation_loss_existence": mean_loss_no_pred_existence,
})
```
**Why:** Depth loss was being computed correctly (after the masking fix) but not logged for real datasets. Since depth is now properly masked, logging it gives useful information about how well the model generalises depth estimation to real data.

---

## Detection Pipeline (`py/training/detection/`)

### `CombinedDataset.py` — Removed subject02 from training

```python
# Before: all 8 sequences loaded into training
hmdhandrect_datasets.append(HMDHandRectsDataset(...train_subject02_sequence00...))
hmdhandrect_datasets.append(HMDHandRectsDataset(...train_subject02_sequence01...))

# After: subject02 excluded with comment explaining why
# subject02 sequences are held out as the validation set.
# They must never appear here — adding them would contaminate evaluation.
```
**Why:** With a random split, val data came from the same sequences as training. This tests that the model didn't memorise specific frames, but not that it generalises to a new subject. Holding out subject02 entirely means val measures generalisation across people — a meaningful signal for cross-device hand tracking.

### `CombinedDataset.py` — Rebalanced dataset weights

```python
# Before
b(torch.utils.data.ConcatDataset(hmdhandrect_datasets), 2)
b(DarknetDataset(local_config.egohands_convert), .5)
b(EpicKitchensDataset(), 5)

# After
b(torch.utils.data.ConcatDataset(hmdhandrect_datasets), 3)  # egocentric XR — most relevant
b(DarknetDataset(local_config.egohands_convert), 1)          # egocentric non-XR — useful
b(EpicKitchensDataset(), 2)                                   # chest-mounted GoPro — wrong device type
```
**Why:** EpicKitchens (GoPro chest-mounted) had 5× weight — higher than everything else combined — despite being the dataset least representative of XR headworn cameras. It dominated training with the wrong camera geometry. HMDHandRects (the actual target device) had only 2×. Flipping this priority (HMDHandRects highest, EpicKitchens lowest) trains the detector for the actual deployment scenario.

### `trainer_detection.py` — Full cleanup and restructuring

**Moved `wandb.init` from module level into `main()`**
```python
# Before (module level — runs on every import)
wandb.init(project="detection_nov1_160x160")

# After (inside main — only runs when actually training)
wandb.init(project="hand_detection_training", entity="col")
```
**Why:** Module-level `wandb.init` fires every time `trainer_detection` is imported, even for unit tests or other scripts that just need `train_batch`. Renamed the project from a date-stamped label to a descriptive name.

**Moved `device` assignment into `main()` with CPU fallback**
```python
# Before
device = torch.device("cuda:0")

# After (inside main)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
**Why:** Hardcoding `cuda:0` at module level crashes immediately on machines without a GPU (e.g. local dev, debugging). The fallback means the code runs on CPU when no GPU is present, which is useful for testing logic without a cluster.

**Added `device` as parameter to `train_batch` and `validate_epoch`**
```python
# Before
def train_batch(val, loss_fn, optimizer, model):
    inp = val['image'].to(device)  # captured from module global

# After
def train_batch(device, batch, loss_fn, optimizer, model):
    inp = batch['image'].to(device)  # explicit dependency
```
**Why:** Functions that silently depend on a module-level global are hard to test and reason about. Passing `device` explicitly makes the data flow clear.

**SLURM-aware `num_workers`**
```python
# Before
num_workers = 24

# After
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
```
**Why:** Hardcoded 24 is wrong on most machines (too many or too few). Same fix as the keypoint trainer — use the SLURM allocation when available, fall back to `cpu_count()` otherwise.

**Replaced random split with cross-device split**
```python
# Before
dataset = CombinedDataset()
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [...])

# After
train_dataloader = DataLoader(CombinedDataset(), ...)

val_dataset = ConcatDataset([
    HMDHandRectsDataset(...train_subject02_sequence00...),
    HMDHandRectsDataset(...train_subject02_sequence01...),
])
val_dataloader = DataLoader(val_dataset, ...)

# test_dataloader: HOT3D placeholder when dataset is obtained
```
**Why:** Same reasoning as keypoint — random split doesn't test generalisation. Val is now subject02 (different person, same device). Test is a HOT3D placeholder for future true cross-device evaluation.

**Saved `best_validation_loss` in checkpoint**
```python
# Before
last_validation_loss = 10000000000
# not saved in checkpoint

# After
best_validation_loss = float('inf')
# saved and loaded: checkpoint['best_validation_loss']
```
**Why:** Without this, if training is interrupted and resumed, `best_validation_loss` resets to infinity and the best checkpoint gets overwritten. Now the best-model tracking survives restarts.

**Replaced `os.system("cp ...")` with `shutil.copy`**
Same reasoning as keypoint — cross-platform, no subprocess, correct path handling.

**Replaced relative checkpoint path with absolute**
Same reasoning as keypoint — SLURM jobs don't start from the script directory.

**Removed `firstbatchcpu`** — defined but never called anywhere.

**Removed debug prints** — `print("inp", inp.shape)` and `print(f"loss is {loss}")` fired every batch, flooding SLURM logs.

**Removed dead wandb comment**
```python
# Removed: # wandb.log({f"{name}_loss": loss})
```
This commented-out line referenced a variable `name` that doesn't exist in `train_batch`. It was left over from an earlier version and would have caused a `NameError` if uncommented.

**Removed unused imports** — `select`, `cv2`, `numpy`.

### `trainer_detection.py` — Removed test evaluation at end of training

The test dataloader (random split) has been removed. A placeholder comment documents where HOT3D should be wired in once the dataset is obtained. The test evaluation block in the original code is gone until that data exists.

---

### `validatoor.py` — Removed broken `main()`
```python
# Removed entirely
def main():
    ...
    hd_val = ArtificialDataset("/3/inshallah7_validation/")  # hardcoded path
    ...
    validation_loop(device, dataloader_val, model, loss_fn, gui=True)  # gui= doesn't exist
```
**Why:** This `main()` had two fatal bugs. First, it called `validation_loop(..., gui=True)` but that function has no `gui` parameter — it would crash immediately with a `TypeError`. Second, it hardcoded the path `/3/inshallah7_validation/` which only exists on one specific machine. Since `validatoor.py` is always called as a module from `kpest_trainer.py` and never run directly as a script, this `main()` served no purpose. Removed rather than leave code that would crash if anyone tried to run it.

### Replaced random 70/10/20 split with cross-device split
```python
# Before
dataset = CombinedDataset.AllOfTheDatasetsCombined()
test_size  = int(0.2 * len(dataset))
val_size   = int(0.1 * len(dataset))
train_size = len(dataset) - val_size - test_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(...)

# After
dataloader_train = DataLoader(CombinedDataset.AllOfTheDatasetsCombined(), ...)
dataloader_val   = DataLoader(RandoDataset(..., "frei_gs.csv"), ...)
dataloader_test  = DataLoader(RandoDataset(..., "tom.csv"), ...)
```
**Why:** A random split drawn from the same combined dataset means val and test are the same distribution as training — the model has never seen those exact samples but has seen the same cameras, lighting, and subjects. This is fine for measuring overfitting but says nothing about cross-device generalisation, which is the thesis topic. The proper structure is: train on one set of sources, validate on FreiHand (a different real capture setup), and test on Tom (completely held out). This way each split tells you something different: training loss = fit, val loss = generalisation to a new real dataset, test loss = unbiased final performance. FreiHand and Tom were also simultaneously present in `CombinedDataset` (training) and used as validation — that was a data leakage bug that made evaluation meaningless.

---

## `visualizer.py`

### Removed debug print statements
```python
# Removed:
print(predicted_keypoints)
print("img", predicted_keypoints_img)
np.set_printoptions(suppress=True)
```
**Why:** These three lines fired inside `display_and_log_output`, which is called every 300 training batches. On SLURM they would dump large arrays into the log file hundreds of times per epoch, making the logs unreadable. `np.set_printoptions` was only there to format those print outputs and has no purpose without them.
