"""
Generates a tiny synthetic stand-in dataset in DarknetDataset's expected
format (images/train/*.jpg + labels/train/*.txt), purely to smoke-test
whether CombinedDataset -> DetNet -> trainer_detection.py runs end to end
without needing any of the real detection datasets (HMDHandRects, EgoHands,
EpicKitchens) to be located first.

This is NOT real training data — the "hands" are just random noise patches,
there is nothing for the model to actually learn. It only proves the code
path (data loading, augmentation, model forward/backward pass, loss,
checkpointing) works mechanically. Delete dummy_egohands/ once you have
real data and don't point local_config.py at this for anything but a
smoke test.

Usage:
    python make_dummy_egohands.py [output_dir] [num_frames]
"""
import os
import sys
import random

import cv2
import numpy as np

IMG_SIZE = 320  # matches augment_image's expected input scale


def make_dummy_dataset(output_dir: str, num_frames: int):
    images_dir = os.path.join(output_dir, "images", "train")
    labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i in range(num_frames):
        fn = f"dummy_{i:05d}"

        # Random grayscale noise image — content doesn't matter, this is
        # only exercising the data-loading/training code path.
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir, fn + ".jpg"), img)

        lines = []
        # ~80% of frames get 1-2 fake hand boxes, ~20% are "no hand" negatives
        # (matches the real pipeline's mix of positive/negative samples).
        if random.random() < 0.8:
            num_hands = random.choice([1, 2])
            used_classes = random.sample([0, 1], k=num_hands)  # 0=left, 1=right
            for cls in used_classes:
                cx = random.uniform(0.2, 0.8)
                cy = random.uniform(0.2, 0.8)
                w = random.uniform(0.1, 0.3)
                h = random.uniform(0.1, 0.3)
                lines.append(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")

        with open(os.path.join(labels_dir, fn + ".txt"), "w") as f:
            f.write("\n".join(lines))

        if i % 50 == 0:
            print(f"generated {i}/{num_frames}")

    print(f"Done. Dummy dataset at {output_dir} ({num_frames} frames).")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "./dummy_egohands"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    make_dummy_dataset(out_dir, n)
