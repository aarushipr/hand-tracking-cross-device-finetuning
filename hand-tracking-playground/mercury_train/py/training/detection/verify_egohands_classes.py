"""
Visual check for EgoHands class-index assumptions.

DarknetDataset.py assumes class 0/1 = the camera wearer's own left/right
hand (used for training), and class 2/3 = someone else's hand across the
table (skipped). That matches EgoHands' own myleft/myright/yourleft/
yourright categories conceptually, but no classes.txt/.labels file shipped
with this particular Roboflow export to confirm the index order matches.

This draws each labeled box on its image, colored and labeled by raw class
index, so it's possible to eyeball whether 0/1-colored boxes consistently
land on the hand that looks like it belongs to the person holding the
camera (usually entering the frame from the bottom/nearest edge, often
larger/closer) versus 2/3 landing on a hand further away / entering from a
different side (e.g. across a table).

Usage:
    python verify_egohands_classes.py \
        --root /storage/user/praa/egohands_raw \
        --num-samples 12 \
        --out-dir /storage/user/praa/egohands_raw/class_check
"""
import argparse
import os
import random

import cv2

# class -> (color BGR, label) per DarknetDataset.py's assumed meaning
CLASS_INFO = {
    0: ((0, 255, 0), "0=my_left?"),
    1: ((0, 255, 255), "1=my_right?"),
    2: ((0, 0, 255), "2=your_left?(skip)"),
    3: ((255, 0, 255), "3=your_right?(skip)"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels_dir = os.path.join(args.root, "labels", args.split)
    images_dir = os.path.join(args.root, "images", args.split)

    label_files = sorted(os.listdir(labels_dir))
    random.seed(0)
    sample = random.sample(label_files, min(args.num_samples, len(label_files)))

    for fn in sample:
        base = fn[:-4]
        img_path = os.path.join(images_dir, base + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: couldn't load {img_path}")
            continue
        h, w = img.shape[:2]

        with open(os.path.join(labels_dir, fn)) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                x0 = int((cx - bw / 2) * w)
                y0 = int((cy - bh / 2) * h)
                x1 = int((cx + bw / 2) * w)
                y1 = int((cy + bh / 2) * h)
                color, label = CLASS_INFO.get(cls, ((255, 255, 255), f"{cls}=unknown"))
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                cv2.putText(img, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(args.out_dir, base + "_check.jpg")
        cv2.imwrite(out_path, img)
        print(f"wrote {out_path}")

    print(f"\nDone. Look at {args.out_dir} -- for each image, do the green/yellow "
          f"(0/1) boxes consistently land on the hand that looks like it belongs "
          f"to whoever is holding the camera (usually nearest/largest, entering "
          f"from the bottom of frame), while red/magenta (2/3) land on a hand "
          f"further away? If it's reversed or mixed, DarknetDataset.py's class "
          f"0/1 vs 2/3 handling needs to flip.")


if __name__ == "__main__":
    main()
