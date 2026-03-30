import os
import subprocess


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


ann_root = "/excluded_epics_from_sync/kitchen_labels/"
img_root = "/media/moses/TRAINDATA1/EPIC-KITCHENS/"
new_img_root = "/media/moses/epickitchens/EPIC-KITCHENS/"

subjects = sorted(os.listdir(ann_root))

ts = 0

for subject_idx, subject in enumerate(subjects):
        print(f"Loading subject idx {subject_idx}/{len(subjects)}")
        for sequence_idx, subject_sequence in enumerate(sorted(os.listdir(os.path.join(ann_root, subject)))):
          img_folder = os.path.join(
                img_root, subject, "rgb_frames", subject_sequence)
          # os.system(f"du -sh {img_folder}")
          ts += get_size(img_folder)
          print(ts/1e+9)