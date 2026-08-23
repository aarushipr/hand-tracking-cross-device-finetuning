import os
import random

# give sequence id
def participant_id_of(seq_dir):
    return os.path.basename(os.path.normpath(seq_dir)).split("_")[0]

# hot3d's own official test split, no ground truth available
NO_GT_TEST_PARTICIPANTS = {"P0004", "P0005", "P0006", "P0008", "P0016", "P0020"}
# this is our test set because its captured on both aria and quest
CROSS_DEVICE_HELD_OUT_PARTICIPANTS = {"P0002", "P0003", "P0010"}

def filter_sequence_dirs(all_seq_dirs, split):
    usable = [d for d in all_seq_dirs if participant_id_of(d) not in NO_GT_TEST_PARTICIPANTS]

    if split == "all_labeled":
        return usable
    if split == "cross_device_test":
        return [d for d in usable if participant_id_of(d) in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]
    if split == "train":
        return [d for d in usable if participant_id_of(d) not in CROSS_DEVICE_HELD_OUT_PARTICIPANTS]

    raise ValueError(f"unknown split {split!r}")
            
def split_train_val(seq_dirs, val_fraction=0.1, seed=0):
    seq_dirs = sorted(seq_dirs)
    rng = random.Random(seed)
    rng.shuffle(seq_dirs)

    val_size = int(len(seq_dirs) * val_fraction)
    val_dirs = seq_dirs[:val_size]
    train_dirs = seq_dirs[val_size:]
    return train_dirs, val_dirs

def list_sequence_dirs(dataset_root, split):
    if not dataset_root or not os.path.isdir(dataset_root):
        return []
    all_dirs = sorted(
        os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("P0"))
    return filter_sequence_dirs(all_dirs, split)

if __name__ == "__main__":
    fake_dirs = [
        "/x/P0001_aaa", "/x/P0002_bbb", "/x/P0003_ccc",
        "/x/P0004_ddd", "/x/P0010_eee", "/x/P0099_fff",
    ]
    print("train:", filter_sequence_dirs(fake_dirs, "train"))
    print("cross_device_test:", filter_sequence_dirs(fake_dirs, "cross_device_test"))
    print("all_labeled:", filter_sequence_dirs(fake_dirs, "all_labeled"))