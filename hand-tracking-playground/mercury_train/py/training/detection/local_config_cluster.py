hot3d_dataset_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset"
# The CLONED HOT3D CODE REPO (contains the data_loaders package), NOT the
# dataset tree. These are two different directories under hot3d_full_setup
# and this key pointed at the dataset one until 2026-09-02, which made
# every detection run fail at import with "No module named data_loaders".
# Nobody hit it because DetNet had never been run. Must match the keypoint
# side's value in py/training/keypoint/local_config_cluster.py.
hot3d_repo_root = "/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d"

# On-disk cache for HOT3DVRSDetectionDataset's sample index. Building the
# index opens every sequence's .vrs over network storage, which is
# I/O-latency bound and slow; caching it makes every run after the first
# start in seconds. Scratch, not the repo -- it is regenerable output, not
# source. Kept separate from the keypoint cache: the two indices hold
# different entries and use different cache keys.
hot3d_index_cache_dir = "/storage/user/praa/scratch/hot3d_detection_index"

# Phanesim: second, fully-synthetic dataset (built by wany, another student
# of the same supervisor) used for DetNet/KeyNet fine-tuning phase 2, on top
# of the HOT3D fine-tuning above. Mirrored from
# /storage/group/dataset_mirrors/01_incoming/phanesim20260908/ into personal
# storage on 2026-09-08. Both roots are pooled -- see thesis chat log for the
# "phanesim only, no HOT3D mixed in" decision.
phanesim_dataset_roots = [
    "/storage/user/praa/phanesim_dataset/dataset",
    "/storage/user/praa/phanesim_dataset/dataset2",
    "/storage/user/praa/phanesim_dataset/dataset3",
]
