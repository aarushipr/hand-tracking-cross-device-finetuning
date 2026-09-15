hot3d_dataset_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset"
# The CLONED HOT3D CODE REPO (has data_loaders), NOT the dataset tree.
# Must match the keypoint side's value in keypoint/local_config_cluster.py.
hot3d_repo_root = "/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d"

# On-disk cache for the sample index; building it opens every .vrs over network storage.
# Scratch, not the repo. Separate from the keypoint cache: different entries and keys.
hot3d_index_cache_dir = "/storage/user/praa/scratch/hot3d_detection_index"

# Phanesim: the second, fully-synthetic dataset used for fine-tuning phase 2.
# Mirrored into personal storage 2026-09-08; both roots are pooled.
phanesim_dataset_roots = [
    "/storage/user/praa/phanesim_dataset/dataset",
    "/storage/user/praa/phanesim_dataset/dataset2",
    "/storage/user/praa/phanesim_dataset/dataset3",
]
