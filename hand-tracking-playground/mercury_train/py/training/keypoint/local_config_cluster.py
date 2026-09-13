# local_config.py for the SLURM cluster (working-code checkout).
#
# IMPORTANT: On the cluster, copy this file over local_config.py:
#   cp local_config_cluster.py local_config.py
# local_config.py itself is gitignored (per-machine, never synced by git) —
# that's why this template exists and is committed instead.
#
# Reuses the same synthetic data path as the other (root) mercury_train
# checkout on this cluster — no need to duplicate the data on disk. Adjust
# if you generate working-code-specific data into a different folder.

real_datasets_basepath = "/storage/user/praa/synth_hands_output"
artificial_dataset_path = "/storage/user/praa/synth_hands_output"
indoor_backgrounds_path = "/storage/user/praa/synth_hands_output"

hot3d_repo_root = "/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d"
hot3d_dataset_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset"
hot3d_object_library_path = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/assets"
# On-disk cache for HOT3DKeypointDataset's sample index. Building the index
# walks every sequence's .vrs over network storage, which is I/O-latency
# bound and slow; caching it makes every run after the first start in
# seconds. Scratch, not the repo -- it is regenerable output, not source.
hot3d_index_cache_dir = "/storage/user/praa/scratch/hot3d_keypoint_index"

# Phanesim (see PhanesimKeypointDataset.py) -- phase-2 KeyNet fine-tuning,
# same two pooled dataset roots as detection's phase 2.
phanesim_dataset_roots = [
    "/storage/user/praa/phanesim_dataset/dataset",
    "/storage/user/praa/phanesim_dataset/dataset2",
    "/storage/user/praa/phanesim_dataset/dataset3",
]
