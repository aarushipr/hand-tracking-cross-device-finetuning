# local_config.py for the SLURM cluster (working-code checkout).
# On the cluster: cp local_config_cluster.py local_config.py
# local_config.py is gitignored and per-machine, which is why this template is committed.
# Reuses the other checkout's synthetic data path; adjust if you generate your own.

real_datasets_basepath = "/storage/user/praa/synth_hands_output"
artificial_dataset_path = "/storage/user/praa/synth_hands_output"
indoor_backgrounds_path = "/storage/user/praa/synth_hands_output"

hot3d_repo_root = "/storage/user/praa/hot3d_full_setup/hot3d_repo/hot3d"
hot3d_dataset_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset"
hot3d_object_library_path = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset/assets"
# On-disk cache for the sample index; building it walks every .vrs over network storage.
# Scratch, not the repo: regenerable output, not source.
hot3d_index_cache_dir = "/storage/user/praa/scratch/hot3d_keypoint_index"

# Phanesim phase-2 KeyNet fine-tuning; same two pooled roots as detection's phase 2.
phanesim_dataset_roots = [
    "/storage/user/praa/phanesim_dataset/dataset",
    "/storage/user/praa/phanesim_dataset/dataset2",
    "/storage/user/praa/phanesim_dataset/dataset3",
]
