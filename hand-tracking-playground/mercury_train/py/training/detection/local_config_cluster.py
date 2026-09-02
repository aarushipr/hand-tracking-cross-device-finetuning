hot3d_dataset_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d/dataset"
hot3d_repo_root = "/storage/user/praa/hot3d_full_setup/hot3d/hot3d"

# On-disk cache for HOT3DVRSDetectionDataset's sample index. Building the
# index opens every sequence's .vrs over network storage, which is
# I/O-latency bound and slow; caching it makes every run after the first
# start in seconds. Scratch, not the repo -- it is regenerable output, not
# source. Kept separate from the keypoint cache: the two indices hold
# different entries and use different cache keys.
hot3d_index_cache_dir = "/storage/user/praa/scratch/hot3d_detection_index"
