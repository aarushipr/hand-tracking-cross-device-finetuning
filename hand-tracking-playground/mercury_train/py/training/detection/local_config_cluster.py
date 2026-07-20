# local_config.py for the SLURM cluster (working-code checkout).
#
# IMPORTANT: On the cluster, copy this file over local_config.py:
#   cp local_config_cluster.py local_config.py
# local_config.py itself is gitignored (per-machine, never synced by git) —
# that's why this template exists and is committed instead.
#
# DetNet's primary training source: same Blender-rendered synthetic
# sequences the keypoint pipeline already uses (SyntheticDetectionDataset
# derives bboxes from the 3D hand joints via pinhole projection). This
# should point at the same directory as keypoint's
# local_config.artificial_dataset_path -- it's the same generator output.
artificial_dataset_path = "/storage/user/praa/synth_hands_output"

# HMDHandRects / EgoHands / EpicKitchens are now val/test-only, not
# training blockers -- CombinedDataset.py skips any of these gracefully if
# missing/unconfigured. Fill in real paths as you locate them (ask your
# supervisor for HMDHandRects specifically, it has no public source), but
# training will run fine without them thanks to the synthetic source above.
hmdhandrects_location = "/storage/user/praa/REPLACE_ME_HMDHandRects/"
egohands_convert = "/storage/user/praa/REPLACE_ME_EgoHands/"

kitchens_images = "/storage/user/praa/REPLACE_ME_EPIC-KITCHENS"
kitchens_annotations = "/storage/user/praa/REPLACE_ME_kitchen_labels/"
kitchens_only_1st_sequence = True  # Set this to False if you're training a production model!

batch_size = 16
