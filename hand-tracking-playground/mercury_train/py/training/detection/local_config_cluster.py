# local_config.py for the SLURM cluster (working-code checkout).
#
# IMPORTANT: On the cluster, copy this file over local_config.py:
#   cp local_config_cluster.py local_config.py
# local_config.py itself is gitignored (per-machine, never synced by git) —
# that's why this template exists and is committed instead.
#
# Unlike the keypoint pipeline, nobody has confirmed where HMDHandRects /
# EgoHands / EpicKitchens actually live on this cluster yet. Run
# `ls /storage/user/praa/` after SSHing in and fill in the real paths below
# before submitting detection.sbatch — it will fail immediately on dataset
# load otherwise.

hmdhandrects_location = "/storage/user/praa/REPLACE_ME_HMDHandRects/"
egohands_convert = "/storage/user/praa/REPLACE_ME_EgoHands/"

kitchens_images = "/storage/user/praa/REPLACE_ME_EPIC-KITCHENS"
kitchens_annotations = "/storage/user/praa/REPLACE_ME_kitchen_labels/"
kitchens_only_1st_sequence = True  # Set this to False if you're training a production model!

batch_size = 16
