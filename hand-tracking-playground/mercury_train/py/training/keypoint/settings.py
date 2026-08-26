# multiplier shows how much they contribute to total loss

# hand pose predicted from previous frame
using_pose_predicted_input = True

# how much depth prediction contributes to total loss
depth_loss_mul = 0.03

# confidence yes/no if there is a hand in the image
#
# Zero for the HOT3D fine-tuning of this thesis. The HOT3D keypoint training
# set contains only real, visible hands -- every sample is a positive,
# because a hand that is not visible enough simply has no valid crop and is
# excluded at index time. Training the existence output on positives alone
# would teach it to answer "yes" unconditionally, which is strictly worse
# than the Monado initialisation it starts from.
#
# Hand presence is DetNet's responsibility, not KeyNet's: KeyNet only ever
# receives an already-cropped hand region, and HOT3DVRSDetectionDataset does
# keep hand-free frames as exists=0 negatives for DetNet. So KeyNet's
# existence head is deliberately left at its pretrained value here, and
# Chapter 4 states that as a scope decision.
#
# Restore 0.0005 if KeyNet is ever trained on a set that includes real
# negative crops (crops of actual frames containing no hand -- not blank
# images, which teach nothing).
existence_loss_mul = 0.0

# which direction elbow is pointing
elbow_loss_mul = 0.001

# how curled the fingers are
curls_loss_mul = 0.00001

# confidence (variance) of curl prediction
curl_min_variance = 0.01
