# multiplier shows how much they contribute to total loss

# hand pose predicted from previous frame
using_pose_predicted_input = True

# how much depth prediction contributes to total loss
depth_loss_mul = 0.03

# confidence yes/no if there is a hand in the image
# Zero: the HOT3D keypoint set is all positives, so this head would learn "yes" always.
# Presence is DetNet's job; Chapter 4 states it as a scope decision.
# Restore 0.0005 if KeyNet is ever trained on real negative crops.
existence_loss_mul = 0.0

# which direction elbow is pointing
# Zero because HOT3D has no body pose, so the dataset zeros mean "no supervision".
# kpest_trainer.py derives has_elbow_curls from has_depth, which HOT3D breaks.
# Left at 0.001 the head would be driven toward always predicting zero.
elbow_loss_mul = 0.0

# how curled the fingers are
# Zero for the same reason as elbow_loss_mul: the zeros mean "unavailable", not zero.
curls_loss_mul = 0.0

# confidence (variance) of curl prediction
curl_min_variance = 0.01
