# multiplier shows how much they contribute to total loss

# hand pose predicted from previous frame
using_pose_predicted_input = True

# how much depth prediction contributes to total loss
depth_loss_mul = 0.03

# confidence yes/no if there is a hand in the image
existence_loss_mul = 0.0005

# which direction elbow is pointing
elbow_loss_mul = 0.001

# how curled the fingers are
curls_loss_mul = 0.00001

# confidence (variance) of curl prediction
curl_min_variance = 0.01
