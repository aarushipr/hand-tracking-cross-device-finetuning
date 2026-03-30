#include "xrt/xrt_defines.h"

#include "pose_csv.hpp"


struct fingerpose_creator;

void
finger_pose_step(fingerpose_creator *cr, xrt_hand_joint_set *out_optimized_hand, TrajectorySample<26> *out_sample_used);

fingerpose_creator *
create_finger_pose(TrajectoryReader<26> *finger_mocap, const char *proportions_json, int64_t start_ts, float frametime);

void
finger_pose_delete(fingerpose_creator **cr);
