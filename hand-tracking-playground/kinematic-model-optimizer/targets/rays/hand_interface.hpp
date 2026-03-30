#pragma once
#include "xrt/xrt_defines.h"
#include "stereokit.h"
#include "defines.hpp"

#define dist_between_cameras 0.139 // Good enough
#define num_out_frames 2000

struct one_frame_one_view
{
	xrt_vec3 rays[21];
	HandScalar confidences[21];
};

struct one_frame_input
{
	one_frame_one_view views[2];
};

struct hand_output
{
	sk::pose_t wrist;
	// finger, bone
	sk::pose_t fingers[5][5];
};


// Opaque
struct LMKinematicHand;


/*
struct LMKinematicHandStatus{
  bool accepts_single_view
}

struct LMKinematicHandStatus hand_status(LMKinematicHand *hand);
*/


void
create_kinematic_hand(HandScalar camera_baseline, LMKinematicHand **out_kinematic_hand);

double
push_frame(LMKinematicHand *hand, one_frame_input &observation, struct hand_output &out_viz_hand);

void
give_identity_hand(LMKinematicHand *hand, struct hand_output &out_viz_hand);

void
destroy_kinematic_hand(LMKinematicHand **hand);

void
eval_to_viz_hand_2(LMKinematicHand *hand_ptr, hand_output &out_viz_hand);

