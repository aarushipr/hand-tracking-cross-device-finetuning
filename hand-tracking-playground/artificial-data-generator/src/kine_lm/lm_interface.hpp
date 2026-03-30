// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Interface for Levenberg-Marquardt kinematic optimizer
 * @author Moses Turner <moses@collabora.com>
 * @ingroup tracking
 */
#pragma once
#include <array>
#include "xrt/xrt_defines.h"
#include "util/u_logging.h"
// #include "lm_defines.hpp"
#include "../kine_common.hpp"
#include "lm_defines.hpp"

namespace xrt::tracking::hand::mercury::lm {

// Yes, this is a weird in-between-C-and-C++ API. Fight me, I like it this way.

// Opaque struct.
struct KinematicHandLM;

// clang-format off

// These were made by hand statically.

enum tip_touch_combos {
  IN_TH, //
  MI_TH, MI_IN, //
  RI_TH, RI_IN, RI_MI, //
  LI_TH, LI_IN, LI_MI, LI_RI, //
  NUM_TIP_TOUCH_COMBOS
};

static constexpr size_t 
ttc_elements[10][2] = {
  {1, 0}, //
  {2, 0}, {2, 1}, //
  {3, 0}, {3, 1}, {3, 2}, //
  {4, 0}, {3, 1}, {3, 2}, {3, 4}, //
};

// clang-format on

struct value_factor
{
	HandScalar value;
	HandScalar factor;
};

struct curl_desires
{
	value_factor c[5];
};

struct splay_desires
{
	value_factor s[5];
};

struct tip_touch_desires
{
	value_factor t[NUM_TIP_TOUCH_COMBOS];
};

struct optimizer_input
{
	curl_desires curls = {};
	splay_desires splays = {};
	tip_touch_desires tip_touches = {};
	std::array<xrt_pose, 26> target_pose = {};
	bool use_target_pose = {};
};

struct hand_proportions
{
	HandScalar hand_size;
	// Quat<HandScalar> thumb_root_orientation;
	Quat<HandScalar> metacarpal_root_orientations[5];
	Translations55<HandScalar> rel_translations;
};



// Constructor
void
optimizer_create(xrt_pose left_in_right,
                 bool is_right,
                 u_logging_level log_level,
                 struct hand_proportions proportions,
                 HandLimit joint_limits,
                 KinematicHandLM **out_kinematic_hand);

/*!
 * The main tracking code calls this function with some 2D(ish) camera observations of the hand, and this function
 * calculates a good 3D hand pose and writes it to out_viz_hand.
 *
 * @param observation The observation of the hand joints. Warning, this function will mutate the observation
 * unpredictably. Keep a copy of it if you need it after.
 * @param hand_was_untracked_last_frame: If the hand was untracked last frame (it was out of view, obscured, ML models
 * failed, etc.) - if it was, we don't want to enforce temporal consistency because we have no good previous hand state
 * with which to do that.
 * @param optimize_hand_size: Whether or not it's allowed to tweak the hand size - when we're calibrating the user's
 * hand size, we want to do that; afterwards we don't want to waste the compute.
 * @param target_hand_size: The hand size we want it to get close to
 * @param hand_size_err_mul: A multiplier to help determine how close it has to get to that hand size
 * @param[out] out_hand: The xrt_hand_joint_set to output its result to
 * @param[out] out_hand_size: The hand size it ended up at
 * @param[out] out_reprojection_error: The reprojection error it ended up at
 */

void
optimizer_run(KinematicHandLM *hand, optimizer_input &input, xrt_hand_joint_set &out_hand);

// Destructor
void
optimizer_destroy(KinematicHandLM **hand);

struct hand_proportions
default_hand_proportions();

} // namespace xrt::tracking::hand::mercury::lm
