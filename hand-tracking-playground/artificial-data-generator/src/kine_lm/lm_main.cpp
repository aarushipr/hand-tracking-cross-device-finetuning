// Copyright 2022, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Levenberg-Marquardt kinematic optimizer
 * @author Moses Turner <moses@collabora.com>
 * @author Charlton Rodda <charlton.rodda@collabora.com>
 * @ingroup tracking
 */

#include "math/m_api.h"
#include "math/m_vec3.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_trace_marker.h"

#include "tinyceres/tiny_solver.hpp"
#include "tinyceres/tiny_solver_autodiff_function.hpp"
#include "lm_rotations.inl"

#include <fenv.h>
#include <iostream>
#include <cmath>
#include <random>
#include "lm_interface.hpp"
#include "lm_optimizer_params_packer.inl"
#include "lm_defines.hpp"
#include "xrt/xrt_defines.h"
#include "util/u_hand_tracking.h"

/*

Some notes:
Everything templated with <typename T> is basically just a scalar template, usually taking float or ceres::Jet<float, N>

*/

namespace xrt::tracking::hand::mercury::lm {

template <typename T> struct StereographicObservation
{
	// T obs[kNumNNJoints][2];
	Vec2<T> obs[kNumNNJoints];
};

struct KinematicHandLM
{
	// optimizer_input &input;
	bool first_frame = true;
	bool is_right = false;

	HandScalar target_hand_size;
	HandScalar hand_size_err_mul;
	u_logging_level log_level;
	struct xrt_hand_joint_set set_with_just_sizes;

	struct hand_proportions proportions;
	HandLimit the_limit;

	optimizer_input *input;
	// std::array<xrt_pose, 26> target_joints;
	// float target_joint_weight;

	// Quat<HandScalar> last_frame_pre_rotation;
	OptimizerHand<HandScalar> last_frame;

	Eigen::Matrix<HandScalar, calc_input_size(false), 1> TinyOptimizerInput;
};



struct CostFunctor
{
	KinematicHandLM &parent;
	size_t num_residuals_;

	template <typename T>
	bool
	operator()(const T *const x, T *residual) const;

	CostFunctor(KinematicHandLM &in_last_hand, size_t const &num_residuals)
	    : parent(in_last_hand), num_residuals_(num_residuals)
	{}



	size_t
	NumResiduals() const
	{
		return num_residuals_;
	}
};

template <typename T>
static inline void
eval_hand_set_rel_translations(struct KinematicHandLM &lm, Translations55<T> &rel_translations)
{
	// Basically, we're walking up rel_translations, writing strictly sequentially. Hopefully this is fast.

	for (int f = 0; f < 5; f++) {
		for (int j = 0; j < 5; j++) {
			rel_translations.t[f][j] = lm.proportions.rel_translations.t[f][j];
		}
	}
}



template <typename T>
inline void
eval_hand_set_rel_orientations(struct KinematicHandLM &lm,
                               const OptimizerHand<T> &opt,
                               Orientations54<T> &rel_orientations)
{

	for (int i = 0; i < 5; i++) {
		rel_orientations.q[i][0] = lm.proportions.metacarpal_root_orientations[i];
	}

	rel_orientations.q[0][1] = Quat<T>::Identity();

	// Thumb MCP orientation
	SwingTwistToQuaternion(opt.thumb.metacarpal.swing, //
	                       opt.thumb.metacarpal.twist, //
	                       rel_orientations.q[0][2]);

	// Thumb curls
	CurlToQuaternion(opt.thumb.rots[0], rel_orientations.q[0][3]);
	CurlToQuaternion(opt.thumb.rots[1], rel_orientations.q[0][4]);

	// Finger orientations
	for (int i = 0; i < 4; i++) {
		// std::cout << "SWINGX FINGER" << i+1 << opt.finger[i].metacarpal.swing.x << " " <<
		// opt.finger[i].metacarpal.swing.y << std::endl;
		SwingTwistToQuaternion(opt.finger[i].metacarpal.swing, //
		                       opt.finger[i].metacarpal.twist, //
		                       rel_orientations.q[i + 1][1]);

		SwingToQuaternion(opt.finger[i].proximal_swing, //
		                  rel_orientations.q[i + 1][2]);

		CurlToQuaternion(opt.finger[i].rots[0], rel_orientations.q[i + 1][3]);
		CurlToQuaternion(opt.finger[i].rots[1], rel_orientations.q[i + 1][4]);
	}

	// for (int f = 0; f < 5; f++) {
	// 	for (int j = 0; j < 5; j++) {
	// 		std::cout << "f" << f << " j" << j << " " << rel_orientations.q[f][j].w<< " " <<
	// rel_orientations.q[f][j].x<< " " << rel_orientations.q[f][j].y<< " " << rel_orientations.q[f][j].z <<
	// std::endl;;
	// 	}
	// }
}


struct hand_proportions
default_hand_proportions()
{
	// Thumb MCP hidden orientation
	struct hand_proportions prop = {};
	for (int i = 0; i < 5; i++) {
		prop.metacarpal_root_orientations[i].w = 1.0;
		prop.metacarpal_root_orientations[i].x = 0.0;
		prop.metacarpal_root_orientations[i].y = 0.0;
		prop.metacarpal_root_orientations[i].z = 0.0;
	}
#if 0
	Vec2<T> mcp_root_swing;

	mcp_root_swing.x = rad<T>((T)(-10));
	mcp_root_swing.y = rad<T>((T)(-40));

	T mcp_root_twist = rad<T>((T)(-80));

	SwingTwistToQuaternion(mcp_root_swing, mcp_root_twist, rel_orientations.q[0][0]);

	std::cout << "\n\n\n\nHIDDEN ORIENTATION\n";
	std::cout << std::setprecision(100);
	std::cout << rel_orientations.q[0][0].w << std::endl;
	std::cout << rel_orientations.q[0][0].x << std::endl;
	std::cout << rel_orientations.q[0][0].y << std::endl;
	std::cout << rel_orientations.q[0][0].z << std::endl;
#else
	// This should be exactly equivalent to the above
	prop.metacarpal_root_orientations[0].w = 0.716990172863006591796875;
	prop.metacarpal_root_orientations[0].x = 0.1541481912136077880859375;
	prop.metacarpal_root_orientations[0].y = -0.31655871868133544921875;
	prop.metacarpal_root_orientations[0].z = -0.6016261577606201171875;
#endif

	Translations55<HandScalar> &out_rel_translations = prop.rel_translations;

	// Thumb metacarpal translation.
	out_rel_translations.t[0][0] = {(HandScalar)0.33097, HandScalar(-0.1), (HandScalar)-0.25968};

	// Comes after the invisible joint.
	out_rel_translations.t[0][1] = {HandScalar(0), HandScalar(0), HandScalar(0)};
	// prox, distal, tip
	out_rel_translations.t[0][2] = {HandScalar(0), HandScalar(0), HandScalar(-0.389626)};
	out_rel_translations.t[0][3] = {HandScalar(0), HandScalar(0), HandScalar(-0.311176)};
	out_rel_translations.t[0][4] = {HandScalar(0), HandScalar(0), (HandScalar)-0.232195};

	// What's the best place to put this? Here works, but is there somewhere we could put it where it gets accessed
	// faster?
	HandScalar finger_joint_lengths[4][4] = {
	    {
	        HandScalar(-0.66),
	        HandScalar(-0.365719),
	        HandScalar(-0.231581),
	        HandScalar(-0.201790),
	    },
	    {
	        HandScalar(-0.645),
	        HandScalar(-0.404486),
	        HandScalar(-0.247749),
	        HandScalar(-0.210121),
	    },
	    {
	        HandScalar(-0.58),
	        HandScalar(-0.365639),
	        HandScalar(-0.225666),
	        HandScalar(-0.187089),
	    },
	    {
	        HandScalar(-0.52),
	        HandScalar(-0.278197),
	        HandScalar(-0.176178),
	        HandScalar(-0.157566),
	    },
	};

	// Index metacarpal
	out_rel_translations.t[1][0] = {HandScalar(0.16926), HandScalar(0), HandScalar(-0.34437)};
	// Middle
	out_rel_translations.t[2][0] = {HandScalar(0.034639), HandScalar(0.01), HandScalar(-0.35573)};
	// Ring
	out_rel_translations.t[3][0] = {HandScalar(-0.063625), HandScalar(0.005), HandScalar(-0.34164)};
	// Little
	out_rel_translations.t[4][0] = {HandScalar(-0.1509), HandScalar(-0.005), HandScalar(-0.30373)};

	// Index to little finger
	for (int finger = 0; finger < 4; finger++) {
		for (int i = 0; i < 4; i++) {
			int bone = i + 1;
			out_rel_translations.t[finger + 1][bone] = {HandScalar(0), HandScalar(0),
			                                            HandScalar(finger_joint_lengths[finger][i])};
		}
	}
	return prop;
}



template <typename T>
void
eval_hand_with_orientation(const OptimizerHand<T> &opt,
                           KinematicHandLM &state,
                           Translations55<T> &translations_absolute,
                           Orientations54<T> &orientations_absolute)

{
	XRT_TRACE_MARKER();

	bool is_right = state.is_right;


	Translations55<T> rel_translations; //[kNumFingers][kNumJointsInFinger];
	Orientations54<T> rel_orientations; //[kNumFingers][kNumOrientationsInFinger];

	eval_hand_set_rel_orientations(state, opt, rel_orientations);

	eval_hand_set_rel_translations(state, rel_translations);

	Quat<T> orientation_root = Quat<T>::Identity();
	Vec3<T> position_root = Vec3<T>::Zero();

	// std::cout << "rt " << orientation_root.w << " " << orientation_root.x << " " << orientation_root.y << " "
	//           << orientation_root.z << " " << std::endl;

	// Quat<T> post_orientation_quat;

	// AngleAxisToQuaternion(opt.wrist_post_orientation_aax, post_orientation_quat);

	// QuaternionProduct(opt.wrist_pre_orientation_quat, post_orientation_quat, orientation_root);

	// Get each joint's tracking-relative orientation by rotating its parent-relative orientation by the
	// tracking-relative orientation of its parent.
	for (size_t finger = 0; finger < kNumFingers; finger++) {
		Quat<T> *last_orientation = &orientation_root;
		for (size_t bone = 0; bone < kNumJointsInFinger; bone++) {
			Quat<T> &out_orientation = orientations_absolute.q[finger][bone];
			Quat<T> &rel_orientation = rel_orientations.q[finger][bone];

			QuaternionProduct(*last_orientation, rel_orientation, out_orientation);
			last_orientation = &out_orientation;
		}
	}

	// Get each joint's tracking-relative position by rotating its parent-relative translation by the
	// tracking-relative orientation of its parent, then adding that to its parent's tracking-relative position.
	for (size_t finger = 0; finger < kNumFingers; finger++) {
		const Vec3<T> *last_translation = &position_root;
		const Quat<T> *last_orientation = &orientation_root;
		for (size_t bone = 0; bone < kNumJointsInFinger; bone++) {
			Vec3<T> &out_translation = translations_absolute.t[finger][bone];
			Vec3<T> &rel_translation = rel_translations.t[finger][bone];

			UnitQuaternionRotateAndScalePoint(*last_orientation, rel_translation, opt.hand_size,
			                                  out_translation);

			// If this is a right hand, mirror it.
			if (is_right) {
				out_translation.x *= -1;
			}

			out_translation.x += last_translation->x;
			out_translation.y += last_translation->y;
			out_translation.z += last_translation->z;

			// Next iteration, the orientation to rotate by should be the tracking-relative orientation of
			// this joint.

			// If bone < 4 so we don't go over the end of orientations_absolute. I hope this gets optimized
			// out anyway.
			if (bone < 4) {
				last_orientation = &orientations_absolute.q[finger][bone + 1];
				// Ditto for translation
				last_translation = &out_translation;
			}
		}
	}
}

template <typename T>
void
computeResidualStability_Finger(const OptimizerFinger<T> &finger,
                                const OptimizerFinger<HandScalar> &finger_last,
                                ResidualHelper<T> &helper)
{
	helper.AddValue((finger.metacarpal.swing.x - finger_last.metacarpal.swing.x) * kStabilityFingerMCPSwing);

	helper.AddValue((finger.metacarpal.swing.y - finger_last.metacarpal.swing.y) * kStabilityFingerMCPSwing);



	helper.AddValue((finger.metacarpal.twist - finger_last.metacarpal.twist) * kStabilityFingerMCPTwist);



	helper.AddValue((finger.proximal_swing.x - finger_last.proximal_swing.x) * kStabilityFingerPXMSwingX);
	helper.AddValue((finger.proximal_swing.y - finger_last.proximal_swing.y) * kStabilityFingerPXMSwingY);

	helper.AddValue((finger.rots[0] - finger_last.rots[0]) * kStabilityCurlRoot);
	helper.AddValue((finger.rots[1] - finger_last.rots[1]) * kStabilityCurlRoot);

	// #ifdef USE_HAND_PLAUSIBILITY
	// 	if (finger.rots[0] < finger.rots[1]) {
	// 		helper.AddValue((finger.rots[0] - finger.rots[1]) * kPlausibilityCurlSimilarityHard);
	// 	} else {
	// 		helper.AddValue((finger.rots[0] - finger.rots[1]) * kPlausibilityCurlSimilaritySoft);
	// 	}
	// #endif
}

template <typename T>
T
getFingerOverallCurl(const OptimizerHand<T> &hand, size_t idx)
{
	return hand.finger[idx].metacarpal.swing.x + hand.finger[idx].proximal_swing.x + hand.finger[idx].rots[0] + hand.finger[idx].rots[1];
}

template <typename T>
void
computeResidualStability(const OptimizerHand<T> &hand,
                         const OptimizerHand<HandScalar> &last_hand,
                         KinematicHandLM &state,
                         ResidualHelper<T> &helper)
{

	helper.AddValue((hand.thumb.metacarpal.swing.x - last_hand.thumb.metacarpal.swing.x) * kStabilityThumbMCPSwing);
	helper.AddValue((hand.thumb.metacarpal.swing.y - last_hand.thumb.metacarpal.swing.y) * kStabilityThumbMCPSwing);
	helper.AddValue((hand.thumb.metacarpal.twist - last_hand.thumb.metacarpal.twist) * kStabilityThumbMCPTwist);

	helper.AddValue((hand.thumb.rots[0] - last_hand.thumb.rots[0]) * kStabilityCurlRoot);
	helper.AddValue((hand.thumb.rots[1] - last_hand.thumb.rots[1]) * kStabilityCurlRoot);
#ifdef USE_HAND_PLAUSIBILITY
		helper.AddValue((getFingerOverallCurl(hand, 0) - getFingerOverallCurl(hand, 1)) *
	                kPlausibilityCurlSimilarity_IndexMiddle);
	helper.AddValue((getFingerOverallCurl(hand, 1) - getFingerOverallCurl(hand, 2)) *
	                kPlausibilityCurlSimilarity_MiddleRing);
	helper.AddValue((getFingerOverallCurl(hand, 2) - getFingerOverallCurl(hand, 3)) *
	                kPlausibilityCurlSimilarity_RingLittle);
#endif


	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		const OptimizerFinger<HandScalar> &finger_last = last_hand.finger[finger_idx];

		const OptimizerFinger<T> &finger = hand.finger[finger_idx];

		computeResidualStability_Finger(finger, finger_last, helper);
	}
}

template <typename T>
void
computeResidualCurlSway(const OptimizerHand<T> &hand, KinematicHandLM &state, ResidualHelper<T> &helper)
{
	optimizer_input &input = *state.input;

	T total_curls[5];

	total_curls[0] = hand.thumb.metacarpal.swing.x + hand.thumb.rots[0] + hand.thumb.rots[1];


	// U_LOG_E("I CRY BEFORE Loop");

	for (int i = 0; i < 4; i++) {
		// U_LOG_E("I CRY inside loop");
		total_curls[i + 1] = hand.finger[i].metacarpal.swing.x + hand.finger[i].proximal_swing.x +
		                     hand.finger[i].rots[0] + hand.finger[i].rots[1];
	}
	// U_LOG_E("I CRY AFTER Loop");

	for (int i = 0; i < 5; i++) {
		helper.AddValue((input.curls.c[i].value - total_curls[i]) * input.curls.c[i].factor);
	}

	T total_splays[5];

	total_splays[0] = hand.thumb.metacarpal.swing.y;
	for (int i = 0; i < 4; i++) {
		// U_LOG_E("I CRY inside loop");
		total_splays[i + 1] = hand.finger[i].proximal_swing.y;
	}

	for (int i = 0; i < 5; i++) {
		T val = (input.splays.s[i].value - total_splays[i]) * input.splays.s[i].factor;
		helper.AddValue((val));
	}
	// U_LOG_E("I CRY AFTER ADD VALUE");
}



// Distance from point to a line segment. Useful for treating joints like capsules.
template <typename T>
T
distance_to_segment(Vec3<T> ls_start, Vec3<T> ls_end, Vec3<T> pt, bool *weird)
{
	Vec3<T> start_to_end = vector_sub(ls_end, ls_start);
	Vec3<T> start_to_pt = vector_sub(pt, ls_start);
	if (vector_dot(start_to_end, start_to_pt) < T(0)) {
		// Point is behind the start of the segment, so we just use distance between them.
		// U_LOG_E("before");
		return vector_diff(ls_start, pt);
	}

	Vec3<T> end_to_start = vector_sub(ls_start, ls_end);
	Vec3<T> end_to_pt = vector_sub(pt, ls_end);

	if (vector_dot(end_to_start, end_to_pt) < T(0)) {
		// U_LOG_E("after");
		return vector_diff(ls_end, pt);
	}

	// U_LOG_E("perpendicular");

	Vec3<T> se_cross_sp;
	vector_cross(start_to_end, start_to_pt, se_cross_sp);

	T len_perpendicular = vector_length(se_cross_sp);
	T len_se = vector_length(start_to_end);

	// assert(len_se != T(0));
	if (len_perpendicular == T(0)) {
		U_LOG_E("Weird.");
		std::cout << ls_start.x << " " << ls_start.y << " " << ls_start.z << std::endl;
		std::cout << ls_end.x << " " << ls_end.y << " " << ls_end.z << std::endl;
		std::cout << pt.x << " " << pt.y << " " << pt.z << std::endl;
		*weird = true;
	}

	return len_perpendicular / len_se;
}


template <typename T>
T
magic_relu(T val)
{
	if (val <= T(0)) {
		return T(0);
	}

	T b = T(0.0005);
	T out_val = (T(-1) * sqrt(b)) + sqrt((val * val) + b);
	// std::cout << "beforeboi " << val << std::endl;
	// std::cout << "thisboi " << out_val << std::endl;

	// happens after a long enough time?
	// no this happens too when
	if (out_val != out_val) {
		U_LOG_E("YOU HAVE A NAN!!!! BADNESS!!!");
		if (val != val) {
			U_LOG_E("Because the input was a nan!!!");
		}
		abort();
	}
	return out_val;
}

#undef F_EXCEPT


// THIS IS NOT GOOD ENOUGH.
// This just compares distances between joints as spheres
// Whereas the correct way of doing it is comparing distances between _bones_ as capsules.
// You basically need lineline.cpp but with a bunch more special cases, so don't
// https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
// https://stackoverflow.com/questions/38637542/finding-the-shortest-distance-between-two-3d-line-segments
template <typename T>
void
computeResidualTouchy(KinematicHandLM &state, Translations55<T> translations_absolute, ResidualHelper<T> &helper)
{



	for (int outer_finger = 0; outer_finger < 5; outer_finger++) {
		for (int outer_joint = 0; outer_joint < 5; outer_joint++) {
			// U_LOG_E("awoo BRICK IN THE WALL... %d %d", outer_finger, outer_joint);
			if ((outer_finger == 0) && (outer_joint == 0)) {
				continue;
			}

			Vec3<T> point = translations_absolute.t[outer_finger][outer_joint];
			T point_radius = T(state.set_with_just_sizes.values
			                       .hand_joint_set_default[joints_5x5_to_26[outer_finger][outer_joint]]
			                       .radius);

			for (int inner_finger = 0; inner_finger < 5; inner_finger++) {
				for (int inner_joint = 0; inner_joint < 4; inner_joint++) {
					// Skip hidden joint

					// Skip the awful new finger hidden joints I made
					if (inner_joint == 0) {
						continue;
					}
					// Skip the regular thumb hidden joint too
					if ((inner_finger == 0) && (inner_joint == 1)) {
						continue;
					}
					// We're comparing the distance between two joints on the same finger
					if (inner_finger == outer_finger) {
						// We're comparing the distance between the same joint - skip
						if (inner_joint == outer_joint) {
							continue;
						}
						// We're comparing the distance between this joint and its direct parent
						// - skip
						if (inner_joint == outer_joint - 1) {
							continue;
						}
						// We're comparing the distance between this joint and its direct child
						// - skip
						if (inner_joint == outer_joint + 1) {
							continue;
						}
					}

					// U_LOG_E("JUST ANOTHER BRICK IN THE WALL... %d %d", inner_finger,
					// inner_joint);
					T radius_root =
					    T(state.set_with_just_sizes.values
					          .hand_joint_set_default[joints_5x5_to_26[inner_finger][inner_joint]]
					          .radius);
					T min_dist = point_radius + radius_root;

					bool weird = false;
					T dist = distance_to_segment(
					    translations_absolute.t[inner_finger][inner_joint],
					    translations_absolute.t[inner_finger][inner_joint + 1], point, &weird);
					if (weird) {
						U_LOG_E("%d %d, %d %d", outer_finger, outer_joint, inner_finger,
						        inner_joint);
					}


					T err = magic_relu(min_dist - dist);

					helper.AddValue(err * T(100));
				}
			}
		}
	}
}


template <typename T, typename T2>
T
quat_difference(Quat<T> q1, T2 q2)
{
	// https://math.stackexchange.com/a/90098
	// d(q1,q2)=1−⟨q1,q2⟩2

	T inner_product = (q1.w * q2.w) + (q1.x * q2.x) + (q1.y * q2.y) + (q1.z * q2.z);
	return T(1.0) - (inner_product * inner_product);
}

template <typename T>
void
computeResidualMatchInputJoints(KinematicHandLM &state,
                                Translations55<T> translations_absolute,
                                Orientations54<T> orientations,
                                ResidualHelper<T> &helper)
{
	for (int finger = 0; finger < 5; finger++) {
		for (int joint = 1; joint < 5; joint++) {
			int xrtj = joints_5x5_to_26[finger][joint];
			xrt_vec3 xrtpos = state.input->target_pose[xrtj].position;
			xrt_quat xrtq = state.input->target_pose[xrtj].orientation;
			T fac = joint == 4 ? T(70) : T(10.0);
			helper.AddValue((T(xrtpos.x) - translations_absolute.t[finger][joint].x) * fac);
			helper.AddValue((T(xrtpos.y) - translations_absolute.t[finger][joint].y) * fac);
			helper.AddValue((T(xrtpos.z) - translations_absolute.t[finger][joint].z) * fac);

			T qdiff = quat_difference(orientations.q[finger][joint], xrtq);
			helper.AddValue(qdiff);
		}
	}
}


template <typename T, int N>
bool
derivativeIsNan(ceres::Jet<T, N> x0)
{
	for (int i = 0; i < x0.v.size(); ++i) {
		if (isnan(x0.v(i))) {
			// The derivative part of x0 contains a nan value
			return true;
		}
	}
	return false;
}

template <typename T>
void
computeResidualFingie(OptimizerHand<T> &hand, KinematicHandLM &state, ResidualHelper<T> &helper)
{

	Translations55<T> translations_absolute;
	Orientations54<T> orientations_absolute;

	HandScalar we_care_joint[] = {1.3, 0.9, 0.9, 1.3};
	HandScalar we_care_finger[] = {1.0, 1.0, 0.8, 0.8, 0.8};

	eval_hand_with_orientation(hand, state, translations_absolute, orientations_absolute);

	if (!state.input->use_target_pose) {

		for (int i = 0; i < NUM_TIP_TOUCH_COMBOS; i++) {
			size_t i0 = ttc_elements[i][0];
			size_t i1 = ttc_elements[i][1];
			T touchy = vector_diff(translations_absolute.t[i0][4], translations_absolute.t[i1][4]);
			if (isnan(touchy)) {
				U_LOG_E("touchy is nan");
				abort();
			}
			if (isnan(state.input->tip_touches.t[i].value)) {
				U_LOG_E("value is nan");
				abort();
			}
			if (isnan(state.input->tip_touches.t[i].factor)) {
				U_LOG_E("factor is nan");
				abort();
			}

			if constexpr (!std::is_same<T, float>::value) {
				if (derivativeIsNan(touchy)) {
					U_LOG_E("nananananananan");
					abort();
				}
			}

			// ok, one of these has
			T val = touchy - T(state.input->tip_touches.t[i].value);
			val *= T(state.input->tip_touches.t[i].factor);
			helper.AddValue(val);

			// std::cout << i0 << " " << i1 << " " << touchy << " " << val << std::endl;
		}
	} else {
		computeResidualMatchInputJoints(state, translations_absolute, orientations_absolute, helper);
	}

	computeResidualTouchy(state, translations_absolute, helper);


	// std::cout << "pinch " << pinch << std::endl;
}

template <typename T>
void
computeResidualDontLetThumbMetacarpalMoveTooMUch(OptimizerHand<T> &hand,
                                                 KinematicHandLM &state,
                                                 ResidualHelper<T> &helper)
{

	// This is because the joint limits on some models is not ideal and lets the thumb swing out more than I'd like
	// We need a way to visualize the scales of each residual.

	Translations55<T> translations_absolute;
	Orientations54<T> orientations_absolute;

	T scaler = T(1.0);
	T scaler2 = T(0.1);
	helper.AddValue(hand.thumb.metacarpal.swing.x * scaler);
	helper.AddValue(hand.thumb.metacarpal.swing.y * scaler * scaler2);
	helper.AddValue(hand.thumb.metacarpal.twist * scaler * scaler2);
}


template <typename T>
bool
CostFunctor::operator()(const T *const x, T *residual) const
{

#ifdef F_EXCEPT
	// https://stackoverflow.com/questions/60731382/c-setting-floating-point-exception-environment
	fetestexcept(FE_ALL_EXCEPT);

	std::fexcept_t my_flag = FE_DIVBYZERO;
	// fesetexceptflag(&my_flag,FE_ALL_EXCEPT); // Uncomment this for version 1
	feenableexcept(FE_INVALID | FE_OVERFLOW); // Uncomment this for version 2
#endif


	struct KinematicHandLM &state = this->parent;
	OptimizerHand<T> hand = {};
	// ??? should I do the below? probably.
	// Quat<T> tmp = this->parent.last_frame_pre_rotation;
	OptimizerHandInit<T>(hand);
	OptimizerHandUnpackFromVector(x, T(state.target_hand_size), state.the_limit, hand);

	XRT_MAYBE_UNUSED size_t residual_size = calc_residual_size(!state.first_frame);

// When you're hacking, you want to set the residuals to always-0 so that any of them you forget to touch keep their 0
// gradient.
// But then later this just becomes a waste.
#if 1
	for (size_t i = 0; i < residual_size; i++) {
		residual[i] = (T)(0);
	}
#endif

	// ResidualHelper<T> *helper = new ResidualHelper<T>(residual, residual_size);


	ResidualHelper<T> helper(residual, residual_size);

	if (!state.first_frame) {
		computeResidualStability<T>(hand, state.last_frame, state, helper);
	}
	computeResidualCurlSway(hand, state, helper);

	computeResidualFingie(hand, state, helper);
	computeResidualDontLetThumbMetacarpalMoveTooMUch(hand, state, helper);

	// Bounds checking - we should have written exactly to the end.
	// U_LOG_E("%zu %zu", helper.out_residual_idx, residual_size);
	// U_LOG_E("%zu should = %zu", helper.out_residual_idx, residual_size);
	// assert(helper.out_residual_idx == residual_size);
	// If you're hacking, feel free to turn this off; just remember to not write off the end, and to initialize
	// everything somewhere (maybe change the above to an #if 1? )

#ifdef F_EXCEPT
	feclearexcept(FE_ALL_EXCEPT);
#endif

	return true;
}


template <typename T>
static inline void
zldtt_ori_right(Quat<T> &orientation, xrt_quat *out)
{
	out->x = -orientation.x;
	out->y = orientation.y;
	out->z = orientation.z;
	out->w = -orientation.w;
}

template <typename T>
static inline void
zldtt_ori_left(Quat<T> &orientation, xrt_quat *out)
{
	out->w = orientation.w;
	out->x = orientation.x;
	out->y = orientation.y;
	out->z = orientation.z;
}

template <typename T>
static inline void
zldtt(Vec3<T> &trans, Quat<T> &orientation, bool is_right, xrt_space_relation &out)
{

	out.relation_flags = (enum xrt_space_relation_flags)(
	    XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
	    XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT);
	out.pose.position.x = trans.x;
	out.pose.position.y = trans.y;
	out.pose.position.z = trans.z;
	if (is_right) {
		zldtt_ori_right(orientation, &out.pose.orientation);
	} else {
		zldtt_ori_left(orientation, &out.pose.orientation);
	}
}

static void
eval_to_viz_hand(KinematicHandLM &state, xrt_hand_joint_set &out_viz_hand)
{
	XRT_TRACE_MARKER();

	//!@todo It's _probably_ fine to have the bigger size?
	Eigen::Matrix<HandScalar, calc_input_size(false), 1> pose = state.TinyOptimizerInput.cast<HandScalar>();

	OptimizerHand<HandScalar> opt = {};
	OptimizerHandInit(opt);
	OptimizerHandUnpackFromVector(pose.data(), state.target_hand_size, state.the_limit, opt);

	Translations55<HandScalar> translations_absolute;
	Orientations54<HandScalar> orientations_absolute;
	// Vec3<HandScalar> translations_absolute[kNumFingers][kNumJointsInFinger];
	// Quat<HandScalar> orientations_absolute[kNumFingers][kNumOrientationsInFinger];

	eval_hand_with_orientation(opt, state, translations_absolute, orientations_absolute);

	Vec3<HandScalar> wrist_location = Vec3<HandScalar>::Zero();
	Quat<HandScalar> wrist_orientation = Quat<HandScalar>::Identity();


	int joint_acc_idx = 0;

	// Palm.

	Vec3<HandScalar> palm_position;
	palm_position.x = (translations_absolute.t[2][0].x + translations_absolute.t[2][1].x) / 2;
	palm_position.y = (translations_absolute.t[2][0].y + translations_absolute.t[2][1].y) / 2;
	palm_position.z = (translations_absolute.t[2][0].z + translations_absolute.t[2][1].z) / 2;

	Quat<HandScalar> &palm_orientation = orientations_absolute.q[2][0];

	zldtt(palm_position, palm_orientation, state.is_right,
	      out_viz_hand.values.hand_joint_set_default[joint_acc_idx++].relation);

	// Wrist.
	zldtt(wrist_location, wrist_orientation, state.is_right,
	      out_viz_hand.values.hand_joint_set_default[joint_acc_idx++].relation);

	for (int finger = 0; finger < 5; finger++) {
		for (int joint = 0; joint < 5; joint++) {
			// This one is necessary
			if (finger == 0 && joint == 0) {
				continue;
			}
			Quat<HandScalar> *orientation;
			if (joint != 4) {
				orientation = &orientations_absolute.q[finger][joint + 1];
			} else {
				orientation = &orientations_absolute.q[finger][joint];
			}
			zldtt(translations_absolute.t[finger][joint], *orientation, state.is_right,
			      out_viz_hand.values.hand_joint_set_default[joint_acc_idx++].relation);
		}
	}
	out_viz_hand.is_active = true;
}



inline float
opt_run(KinematicHandLM &state, optimizer_input &observation, xrt_hand_joint_set &out_viz_hand)
{
	constexpr size_t input_size = calc_input_size(false);

	size_t residual_size = calc_residual_size(!state.first_frame);

	LM_DEBUG(state, "Running with %zu inputs and %zu residuals!", input_size, residual_size);

	CostFunctor cf(state, residual_size);

	using AutoDiffCostFunctor =
	    ceres::TinySolverAutoDiffFunction<CostFunctor, Eigen::Dynamic, input_size, HandScalar>;

	AutoDiffCostFunctor f(cf);


	// Okay I have no idea if this should be {}-initialized or not. Previous me seems to have thought no, but it
	// works either way.
	ceres::TinySolver<AutoDiffCostFunctor> solver = {};
	solver.options.max_num_iterations = 50;
	// We need to do a parameter sweep for the trust region and see what's fastest.
	// solver.options.initial_trust_region_radius = 1e3;
	solver.options.function_tolerance = 1e-6;

	Eigen::Matrix<HandScalar, input_size, 1> inp = state.TinyOptimizerInput.head<input_size>();

	XRT_MAYBE_UNUSED uint64_t start = os_monotonic_get_ns();
	XRT_MAYBE_UNUSED auto summary = solver.Solve(f, &inp);
	XRT_MAYBE_UNUSED uint64_t end = os_monotonic_get_ns();

	//!@todo Is there a zero-copy way of doing this?
	state.TinyOptimizerInput.head<input_size>() = inp;

	if (state.log_level <= U_LOGGING_DEBUG) {

		uint64_t diff = end - start;
		double time_taken = (double)diff / (double)U_TIME_1MS_IN_NS;

		const char *status = "UNDEFINED";

		switch (summary.status) {
		case 0: {
			status = "GRADIENT_TOO_SMALL";
		} break;
		case 1: {
			status = "RELATIVE_STEP_SIZE_TOO_SMALL";
		} break;
		case 2: {
			status = "COST_TOO_SMALL";
		} break;
		case 3: {
			status = "HIT_MAX_ITERATIONS";
		} break;
		case 4: {
			status = "COST_CHANGE_TOO_SMALL";
		} break;
		}

		LM_DEBUG(state, "Status: %s, num_iterations %d, max_norm %E, gtol %E", status, summary.iterations,
		         summary.gradient_max_norm, solver.options.gradient_tolerance);
		LM_DEBUG(state, "Took %f ms", time_taken);
		if (summary.iterations < 3) {
			LM_DEBUG(state, "Suspiciouisly low number of iterations!");
		}
	}
	return 0;
}


void
hand_was_untracked(KinematicHandLM *hand)
{
	hand->first_frame = true;

	OptimizerHandInit(hand->last_frame);
	OptimizerHandPackIntoVector(hand->last_frame, hand->the_limit, hand->TinyOptimizerInput.data());
}

void
optimizer_run(KinematicHandLM *hand, optimizer_input &input, xrt_hand_joint_set &out_viz_hand)
{
	KinematicHandLM &state = *hand;

	state.input = &input;



	// For now, we have to statically instantiate different versions of the optimizer depending on how many input
	// parameters there are. For now, there are only two cases - either we are optimizing the hand size or we are
	// not optimizing it.
	// !@todo Can we make a magic template that automatically instantiates the right one, and also make it so we can
	// decide to either make the residual size dynamic or static? Currently, it's dynamic, which is easier for us
	// and makes compile times a lot lower, but it probably makes things some amount slower at runtime.

	opt_run(state, input, out_viz_hand);



	// Postfix - unpack,
	OptimizerHandUnpackFromVector(state.TinyOptimizerInput.data(), state.target_hand_size, state.the_limit,
	                              state.last_frame);



	// Squash the orientations
	// OptimizerHandSquashRotations(state.last_frame, state.last_frame_pre_rotation);

	// Repack - brings the curl values back into original domain. Look at ModelToLM/LMToModel, we're using sin/asin.
	OptimizerHandPackIntoVector(state.last_frame, hand->the_limit, state.TinyOptimizerInput.data());



	eval_to_viz_hand(state, out_viz_hand);

	state.first_frame = false;
}



void
optimizer_create(xrt_pose left_in_right,
                 bool is_right,
                 u_logging_level log_level,
                 struct hand_proportions proportions,
                 HandLimit joint_limits,
                 KinematicHandLM **out_kinematic_hand)
{
	KinematicHandLM *hand = new KinematicHandLM;
	hand->proportions = proportions;
	hand->target_hand_size = proportions.hand_size;
	hand->the_limit = joint_limits;

	hand->is_right = is_right;
	hand->log_level = log_level;

	u_hand_joints_apply_joint_width(&hand->set_with_just_sizes);

	// Probably unnecessary.
	hand_was_untracked(hand);

	*out_kinematic_hand = hand;
}

void
optimizer_destroy(KinematicHandLM **hand)
{
	delete *hand;
	hand = NULL;
}
} // namespace xrt::tracking::hand::mercury::lm
