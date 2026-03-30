#include "math/m_vec3.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_trace_marker.h"

#include "stereokit.h"
#include "stereokit_ui.h"
using namespace sk;

#include <Eigen/Core>
#include <Eigen/Geometry>

// #include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/tiny_solver.h"
#include "ceres/tiny_solver_autodiff_function.h"

#include "glog/logging.h"

#include <iostream>
#include <cmath>
#include <random>
#include "randoviz.hpp"
#include "hand_interface.hpp"
#include "OptimizerParams_SNC.hpp"

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)


template <typename T> struct StereographicObservation
{
	T obs[21][2];
};

struct LMKinematicHand
{
	bool first_frame = true;
	HandScalar camera_baseline;
	one_frame_input *observation;

	StereographicObservation<HandScalar> sgo[2];

	HandScalar last_frame_pre_rotation[4];
	OptimizerHand<HandScalar> last_frame;
	// OptimizerHand<HandScalar> this_frame; // don't need to store it anywhere

	Eigen::Vector<HandScalar, kFunctorInputDim> TinyOptimizer_input;
};


struct CostFunctor
{
	LMKinematicHand &parent;

	template <typename T>
	bool
	operator()(const T *const x, T *residual) const;

	CostFunctor(LMKinematicHand &in_last_hand) : parent(in_last_hand) {}
};



template <typename T>
inline void
set_1dof_joint_trans(T length, T hand_size, T out_pos[3])
{
	out_pos[0] = (T)(0);
	out_pos[1] = (T)(0);
	out_pos[2] = length;
}

template <typename T>
inline void
set_finger_metacarpal_trans(T x, T z, T hand_size, T out_pos[3])
{
	out_pos[0] = x;
	out_pos[1] = (T)(0);
	out_pos[2] = z;
}


template <typename T>
inline void
eval_hand_set_rel_translations(const OptimizerHand<T> &opt, T rel_translations[5][5][3])
{
	// XRT_TRACE_MARKER();
	T hs = opt.hand_size;

	// Basically, we're walking up rel_translations, writing strictly sequentially. Hopefully this is fast.

	// Thumb metacarpal translation.
	set_finger_metacarpal_trans((T)0.33097, (T)-0.25968, hs, rel_translations[0][0]);

	// Comes after the invisible joint.
	set_1dof_joint_trans((T)0.0, hs, rel_translations[0][1]);
	// prox, distal, tip
	set_1dof_joint_trans((T)-0.389626, hs, rel_translations[0][2]);
	set_1dof_joint_trans((T)-0.311176, hs, rel_translations[0][3]);
	set_1dof_joint_trans((T)-0.232195, hs, rel_translations[0][4]);

	// What's the best place to put this? Here works, but is there somewhere we could put it where it gets accessed
	// faster?
	constexpr HandScalar finger_joint_lengths[4][4] = {
	    {
	        -0.66,
	        -0.365719,
	        -0.231581,
	        -0.201790,
	    },
	    {
	        -0.645,
	        -0.404486,
	        -0.247749,
	        -0.210121,
	    },
	    {
	        -0.58,
	        -0.365639,
	        -0.225666,
	        -0.187089,
	    },
	    {
	        -0.52,
	        -0.278197,
	        -0.176178,
	        -0.157566,
	    },
	};

	constexpr HandScalar metacarpal_translations[4][2]{
	    {0.16926, -0.34437},
	    {0.034639, -0.35573},
	    {-0.063625, -0.34164},
	    {-0.1509, -0.30373},
	};

	for (int finger = HF_INDEX; finger <= HF_LITTLE; finger++) {
		set_finger_metacarpal_trans(T(metacarpal_translations[finger - 1][0]), //
		                            T(metacarpal_translations[finger - 1][1]), //
		                            hs,                                        //
		                            rel_translations[finger][0]);

		for (int i = 0; i < 4; i++) {
			int bone = i + 1;
			set_1dof_joint_trans((T)finger_joint_lengths[finger - 1][i], hs,
			                     rel_translations[finger][bone]);
		}
	}


	// // This seems to be just about the same speed as multiplying as-needed above, so I'm doing it this way.
	// //
	// for (int finger = 0; finger < 5; finger++) {
	// 	for (int bone = 0; bone < 5; bone++) {
	// 		rel_translations[finger][bone][0] *= opt.hand_size;
	// 		rel_translations[finger][bone][1] *= opt.hand_size;
	// 		rel_translations[finger][bone][2] *= opt.hand_size;
	// 	}
	// }
}



template <typename T>
inline void
swing_to_quat(const T swing[2], T quaternion[4])
{

	const T &a0 = swing[0];
	const T &a1 = swing[1];
	const T theta_squared = a0 * a0 + a1 * a1;

	// For points not at the origin, the full conversion is numerically stable.
	if (likely(theta_squared > T(0.0))) {
		const T theta = sqrt(theta_squared);
		const T half_theta = theta * T(0.5);
		const T k = sin(half_theta) / theta;
		quaternion[0] = cos(half_theta);
		quaternion[1] = a0 * k;
		quaternion[2] = a1 * k;
		quaternion[3] = T(0);
	} else {
		// At the origin, sqrt() will produce NaN in the derivative since
		// the argument is zero.  By approximating with a Taylor series,
		// and truncating at one term, the value and first derivatives will be
		// computed correctly when Jets are used.
		const T k(0.5);
		quaternion[0] = T(1.0);
		quaternion[1] = a0 * k;
		quaternion[2] = a1 * k;
		quaternion[3] = T(0);
	}
}

template <typename T>
inline void
swing_twist_to_quat(const T swing[2], const T twist, T out_quat[4])
{
	// We can definitely speed this up.
	T swing_quat[4];
	T twist_quat[4];

	T aax_twist[3];

	aax_twist[0] = (T)(0);
	aax_twist[1] = (T)(0);
	aax_twist[2] = twist;

	swing_to_quat(swing, swing_quat);

	ceres::AngleAxisToQuaternion(aax_twist, twist_quat);

	ceres::QuaternionProduct(swing_quat, twist_quat, out_quat);
}

// no! do this in 2D!
template <typename T>
inline void
curl_to_quat(const T &curl, T quaternion[4])
{
	T curl_aax[3] = {curl, (T)(0), (T)(0)};

	const T theta_squared = curl * curl;

	// For points not at the origin, the full conversion is numerically stable.
	if (likely(theta_squared > T(0.0))) {
		// const T theta = sqrt(theta_squared);
		// const T half_theta = theta * T(0.5);
		const T k = sin(curl * T(0.5)) / curl;
		quaternion[0] = cos(curl);
		quaternion[1] = curl * k;
		quaternion[2] = T(0.0);
		quaternion[3] = T(0.0);
	} else {
		// At the origin, dividing by 0 is probably bad. By approximating with a Taylor series,
		// and truncating at one term, the value and first derivatives will be
		// computed correctly when Jets are used.
		const T k(0.5);
		quaternion[0] = T(1.0);
		quaternion[1] = curl * k;
		quaternion[2] = T(0.0);
		quaternion[3] = T(0.0);
	}

	ceres::AngleAxisToQuaternion(curl_aax, quaternion);
}

template <typename T>
inline void
eval_hand_set_rel_orientations(const OptimizerHand<T> &opt, T rel_orientations[5][5][4])
{

// Thumb MCP hidden orientation
#if 0
	T mcp_root_swing[2];
	mcp_root_swing[0] = rad<T>((T)(-10));
	mcp_root_swing[1] = rad<T>((T)(-40));

	T mcp_root_twist = rad<T>((T)(-70));

	swing_twist_to_quat(mcp_root_swing, mcp_root_twist, rel_orientations[0][0]);

	std::cout << "\n\n\n\nHIDDEN ORIENTATION\n";
	std::cout << std::setprecision(100);
	for (int i = 0; i < 4; i++) {
		std::cout << rel_orientations[0][0][i] << std::endl;
	}
#else
	// This should be exactly equivalent to the above
	rel_orientations[0][0][0] = T(0.7666969438299513495138626240077428519725799560546875);
	rel_orientations[0][0][1] = T(0.1259717229573083796534405109923682175576686859130859375);
	rel_orientations[0][0][2] = T(-0.32878905371049393924209880424314178526401519775390625);
	rel_orientations[0][0][3] = T(-0.53684697959207838824369218855281360447406768798828125);
#endif

	// Thumb MCP orientation
	swing_twist_to_quat(opt.thumb.metacarpal.swing, //
	                    opt.thumb.metacarpal.twist, //
	                    rel_orientations[0][1]);

	// Thumb curls
	curl_to_quat(opt.thumb.rots[0], rel_orientations[0][2]);
	curl_to_quat(opt.thumb.rots[1], rel_orientations[0][3]);

	// Finger orientations
	for (int i = 0; i < 4; i++) {
		swing_twist_to_quat(opt.finger[i].metacarpal.swing, //
		                    opt.finger[i].metacarpal.twist, //
		                    rel_orientations[i + 1][0]);

		swing_to_quat(opt.finger[i].proximal_swing, //
		              rel_orientations[i + 1][1]);

		curl_to_quat(opt.finger[i].rots[0], rel_orientations[i + 1][2]);
		curl_to_quat(opt.finger[i].rots[1], rel_orientations[i + 1][3]);
	}
}

template <typename T>
inline void
UnitQuaternionRotateAndScalePoint(const T q[4], const T pt[3], const T scale, T result[3])
{
	T uv0 = q[2] * pt[2] - q[3] * pt[1];
	T uv1 = q[3] * pt[0] - q[1] * pt[2];
	T uv2 = q[1] * pt[1] - q[2] * pt[0];
	uv0 += uv0;
	uv1 += uv1;
	uv2 += uv2;
	result[0] = ((pt[0] + q[0] * uv0) + (q[2] * uv2 - q[3] * uv1)) * scale;
	result[1] = ((pt[1] + q[0] * uv1) + (q[3] * uv0 - q[1] * uv2)) * scale;
	result[2] = ((pt[2] + q[0] * uv2) + (q[1] * uv1 - q[2] * uv0)) * scale;
}

template <typename T>
void
eval_hand_with_orientation(const OptimizerHand<T> &opt,
                           T translations_absolute[5][5][3],
                           T orientations_absolute[5][4][4])
{
	XRT_TRACE_MARKER();


	T rel_translations[5][5][3];
	T rel_orientations[5][5][4];

	eval_hand_set_rel_orientations(opt, rel_orientations);

	eval_hand_set_rel_translations(opt, rel_translations);

	T orientation_root[4];

	T post_orientation_quat[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax, post_orientation_quat);

	ceres::QuaternionProduct(opt.wrist_pre_orientation_quat, post_orientation_quat, orientation_root);

	// Do orientations.
	// Tip joint has the same orientation as distal joint, and for optimization we don't need to care about its
	// orientation, so we only do the first 4.
	for (int finger = 0; finger < 5; finger++) {
		const T *last_orientation = orientation_root;
		for (int bone = 0; bone < 4; bone++) {
			T *out_orientation = orientations_absolute[finger][bone];
			T *rel_orientation = rel_orientations[finger][bone];

			ceres::QuaternionProduct(last_orientation, rel_orientation, out_orientation);
			last_orientation = out_orientation;
		}
	}



	for (int finger = 0; finger < 5; finger++) {
		const T *last_translation = opt.wrist_location;
		const T *last_orientation = orientation_root;
		for (int bone = 0; bone < 5; bone++) {
			T *out_translation = translations_absolute[finger][bone];
			T *rel_translation = rel_translations[finger][bone];
			// doing this is 1750-1900.
			// applying scale in in eval_hand_set_rel_translations is 2000ns on double.
			UnitQuaternionRotateAndScalePoint(last_orientation, rel_translation, opt.hand_size,
			                                  out_translation);

			out_translation[0] += last_translation[0];
			out_translation[1] += last_translation[1];
			out_translation[2] += last_translation[2];

			// Next iteration, the orientation to rotate by should be the head-relative orientation of this
			// joint.
			last_orientation = orientations_absolute[finger][bone];
			// Ditto for translation
			last_translation = out_translation;
		}
	}
}


template <typename T>
void
CostFunctor_StabilityPart(const OptimizerHand<T> &hand,
                          const OptimizerHand<HandScalar> &last_hand,
                          T *residual,
                          int &out_residual_idx)
{

	constexpr HandScalar root = 1.2;
	constexpr HandScalar curl_root = root * 0.06;
	constexpr HandScalar other_root = root * 0.03;

	constexpr HandScalar thumb_mcp_swing_fac = curl_root * 1.5;
	constexpr HandScalar thumb_mcp_twist_fac = curl_root * 1.5;

	constexpr HandScalar finger_mcp_swing_fac = curl_root * 3.0;
	constexpr HandScalar finger_mcp_twist_fac = curl_root * 3.0;

	constexpr HandScalar finger_pxm_x_fac = curl_root * 1.0;
	constexpr HandScalar finger_pxm_y_fac = curl_root * 1.0;

	constexpr HandScalar root_position_fac = other_root * 30;
	constexpr HandScalar hand_size_fac = other_root * 100;

	constexpr HandScalar hand_orientation_fac = other_root * 3;



	// std::cout << "hs " << hand.hand_size() << " " << last_hand.hand_size() << std::endl;
	residual[out_residual_idx++] = (hand.hand_size - last_hand.hand_size) * (T)(hand_size_fac);


	residual[out_residual_idx++] = (last_hand.wrist_location[0] - hand.wrist_location[0]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location[1] - hand.wrist_location[1]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location[2] - hand.wrist_location[2]) * root_position_fac;


	for (int i = 0; i < 3; i++) {
		residual[out_residual_idx++] = (hand.wrist_post_orientation_aax[i]) * (T)(hand_orientation_fac);
	}



	residual[out_residual_idx++] =
	    (hand.thumb.metacarpal.swing[0] - last_hand.thumb.metacarpal.swing[0]) * thumb_mcp_swing_fac;
	residual[out_residual_idx++] =
	    (hand.thumb.metacarpal.swing[1] - last_hand.thumb.metacarpal.swing[1]) * thumb_mcp_swing_fac;
	residual[out_residual_idx++] =
	    (hand.thumb.metacarpal.twist - last_hand.thumb.metacarpal.twist) * thumb_mcp_twist_fac;

	residual[out_residual_idx++] = (hand.thumb.rots[0] - last_hand.thumb.rots[0]) * curl_root;
	residual[out_residual_idx++] = (hand.thumb.rots[1] - last_hand.thumb.rots[1]) * curl_root;



	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		const OptimizerFinger<HandScalar> &finger_last = last_hand.finger[finger_idx];

		const OptimizerFinger<T> &finger = hand.finger[finger_idx];

		residual[out_residual_idx++] =
		    (finger.metacarpal.swing[0] - finger_last.metacarpal.swing[0]) * finger_mcp_swing_fac;

		residual[out_residual_idx++] =
		    (finger.metacarpal.swing[1] - finger_last.metacarpal.swing[1]) * finger_mcp_swing_fac;



		residual[out_residual_idx++] =
		    (finger.metacarpal.twist - finger_last.metacarpal.twist) * finger_mcp_twist_fac;



		residual[out_residual_idx++] =
		    (finger.proximal_swing[0] - finger_last.proximal_swing[0]) * finger_pxm_x_fac;
		residual[out_residual_idx++] =
		    (finger.proximal_swing[1] - finger_last.proximal_swing[1]) * finger_pxm_y_fac;

		residual[out_residual_idx++] = (finger.rots[0] - finger_last.rots[0]) * curl_root;
		residual[out_residual_idx++] = (finger.rots[1] - finger_last.rots[1]) * curl_root;
	}
}


template <typename T>
inline void
normalize_vector_inplace(T *vector)
{
	T len = (T)(0);

	for (int i = 0; i < 3; i++) {
		len += vector[i] * vector[i];
	}

	len = sqrt(len);

	if (len <= DBL_EPSILON) {
		// std::cout << "WEEEEOOOOOO ZERO LENGTH VECTOR" << std::endl;
		vector[2] = (T)-1;
		return;
	}

	for (int i = 0; i < 3; i++) {
		vector[i] /= len;
	}
}

// def stereographic_2d_to_
// in size: 3, out size: 2
template <typename T>
inline void
unit_vector_to_stereographic(const T *in, T *out)
{
	out[0] = in[0] / ((T)1 - in[2]);
	out[1] = in[1] / ((T)1 - in[2]);
}


template <typename T>
inline void
unit_xrt_vec3_to_stereographic(const xrt_vec3 in, T *out)
{
	T vec[3];
	vec[0] = (T)(in.x);
	vec[1] = (T)(in.y);
	vec[2] = (T)(in.z);

	normalize_vector_inplace(vec);

	unit_vector_to_stereographic(vec, out);
}

template <typename T>
void
diff(const T *model_joint_pos,
     const T move_joint_amount,
     const StereographicObservation<HandScalar> &observation,
     const HandScalar *confidences,
     const HandScalar amount_we_care,
     int &hand_joint_idx,
     int &out_residual_idx,
     T *out_residual)
{
	T model_joint_dir_rel_camera[3];
	model_joint_dir_rel_camera[0] = model_joint_pos[0] + move_joint_amount;
	model_joint_dir_rel_camera[1] = model_joint_pos[1];
	model_joint_dir_rel_camera[2] = model_joint_pos[2];

	normalize_vector_inplace(model_joint_dir_rel_camera);

	T stereographic_model_dir[2];
	unit_vector_to_stereographic(model_joint_dir_rel_camera, stereographic_model_dir);


	const HandScalar confidence = confidences[hand_joint_idx] * amount_we_care;
	const HandScalar *observed_ray_sg = observation.obs[hand_joint_idx];

	out_residual[out_residual_idx++] = (stereographic_model_dir[0] - (T)(observed_ray_sg[0])) * confidence;
	out_residual[out_residual_idx++] = (stereographic_model_dir[1] - (T)(observed_ray_sg[1])) * confidence;

	hand_joint_idx++;
}



template <typename T>
void
CostFunctor_PositionsPart(OptimizerHand<T> &hand, LMKinematicHand &state, T *residual, int &out_residual_idx)
{

	T translations_absolute[5][5][3];
	T orientations_absolute[5][4][4];

	HandScalar we_care_joint[] = {1.0, 0.2, 0.2, 1.4};
	HandScalar we_care_finger[] = {1.0, 1.0, 0.8, 0.8, 0.8};

	eval_hand_with_orientation(hand, translations_absolute, orientations_absolute);
	// eval_hand_fast(hand, translations_absolute);
	for (int view = 0; view < 2; view++) {
		T move_amt;

		if (view == 0) {
			// left camera.
			move_amt = (T)(dist_between_cameras / 2);
		} else {
			move_amt = -(T)(dist_between_cameras / 2);
		}
		int joint_acc_idx = 0;

		HandScalar *confidences = state.observation->views[view].confidences;



		diff<T>(hand.wrist_location, move_amt, state.sgo[view], confidences, 1.5, joint_acc_idx, out_residual_idx,
		     residual);


		for (int finger_idx = 0; finger_idx < 5; finger_idx++) {
			for (int joint_idx = 0; joint_idx < 4; joint_idx++) {
				diff<T>(translations_absolute[finger_idx][joint_idx + 1], move_amt, state.sgo[view],
				     confidences, we_care_finger[finger_idx] * we_care_joint[joint_idx], joint_acc_idx,
				     out_residual_idx, residual);
			}
		}
	}
}


template <typename T>
bool
CostFunctor::operator()(const T *const x, T *residual) const
{

	OptimizerHand<T> hand = {};
	// ??? should I do the below? probably.
	OptimizerHandInit<T>(hand, this->parent.last_frame_pre_rotation);
	OptimizerHandUnpackFromVector(x, hand);

	for (int i = 0; i < kHandResidualSize; i++) {
		residual[i] = (T)(0);
	}

	int out_residual_idx = 0;

	CostFunctor_PositionsPart(hand, this->parent, residual, out_residual_idx);
	CostFunctor_StabilityPart(hand, this->parent.last_frame, residual, out_residual_idx);

	return true;
}

template <typename T>
inline void
zldtt(T *trans, T *orientation, sk::pose_t &out)
{
	out.position.x = trans[0];
	out.position.y = trans[1];
	out.position.z = trans[2];

	out.orientation.w = orientation[0];
	out.orientation.x = orientation[1];
	out.orientation.y = orientation[2];
	out.orientation.z = orientation[3];
}

void
eval_to_viz_hand(LMKinematicHand &state, hand_output &out_viz_hand)
{
	XRT_TRACE_MARKER();

	Eigen::Vector<HandScalar, kFunctorInputDim> pose = state.TinyOptimizer_input.cast<HandScalar>();

	OptimizerHand<HandScalar> opt = {};
	OptimizerHandInit(opt, state.last_frame_pre_rotation);
	OptimizerHandUnpackFromVector(pose.data(), opt);

	HandScalar translations_absolute[5][5][3];
	HandScalar orientations_absolute[5][4][4];

	eval_hand_with_orientation(opt, translations_absolute, orientations_absolute);

	HandScalar post_wrist_orientation[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax, post_wrist_orientation);

	HandScalar pre_wrist_orientation[4];
	for (int i = 0; i < 4; i++) {
		pre_wrist_orientation[i] = state.last_frame_pre_rotation[i];
	}

	HandScalar final_wrist_orientation[4];

	ceres::QuaternionProduct(pre_wrist_orientation, post_wrist_orientation, final_wrist_orientation);

	zldtt(opt.wrist_location, final_wrist_orientation, out_viz_hand.wrist);

	for (int finger = 0; finger < 5; finger++) {
		for (int joint = 0; joint < 5; joint++) {
			HandScalar *orientation;
			if (joint != 4) {
				orientation = orientations_absolute[finger][joint];
			} else {
				orientation = orientations_absolute[finger][joint - 1];
			}
			zldtt(translations_absolute[finger][joint], orientation, out_viz_hand.fingers[finger][joint]);
		}
	}
}

double
push_frame(LMKinematicHand *hand_ptr, one_frame_input &observation, struct hand_output &out_viz_hand)
{
	LMKinematicHand &state = *hand_ptr;

	state.observation = &observation;

	for (int i = 0; i < 21; i++) {
		for (int view = 0; view < 2; view++) {
			unit_xrt_vec3_to_stereographic(observation.views[view].rays[i], state.sgo[view].obs[i]);
		}
	}


	CostFunctor cf(state);


	using AutoDiffCostFunctor = ceres::TinySolverAutoDiffFunction<CostFunctor, kHandResidualSize, kFunctorInputDim, HandScalar>;

	AutoDiffCostFunctor f(cf);

	ceres::TinySolver<AutoDiffCostFunctor> solver; // do NOT initialize to = {}, apparently.
	solver.options.max_num_iterations = 50;
	// We need to do a parameter sweep for the trust region and see what's fastest.
	// solver.options.initial_trust_region_radius = 1e3;
	solver.options.function_tolerance = 1e-6;



	uint64_t start = os_monotonic_get_ns();
	auto summary = solver.Solve(f, &state.TinyOptimizer_input);
	uint64_t end = os_monotonic_get_ns();

	uint64_t diff = end - start;
	double time_taken = (double)diff / (double)U_TIME_1MS_IN_NS;

	char *status;

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

	std::cout << "Status: " << status << " num_iterations " << summary.iterations << " max norm "
	          << summary.gradient_max_norm << " gtol " << solver.options.gradient_tolerance << std::endl;
	U_LOG_I("Took %f ms", time_taken);
	if (summary.iterations < 3) {
		U_LOG_E("Suspiciouisly low number of iterations!");
	}



	// Postfix - unpack,
	OptimizerHandUnpackFromVector(state.TinyOptimizer_input.data(), state.last_frame);



	// Squash the orientations
	OptimizerHandSquashRotations(state.last_frame, state.last_frame_pre_rotation);

	// Repack - brings the curl values back into original domain. Look at ModelToLM/LMToModel, we're using sin/asin.
	OptimizerHandPackIntoVector(state.last_frame, state.TinyOptimizer_input.data());


	eval_to_viz_hand(state, out_viz_hand);

	return time_taken;
}


void
give_identity_hand(LMKinematicHand *hand, struct hand_output &out_viz_hand)
{
	eval_to_viz_hand(*hand, out_viz_hand);
}

void
create_kinematic_hand(HandScalar camera_baseline, LMKinematicHand **out_kinematic_hand)
{
	LMKinematicHand *hand = new LMKinematicHand;
	hand->first_frame = true;
	hand->camera_baseline = camera_baseline;
	// Probably unnecessary.
	hand->last_frame_pre_rotation[0] = 1.0;
	hand->last_frame_pre_rotation[1] = 0.0;
	hand->last_frame_pre_rotation[2] = 0.0;
	hand->last_frame_pre_rotation[3] = 0.0;

	OptimizerHandInit(hand->last_frame, hand->last_frame_pre_rotation);
	OptimizerHandPackIntoVector(hand->last_frame, hand->TinyOptimizer_input.data());

	*out_kinematic_hand = hand;
}



void
destroy_kinematic_hand(LMKinematicHand **hand)
{
	U_LOG_E("awa");
}
