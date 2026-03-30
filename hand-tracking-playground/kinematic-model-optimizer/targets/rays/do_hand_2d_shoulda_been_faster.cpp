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

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/tiny_solver.h"
#include "ceres/tiny_solver_autodiff_function.h"

#include "glog/logging.h"


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
	double camera_baseline;
	one_frame_input *observation;

	StereographicObservation<double> sgo[2];

	double last_frame_pre_rotation[4];
	OptimizerHand<double> last_frame;
	// OptimizerHand<double> this_frame; // don't need to store it anywhere

	Eigen::Vector<double, kFunctorInputDim> TinyOptimizer_input;
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
swing_twist_to_quat(T swing[2], T twist, T out_quat[4])
{
	// Just derive it, nerd! this is very very slow as-is.
	T swing_quat[4];
	T twist_quat[4];

	T aax_swing[3];
	T aax_twist[3];

	aax_swing[0] = swing[0];
	aax_swing[1] = swing[1];
	aax_swing[2] = (T)(0);

	aax_twist[0] = (T)(0);
	aax_twist[1] = (T)(0);
	aax_twist[2] = twist;

	ceres::AngleAxisToQuaternion(aax_swing, swing_quat);
	ceres::AngleAxisToQuaternion(aax_twist, twist_quat);

	ceres::QuaternionProduct(swing_quat, twist_quat, out_quat);
}

template <typename T> struct eval_hand_static_struct
{
	T finger_metacarpal_translations[4][3] = {
	    {(T)0.16926, (T)0, (T)-0.34437},
	    {(T)0.034639, (T)0, (T)-0.35573},
	    {(T)-0.063625, (T)0, (T)-0.34164},
	    {(T)-0.1509, (T)0, (T)-0.30373},
	};


	T thumb_hidden_mcp_translation[3] = {(T)0.33097, (T)0, (T)-0.25968};
	T thumb_hidden_mcp_orientation[4];

	T joint_lengths_1dof[5][3] = //
	    {{(T)0.389626, (T)0.311176, (T)0.232195},
	     {(T)0.365719, (T)0.231581, (T)0.201790},
	     {(T)0.404486, (T)0.247749, (T)0.210121},
	     {(T)0.365639, (T)0.225666, (T)0.187089},
	     {(T)0.278197, (T)0.176178, (T)0.157566}};

	T finger_metacarpal_lengths[4] = {(T)-0.66, (T)-0.645, (T)-0.58, (T)-0.52};


	eval_hand_static_struct()
	{
		// Todo: static-ify this - just write out the quat and paste it in.
		T mcp_root_swing[2];
		mcp_root_swing[0] = rad((T)(-10));
		mcp_root_swing[1] = rad((T)(-40));

		T mcp_root_twist = rad((T)(-70));

		swing_twist_to_quat<T>(mcp_root_swing, mcp_root_twist, this->thumb_hidden_mcp_orientation);
	}
};

static const struct eval_hand_static_struct<double> the_statics =
{
};

struct eval_hand_static_struct<float> the_statics_float =
{
};

struct eval_hand_static_struct<ceres::Jet<double, 40>> the_statics_jet =
{
};

// template <typename T>
// inline struct eval_hand_static_struct<T> &
// get_statics();

// template <>
// inline struct eval_hand_static_struct<double> &
// get_statics<double>()
// {
// 	return the_statics;
// }

// template <>
// inline struct eval_hand_static_struct<float> &
// get_statics<float>()
// {
// 	return the_statics_float;
// }


template <typename T>
inline void
ThumbQuaternionProduct(const double z[4], const T w[4], T zw[4])
{
	//   DCHECK_NE(z, zw) << "Inplace quaternion product is not supported.";
	//   DCHECK_NE(w, zw) << "Inplace quaternion product is not supported.";

	// clang-format off
  zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
  zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
  zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
  zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
	// clang-format on
}


template <typename T>
inline void
UnsafeUnitQuaternionRotatePoint(const T q[4], const T pt[3], T result[3])
{

	// clang-format off
  T uv0 = q[2] * pt[2] - q[3] * pt[1];
  T uv1 = q[3] * pt[0] - q[1] * pt[2];
  T uv2 = q[1] * pt[1] - q[2] * pt[0];
  uv0 += uv0;
  uv1 += uv1;
  uv2 += uv2;
  result[0] = pt[0] + q[0] * uv0;
  result[1] = pt[1] + q[0] * uv1;
  result[2] = pt[2] + q[0] * uv2;
  result[0] += q[2] * uv2 - q[3] * uv1;
  result[1] += q[3] * uv0 - q[1] * uv2;
  result[2] += q[1] * uv1 - q[2] * uv0;
	// clang-format on
}


template <typename T, typename T2, typename T3>
inline void
add_vector(const T *one, const T2 *two, T3 *out)
{
	for (int i = 0; i < 3; i++) {
		out[i] = one[i] + two[i];
	}
}


template <typename T>
inline void
eval_hand_fast_thumb_part(OptimizerHand<T> &opt, T translations_absolute[5][5][3])
{
	XRT_TRACE_MARKER();
	T mcp_opt_rotation[4];
	swing_twist_to_quat(opt.thumb().metacarpal().swing(), //
	                    opt.thumb().metacarpal().twist(), //
	                    mcp_opt_rotation);

	T mcp_final_rotation[4];

	ThumbQuaternionProduct(the_statics.thumb_hidden_mcp_orientation, //
	                       mcp_opt_rotation,                         //
	                       mcp_final_rotation);

	T pxm_trans[3] = {(T)(0), (T)(0), -(T)the_statics.joint_lengths_1dof[0][0]};

	// translations_absolute[0][2][0] =
	ceres::UnitQuaternionRotatePoint(mcp_final_rotation, pxm_trans, translations_absolute[0][2]);


	add_vector(translations_absolute[0][2], translations_absolute[0][1], translations_absolute[0][2]);
	// add_vector(translations_absolute[0][2], translations_absolute[0][1], translations_absolute[0][2]);
	// add_vector(translations_absolute[0][2], translations_absolute[0][1], translations_absolute[0][2]);


	T last_z = (T)(0);
	T last_y = (T)(0);
	T last_dir = (T)(0);


	for (int i = 0; i < 2; i++) {
		last_dir += opt.thumb().rots()[i];
		last_z += (T)(-1) * cos(last_dir) * the_statics.joint_lengths_1dof[0][1 + i];
		last_y += sin(last_dir) * the_statics.joint_lengths_1dof[0][1 + i];

		T trans_rel_mcp[3] = {(T)(0), last_y, last_z};
		ceres::UnitQuaternionRotatePoint(mcp_final_rotation, trans_rel_mcp, translations_absolute[0][3 + i]);
		add_vector(translations_absolute[0][3 + i], translations_absolute[0][2],
		           translations_absolute[0][3 + i]);
	}
}

template <typename T>
inline void
eval_hand_fast_finger(OptimizerHand<T> &opt, int finger_idx, T translations_absolute_finger[5][3])
{
	XRT_TRACE_MARKER();
	T mcp_wrist_rel_rotation[4];

	int abs_out_idx = finger_idx + 1;

	swing_twist_to_quat(opt.finger(finger_idx).metacarpal().swing(), //
	                    opt.finger(finger_idx).metacarpal().twist(), //
	                    mcp_wrist_rel_rotation);



	T trans_mcp_to_pxm[3] = {(T)(0), (T)(0), (T)the_statics.finger_metacarpal_lengths[finger_idx]};

	ceres::UnitQuaternionRotatePoint(mcp_wrist_rel_rotation, trans_mcp_to_pxm,
	                                 translations_absolute_finger[1]);


	add_vector(translations_absolute_finger[1], //
	           translations_absolute_finger[0], //
	           translations_absolute_finger[1]);


	T pxm_mcp_rel_rotation[4];
	T pxm_wrist_rel_rotation[4];

	swing_twist_to_quat(opt.finger(finger_idx).proximal_swing(), //
	                    (T)(0),                                  //
	                    pxm_mcp_rel_rotation);

	ceres::QuaternionProduct(mcp_wrist_rel_rotation, pxm_mcp_rel_rotation, pxm_wrist_rel_rotation);

	T pxm_trans[3] = {(T)(0), (T)(0), -(T)the_statics.joint_lengths_1dof[abs_out_idx][0]};


	ceres::UnitQuaternionRotatePoint(pxm_wrist_rel_rotation, pxm_trans, translations_absolute_finger[2]);

	add_vector(translations_absolute_finger[2], //
	           translations_absolute_finger[1], //
	           translations_absolute_finger[2]);

	T last_z = (T)(0);
	T last_y = (T)(0);
	T last_dir = (T)(0);


	for (int i = 0; i < 2; i++) {
		last_dir += opt.finger(finger_idx).rots()[i];
		last_z += (T)(-1) * cos(last_dir) * the_statics.joint_lengths_1dof[abs_out_idx][1 + i];
		last_y += sin(last_dir) * the_statics.joint_lengths_1dof[abs_out_idx][1 + i];

		T trans_rel_mcp[3] = {(T)(0), last_y, last_z};
		ceres::UnitQuaternionRotatePoint(pxm_wrist_rel_rotation, trans_rel_mcp,
		                                 translations_absolute_finger[3 + i]);

		add_vector(translations_absolute_finger[3 + i], translations_absolute_finger[2],
		           translations_absolute_finger[3 + i]);
	}
}


template <typename T>
void
eval_hand_fast(OptimizerHand<T> &opt, T translations_absolute[5][5][3]) //, T fingertips_norm_wrist_rel[5][3])
{
	XRT_TRACE_MARKER();

	/*finger 1.

	get pos/orientation of mcp.
	write pos out to wrist_rel_translations.

	get orientation of pxm, apply on top of mcp.
	save pos and orientation, write pos to wrist_rel_translations.

	do 2D pos relative to pxm of the next 3 joints, use hworld code and notebook.

	transform all these by pxm pose, write out.

	!! fingertips_wrist_rel should indeed be normalized with hand size.




	*/

	translations_absolute[0][1][0] = (T)(the_statics.thumb_hidden_mcp_translation[0]);
	translations_absolute[0][1][1] = (T)(the_statics.thumb_hidden_mcp_translation[1]);
	translations_absolute[0][1][2] = (T)(the_statics.thumb_hidden_mcp_translation[2]);

	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		int abs_out_idx = finger_idx + 1;

		translations_absolute[abs_out_idx][0][0] =
		    (T)(the_statics.finger_metacarpal_translations[finger_idx][0]);
		translations_absolute[abs_out_idx][0][1] =
		    (T)(the_statics.finger_metacarpal_translations[finger_idx][1]);
		translations_absolute[abs_out_idx][0][2] =
		    (T)(the_statics.finger_metacarpal_translations[finger_idx][2]);
	}

	eval_hand_fast_thumb_part(opt, translations_absolute);


	for (int i = 0; i < 4; i++) {
		eval_hand_fast_finger(opt, i, translations_absolute[i+1]);
	}
	T orientation_root[4];

	T post_orientation_quat[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax(), post_orientation_quat);

	ceres::QuaternionProduct(opt.wrist_pre_orientation_quat(), post_orientation_quat, orientation_root);

	// for (int i = 0; i < 4; i++) {
	// 	orientation_root[i] *= opt.hand_size();
	// }


	for (int finger_idx = 0; finger_idx < 5; finger_idx++) {
		for (int bone_idx = 0; bone_idx < 5; bone_idx++) {
			// if (finger_idx == 0 && bone_idx == 0) {
			// 	// Hidden extra finger joint
			// 	continue;
			// }

			T temp[3] = {
			    translations_absolute[finger_idx][bone_idx][0], // * opt.hand_size(),
			    translations_absolute[finger_idx][bone_idx][1], // * opt.hand_size(),
			    translations_absolute[finger_idx][bone_idx][2], // * opt.hand_size(),
			};
			UnsafeUnitQuaternionRotatePoint(orientation_root, translations_absolute[finger_idx][bone_idx],
			                                translations_absolute[finger_idx][bone_idx]);
			for (int i = 0; i < 3; i++) {
				translations_absolute[finger_idx][bone_idx][i] *= opt.hand_size();
			}
			add_vector(translations_absolute[finger_idx][bone_idx], opt.wrist_location(),
			           translations_absolute[finger_idx][bone_idx]);
		}
	}
}

// no! do this in 2D!
template <typename T>
void
curl_to_quat(T &curl, T quaternion[4])
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
		quaternion[2] = 0;
		quaternion[3] = 0;
	} else {
		// At the origin, dividing by 0 is probably bad. By approximating with a Taylor series,
		// and truncating at one term, the value and first derivatives will be
		// computed correctly when Jets are used.
		const T k(0.5);
		quaternion[0] = T(1.0);
		quaternion[1] = curl * k;
		quaternion[2] = 0;
		quaternion[3] = 0;
	}

	ceres::AngleAxisToQuaternion(curl_aax, quaternion);
}



template <typename T>
void
set_1dof_joint_trans(T length, T hand_size, T out_pos[3])
{
	out_pos[0] = (T)(0);
	out_pos[1] = (T)(0);
	out_pos[2] = length * hand_size;
}

template <typename T>
void
set_finger_metacarpal_trans(T x, T z, T hand_size, T out_pos[3])
{
	out_pos[0] = x * hand_size;
	out_pos[1] = (T)(0);
	out_pos[2] = z * hand_size;
}

template <typename T>
void
eval_hand_with_orientation(OptimizerHand<T> &opt,
                           //  pgm_state &state,
                           T translations_absolute[5][5][3],
                           T orientations_absolute[5][5][4])
{
	XRT_TRACE_MARKER();

	T &hs = opt.hand_size();



	// GET RID OF THESE INITIALIZATIONS LATER! SLOW!
	T rel_translations[5][5][3] = {};
	T rel_orientations[5][5][4] = {};

	{ // Root translation stuffs. Depends on hand size.

		// Thumb metacarpal translation.
		set_finger_metacarpal_trans((T)0.33097, (T)-0.25968, hs, rel_translations[0][0]);

		// Comes after the invisible joint.
		set_1dof_joint_trans((T)0.0, opt.hand_size(), rel_translations[0][1]);
		// prox, distal, tip
		set_1dof_joint_trans((T)-0.389626, opt.hand_size(), rel_translations[0][2]);
		set_1dof_joint_trans((T)-0.311176, opt.hand_size(), rel_translations[0][3]);
		set_1dof_joint_trans((T)-0.232195, opt.hand_size(), rel_translations[0][4]);

		// Finger metacarpal root translations.

		set_finger_metacarpal_trans((T)0.16926, (T)-0.34437, hs, rel_translations[HF_INDEX][0]);
		set_finger_metacarpal_trans((T)0.034639, (T)-0.35573, hs, rel_translations[HF_MIDDLE][0]);
		set_finger_metacarpal_trans((T)-0.063625, (T)-0.34164, hs, rel_translations[HF_RING][0]);
		set_finger_metacarpal_trans((T)-0.1509, (T)-0.30373, hs, rel_translations[HF_LITTLE][0]);

		// Metacarpal bone lengths
		set_1dof_joint_trans((T)-0.66, hs, rel_translations[HF_INDEX][1]);
		set_1dof_joint_trans((T)-0.645, hs, rel_translations[HF_MIDDLE][1]);
		set_1dof_joint_trans((T)-0.58, hs, rel_translations[HF_RING][1]);
		set_1dof_joint_trans((T)-0.52, hs, rel_translations[HF_LITTLE][1]);

		double finger_joint_lengths[4][3] = {
		    {
		        -0.365719,
		        -0.231581,
		        -0.201790,
		    },
		    {
		        -0.404486,
		        -0.247749,
		        -0.210121,
		    },
		    {
		        -0.365639,
		        -0.225666,
		        -0.187089,
		    },
		    {
		        -0.278197,
		        -0.176178,
		        -0.157566,
		    },
		};

		for (int finger = HF_INDEX; finger <= HF_LITTLE; finger++) {
			for (int i = 0; i < 3; i++) {
				int bone = i + 2;
				set_1dof_joint_trans((T)finger_joint_lengths[finger - 1][i], hs,
				                     rel_translations[finger][bone]);
			}
		}
	} // Root translation

#if 0
	{ // Hack
		for (int finger = 0; finger < 5; finger++) {
			for (int bone = 0; bone < 5; bone++) {
				rel_orientations[finger][bone][0] = 1.0;
			}
		}
	}
#endif

#if 1
	{
		// Slow but helps viz: init tip orientations
		for (int finger = 0; finger < 5; finger++) {
			rel_orientations[finger][4][0] = (T)(1.0);
		}
	}
#endif

	{ // Thumb MCP hidden orientation
#if 0
		T mcp_root_aax[3];
		mcp_root_aax[0] = rad<T>(-10);
		mcp_root_aax[1] = rad<T>(-40);
		mcp_root_aax[2] = rad<T>(-70);

		ceres::AngleAxisToQuaternion(mcp_root_aax, rel_orientations[0][0]);
#else
		T mcp_root_swing[2];
		mcp_root_swing[0] = rad<T>((T)(-10));
		mcp_root_swing[1] = rad<T>((T)(-40));

		T mcp_root_twist = rad<T>((T)(-70));

		swing_twist_to_quat(mcp_root_swing, mcp_root_twist, rel_orientations[0][0]);
#endif
	}


	{
		// Thumb MCP orientation
		swing_twist_to_quat(opt.thumb().metacarpal().swing(), //
		                    opt.thumb().metacarpal().twist(), //
		                    rel_orientations[0][1]);
	}

	{
		// Thumb curls
		curl_to_quat(opt.thumb().rots()[0], rel_orientations[0][2]);
		curl_to_quat(opt.thumb().rots()[1], rel_orientations[0][3]);
	}

	{
		// Finger MCP orientation

		for (int i = 0; i < 4; i++) {

			swing_twist_to_quat(opt.finger(i).metacarpal().swing(), //
			                    opt.finger(i).metacarpal().twist(), //
			                    rel_orientations[i + 1][0]);
			// no! speicalize!
			T zero = (T)(0);
			swing_twist_to_quat(opt.finger(i).proximal_swing(), //
			                    zero,                           //
			                    rel_orientations[i + 1][1]);
		}
	}

	{
		// Finger curls
		for (int i = 0; i < 4; i++) {
			curl_to_quat(opt.finger(i).rots()[0], rel_orientations[i + 1][2]);
			curl_to_quat(opt.finger(i).rots()[1], rel_orientations[i + 1][3]);
			// U_LOG_E("%d %f %f %f %f", i, rel_orientations[i][3][0], rel_orientations[i][3][1],
			//         rel_orientations[i][3][2], rel_orientations[i][3][3]);
		}
	}

	T orientation_root[4];

	T post_orientation_quat[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax(), post_orientation_quat);

	ceres::QuaternionProduct(opt.wrist_pre_orientation_quat(), post_orientation_quat, orientation_root);



	for (int finger = 0; finger < 5; finger++) {
		T *last_translation = opt.wrist_location();
		T *last_orientation = orientation_root;
		for (int bone = 0; bone < 5; bone++) {
			T *out_translation = translations_absolute[finger][bone];
			T *out_orientation = orientations_absolute[finger][bone];

			T *rel_translation = rel_translations[finger][bone];
			T *rel_orientation = rel_orientations[finger][bone];

			ceres::QuaternionProduct(last_orientation, rel_orientation, out_orientation);


			ceres::UnitQuaternionRotatePoint(last_orientation, rel_translation, out_translation);

			out_translation[0] += last_translation[0];
			out_translation[1] += last_translation[1];
			out_translation[2] += last_translation[2];

			last_translation = out_translation;
			last_orientation = out_orientation;
		}
	}
}


template <typename T>
void
CostFunctor_StabilityPart(OptimizerHand<T> &hand, OptimizerHand<double> &last_hand, T *residual, int &out_residual_idx)
{

	// const double curl_fac = 0.05;
	// const double curl_fac = 0.02;
	constexpr double root = 1.2;
	constexpr double curl_root = root * 0.06;
	constexpr double other_root = root * 0.03;

	constexpr double thumb_mcp_swing_fac = curl_root * 1.5;
	constexpr double thumb_mcp_twist_fac = curl_root * 1.5;

	constexpr double finger_mcp_swing_fac = curl_root * 3.0;
	constexpr double finger_mcp_twist_fac = curl_root * 3.0;

	constexpr double finger_pxm_x_fac = curl_root * 1.0;
	constexpr double finger_pxm_y_fac = curl_root * 1.0;

	constexpr double root_position_fac = other_root * 30;
	constexpr double hand_size_fac = other_root * 100;

	constexpr double hand_orientation_fac = other_root * 3;



	// std::cout << "hs " << hand.hand_size() << " " << last_hand.hand_size() << std::endl;
	residual[out_residual_idx++] = (hand.hand_size() - last_hand.hand_size()) * (T)(hand_size_fac);


	residual[out_residual_idx++] = (last_hand.wrist_location()[0] - hand.wrist_location()[0]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location()[1] - hand.wrist_location()[1]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location()[2] - hand.wrist_location()[2]) * root_position_fac;


	for (int i = 0; i < 3; i++) {
		residual[out_residual_idx++] = (hand.wrist_post_orientation_aax()[i]) * (T)(hand_orientation_fac);
	}



	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().swing()[0] - last_hand.thumb().metacarpal().swing()[0]) * thumb_mcp_swing_fac;
	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().swing()[1] - last_hand.thumb().metacarpal().swing()[1]) * thumb_mcp_swing_fac;
	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().twist() - last_hand.thumb().metacarpal().twist()) * thumb_mcp_twist_fac;

	residual[out_residual_idx++] = (hand.thumb().rots()[0] - last_hand.thumb().rots()[0]) * curl_root;
	residual[out_residual_idx++] = (hand.thumb().rots()[1] - last_hand.thumb().rots()[1]) * curl_root;



	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		OptimizerFinger<double> &finger_last = last_hand.finger(finger_idx);

		OptimizerFinger<T> &finger = hand.finger(finger_idx);

		residual[out_residual_idx++] =
		    (finger.metacarpal().swing()[0] - finger_last.metacarpal().swing()[0]) * finger_mcp_swing_fac;

		residual[out_residual_idx++] =
		    (finger.metacarpal().swing()[1] - finger_last.metacarpal().swing()[1]) * finger_mcp_swing_fac;



		residual[out_residual_idx++] =
		    (finger.metacarpal().twist() - finger_last.metacarpal().twist()) * finger_mcp_twist_fac;



		residual[out_residual_idx++] =
		    (finger.proximal_swing()[0] - finger_last.proximal_swing()[0]) * finger_pxm_x_fac;
		residual[out_residual_idx++] =
		    (finger.proximal_swing()[1] - finger_last.proximal_swing()[1]) * finger_pxm_y_fac;

		residual[out_residual_idx++] = (finger.rots()[0] - finger_last.rots()[0]) * curl_root;
		residual[out_residual_idx++] = (finger.rots()[1] - finger_last.rots()[1]) * curl_root;
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
		std::cout << "WEEEEOOOOOO ZERO LENGTH VECTOR" << std::endl;
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
void
unit_vector_to_stereographic(T *in, T *out)
{
	out[0] = in[0] / ((T)1 - in[2]);
	out[1] = in[1] / ((T)1 - in[2]);
}


template <typename T>
void
unit_xrt_vec3_to_stereographic(xrt_vec3 in, T *out)
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
diff(T *model_joint_pos,
     T move_joint_amount,
     StereographicObservation<double> &observation,
     float *confidences,
     double amount_we_care,
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


	double confidence = confidences[hand_joint_idx] * amount_we_care;
	double *observed_ray_sg = observation.obs[hand_joint_idx];

	out_residual[out_residual_idx++] = (stereographic_model_dir[0] - (T)(observed_ray_sg[0])) * confidence;
	out_residual[out_residual_idx++] = (stereographic_model_dir[1] - (T)(observed_ray_sg[1])) * confidence;

	hand_joint_idx++;
}



template <typename T>
void
CostFunctor_PositionsPart(OptimizerHand<T> &hand, LMKinematicHand &state, T *residual, int &out_residual_idx)
{

	T translations_absolute[5][5][3];
	// T orientations_absolute[5][5][4];

	double we_care_joint[] = {1.0, 0.2, 0.2, 1.4};
	double we_care_finger[] = {1.0, 1.0, 0.8, 0.8, 0.8};

	// eval_hand_with_orientation(hand, translations_absolute, orientations_absolute);
	eval_hand_fast(hand, translations_absolute);
	for (int view = 0; view < 2; view++) {
		T move_amt;

		if (view == 0) {
			// left camera.
			move_amt = (T)(dist_between_cameras / 2);
		} else {
			move_amt = -(T)(dist_between_cameras / 2);
		}
		int joint_acc_idx = 0;

		float *confidences = state.observation->views[view].confidences;



		diff(hand.wrist_location(), move_amt, state.sgo[view], confidences, 1.5, joint_acc_idx,
		     out_residual_idx, residual);


		for (int finger_idx = 0; finger_idx < 5; finger_idx++) {
			for (int joint_idx = 0; joint_idx < 4; joint_idx++) {
				diff(translations_absolute[finger_idx][joint_idx + 1], move_amt, state.sgo[view],
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

	Eigen::Vector<float, kFunctorInputDim> pose = state.TinyOptimizer_input.cast<float>();

	OptimizerHand<float> opt = {};
	OptimizerHandInit(opt, state.last_frame_pre_rotation);
	OptimizerHandUnpackFromVector(pose.data(), opt);

	float translations_absolute[5][5][3];
	float orientations_absolute[5][5][4];

	eval_hand_with_orientation(opt, translations_absolute, orientations_absolute);

	float post_wrist_orientation[4];

	ceres::AngleAxisToQuaternion(opt.wrist_post_orientation_aax(), post_wrist_orientation);

	float pre_wrist_orientation[4];
	for (int i = 0; i < 4; i++) {
		pre_wrist_orientation[i] = state.last_frame_pre_rotation[i];
	}

	float final_wrist_orientation[4];

	ceres::QuaternionProduct(pre_wrist_orientation, post_wrist_orientation, final_wrist_orientation);

	zldtt(opt.wrist_location(), final_wrist_orientation, out_viz_hand.wrist);

	for (int finger = 0; finger < 5; finger++) {
		for (int joint = 0; joint < 5; joint++) {
			zldtt(translations_absolute[finger][joint], orientations_absolute[finger][joint],
			      out_viz_hand.fingers[finger][joint]);
		}
	}
}

void
eval_to_viz_hand_2(LMKinematicHand *hand_ptr, hand_output &out_viz_hand)
{
	LMKinematicHand &state = *hand_ptr;


	Eigen::Vector<float, kFunctorInputDim> pose = state.TinyOptimizer_input.cast<float>();

	OptimizerHand<float> opt = {};
	OptimizerHandInit(opt, state.last_frame_pre_rotation);
	OptimizerHandUnpackFromVector(pose.data(), opt);

	float translations_absolute[5][5][3] = {};



	eval_hand_fast(opt, translations_absolute);


	float identity_orientation[4] = {1, 0, 0, 0};

	zldtt(opt.wrist_location(), identity_orientation, out_viz_hand.wrist);

	for (int finger = 0; finger < 5; finger++) {
		for (int joint = 0; joint < 5; joint++) {
			zldtt(translations_absolute[finger][joint], identity_orientation,
			      out_viz_hand.fingers[finger][joint]);
		}
	}
}

void
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


	using AutoDiffCostFunctor = ceres::TinySolverAutoDiffFunction<CostFunctor, kHandResidualSize, kFunctorInputDim>;

	AutoDiffCostFunctor f(cf);

	ceres::TinySolver<AutoDiffCostFunctor> solver; // do NOT initialize to = {}, apparently.
	solver.options.max_num_iterations = 10000;
	solver.options.function_tolerance = 1e-6;
	// solver.options.function_tolerance = 1e-6;

	// Eigen::Vector<double, kFunctorInputDim> pose_2 = {};
	// pose_2.setZero();



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


	eval_to_viz_hand_2(&state, out_viz_hand);
}


void
give_identity_hand(LMKinematicHand *hand, struct hand_output &out_viz_hand)
{
	eval_to_viz_hand(*hand, out_viz_hand);
}

void
create_kinematic_hand(double camera_baseline, LMKinematicHand **out_kinematic_hand)
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
