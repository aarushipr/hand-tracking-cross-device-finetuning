// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "os/os_time.h"
#include "util/u_logging.h"

#include "stereokit.h"
#include "stereokit_ui.h"
using namespace sk;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/tiny_solver.h"
#include "ceres/tiny_solver_autodiff_function.h"

// #include "glog/logging.h"


#include <cmath>
#include <random>
#include "randoviz.hpp"



#include "os/os_time.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"

#include "OptimizerParams_SmartNotClever.hpp"
#include "do_json.hpp"


struct pgm_state;

struct CostFunctor
{
	// pgm_state *state;
	xrt_vec3 *target_joints;
	OptimizerHand<double> &last_hand;


	template <typename T>
	bool
	operator()(const T *const x, T *residual) const;

	CostFunctor(xrt_vec3 *in_target, OptimizerHand<double> &in_last_hand)
	    : target_joints(in_target), last_hand(in_last_hand)
	{}
};


struct pgm_state
{
	ui_state uis = {};
	size_t frame_idx = 0;

	xrt_vec3 *target_jts = NULL;

	Eigen::Vector<double, kFunctorInputDim> last_pose;
	OptimizerHand<double> last_pose_hand;


	float translations_absolute[1000][5][5][3];
	float orientations_absolute[1000][5][5][4];

	bool first_frame = true;
	pgm_state()
	{
		last_pose.setZero();
		OptimizerHandUnpackFromVector(last_pose.data(), last_pose_hand);
	}
};



template <typename T>
void
swing_twist_to_quat(T swing[2], T &twist, T out_quat[4])
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

// no! do this in 2D!
template <typename T>
void
curl_to_quat(T &curl, T out_quat[4])
{
	T curl_aax[3] = {curl, (T)(0), (T)(0)};

	ceres::AngleAxisToQuaternion(curl_aax, out_quat);
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
	// OptimizerHand<T> opt;
	// OptimizerHandUnpackFromVector(in_pose, opt);

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
	ceres::AngleAxisToQuaternion(opt.wrist_orientation_aax(), orientation_root);

	// std::cout << __PRETTY_FUNCTION__ << " " << opt.wrist_location()[0] << " " << opt.wrist_location()[1] << " "
	//           << opt.wrist_location()[2] << std::endl;

	// 	std::cout << __PRETTY_FUNCTION__ << " " << opt.hand_size() << std::endl;

	// U_LOG_E("root %f %f %f %f", orientation_root[0], orientation_root[1], orientation_root[2],
	// orientation_root[3]);



	for (int finger = 0; finger < 5; finger++) {
		T *last_translation = opt.wrist_location();
		T *last_orientation = orientation_root;
		for (int bone = 0; bone < 5; bone++) {
			T *out_translation = translations_absolute[finger][bone];
			T *out_orientation = orientations_absolute[finger][bone];

			T *rel_translation = rel_translations[finger][bone];
			T *rel_orientation = rel_orientations[finger][bone];

			ceres::QuaternionProduct(last_orientation, rel_orientation, out_orientation);

			// U_LOG_E("rel %d %d %f %f %f %f", finger, bone, rel_orientation[0], rel_orientation[1],
			//         rel_orientation[2], rel_orientation[3]);

			// U_LOG_E("%f %f %f %f", out_orientation[0], out_orientation[1], out_orientation[2],
			//         out_orientation[3]);
			ceres::QuaternionRotatePoint(last_orientation, rel_translation, out_translation);

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
diff(xrt_vec3 tgt, T *curr, double amt_we_care, T *out_residual, int &out_idx)
{
	out_residual[out_idx++] = (curr[0] - (T)(tgt.x)) * amt_we_care;
	out_residual[out_idx++] = (curr[1] - (T)(tgt.y)) * amt_we_care;
	out_residual[out_idx++] = (curr[2] - (T)(tgt.z)) * amt_we_care;

	// out_residual[out_idx++] = (curr[0] - (T)(tgt.x));
	// out_residual[out_idx++] = (curr[1] - (T)(tgt.y));
	// out_residual[out_idx++] = (curr[2] - (T)(tgt.z));
}

template <typename T>
void
CostFunctor_PositionsPart(OptimizerHand<T> &hand, xrt_vec3 *target_joints, T *residual, int &out_residual_idx)
{

	T translations_absolute[5][5][3];
	T orientations_absolute[5][5][4];

	double we_care_joint[] = {1.0, 0.2, 0.2, 1.4};
	double we_care_finger[] = {1.0, 1.0, 0.8, 0.8, 0.8};

	eval_hand_with_orientation(hand, translations_absolute, orientations_absolute);
	int joint_acc_idx = 0;

	diff(target_joints[joint_acc_idx++], hand.wrist_location(), 1.5f, residual, out_residual_idx);


	for (int finger_idx = 0; finger_idx < 5; finger_idx++) {
		for (int joint_idx = 0; joint_idx < 4; joint_idx++) {
			diff(target_joints[joint_acc_idx++], translations_absolute[finger_idx][joint_idx + 1],
			     we_care_joint[joint_idx] * we_care_finger[finger_idx], residual, out_residual_idx);
		}
	}
}

template <typename T>
void
CostFunctor_StabilityPart(OptimizerHand<T> &hand, OptimizerHand<double> &last_hand, T *residual, int &out_residual_idx)
{
	// std::cout << "hs " << hand.hand_size() << " " << last_hand.hand_size() << std::endl;
	residual[out_residual_idx++] = (hand.hand_size() - last_hand.hand_size()) * (T)(100);

	// const double curl_fac = 0.05;
	// const double curl_fac = 0.02;
	const double curl_fac = 0.04;
	// const double curl_fac = 0.2;
	const double thumb_mcp_y_fac = curl_fac * 1.5;

	const double finger_mcp_rot_fac = curl_fac * 3.0;
	const double finger_pxm_y_fac = curl_fac * 1.5;


	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().swing()[0] - last_hand.thumb().metacarpal().swing()[0]) * curl_fac;
	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().swing()[1] - last_hand.thumb().metacarpal().swing()[1]) * thumb_mcp_y_fac;
	residual[out_residual_idx++] =
	    (hand.thumb().metacarpal().twist() - last_hand.thumb().metacarpal().twist()) * curl_fac;
	residual[out_residual_idx++] = (hand.thumb().rots()[0] - last_hand.thumb().rots()[0]) * curl_fac;
	residual[out_residual_idx++] = (hand.thumb().rots()[1] - last_hand.thumb().rots()[1]) * curl_fac;



	for (int finger_idx = 0; finger_idx < 4; finger_idx++) {
		OptimizerFinger<double> &finger_last = last_hand.finger(finger_idx);

		OptimizerFinger<T> &finger = hand.finger(finger_idx);

		residual[out_residual_idx++] =
		    (finger.metacarpal().swing()[0] - finger_last.metacarpal().swing()[0]) * finger_mcp_rot_fac;

		residual[out_residual_idx++] =
		    (finger.metacarpal().swing()[1] - finger_last.metacarpal().swing()[1]) * finger_mcp_rot_fac;



		residual[out_residual_idx++] =
		    (finger.metacarpal().twist() - finger_last.metacarpal().twist()) * finger_mcp_rot_fac;



		residual[out_residual_idx++] =
		    (finger.proximal_swing()[0] - finger_last.proximal_swing()[0]) * curl_fac;
		residual[out_residual_idx++] =
		    (finger.proximal_swing()[1] - finger_last.proximal_swing()[1]) * finger_pxm_y_fac;

		residual[out_residual_idx++] = (finger.rots()[0] - finger_last.rots()[0]) * curl_fac;
		residual[out_residual_idx++] = (finger.rots()[1] - finger_last.rots()[1]) * curl_fac;
	}

	double quat_last[4];
	T quat_curr[4];

	ceres::AngleAxisToQuaternion(hand.wrist_orientation_aax(), quat_curr);
	ceres::AngleAxisToQuaternion(last_hand.wrist_orientation_aax(), quat_last);



	// https://math.stackexchange.com/a/90098
	T dot = (T)(0);


	dot += quat_last[0] * quat_curr[0];
	dot += quat_last[1] * quat_curr[1];
	dot += quat_last[2] * quat_curr[2];
	dot += quat_last[3] * quat_curr[3];

	// between
	// Too lazy to figure out power in Ceres-land
	T diff = (T)(1) - (dot * dot);

	residual[out_residual_idx++] = diff * curl_fac * (T)(2);

	double root_position_fac = curl_fac * 30;

	residual[out_residual_idx++] = (last_hand.wrist_location()[0] - hand.wrist_location()[0]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location()[1] - hand.wrist_location()[1]) * root_position_fac;
	residual[out_residual_idx++] = (last_hand.wrist_location()[2] - hand.wrist_location()[2]) * root_position_fac;
}

template <typename T>
bool
CostFunctor::operator()(const T *const x, T *residual) const
{

	OptimizerHand<T> hand = {};
	OptimizerHandInit<T>(hand);
	OptimizerHandUnpackFromVector(x, hand);

	for (int i = 0; i < kHandResidualSize; i++) {
		residual[i] = (T)(0);
	}

	int out_residual_idx = 0;

	CostFunctor_PositionsPart(hand, this->target_joints, residual, out_residual_idx);

	CostFunctor_StabilityPart(hand, this->last_hand, residual, out_residual_idx);

	// std::cout << out_residual_idx << std::endl;

	// for (int i = 0; i < kHandResidualSize; i++) {
	// 	std::cout << residual[i] << std::endl;
	// }

	// std::cout << std::endl << std::endl << std::endl;

	// std::cout << Eigen::Wrap<Eigen::Vector<T, kHandResidualSize>>(residual) << std::endl;



	return true;
}



double
do_it(pgm_state &state)
{
	CostFunctor cf(state.target_jts, state.last_pose_hand);

	using AutoDiffCostFunctor = ceres::TinySolverAutoDiffFunction<CostFunctor, kHandResidualSize, kFunctorInputDim>;

	AutoDiffCostFunctor f(cf);

	ceres::TinySolver<AutoDiffCostFunctor> solver; // do NOT initialize to = {}, apparently.
	solver.options.max_num_iterations = 100;
	solver.options.function_tolerance = 1e-6;
	// solver.options.function_tolerance = 1e-6;

	// Eigen::Vector<double, kFunctorInputDim> pose_2 = {};
	// pose_2.setZero();



	// std::cout << "before " << pose_2 << std::endl;

	uint64_t start = os_monotonic_get_ns();
	auto summary = solver.Solve(f, &state.last_pose);
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

	std::cout << "Status: " << status << " max norm " << summary.gradient_max_norm << " gtol "
	          << solver.options.gradient_tolerance << std::endl;
	U_LOG_E("Took %f ms", time_taken);

	// std::cout << "after " << pose_2 << std::endl;
	OptimizerHandUnpackFromVector(state.last_pose.data(), state.last_pose_hand);



	// std::cout << summary.iterations << std::endl;


	Eigen::Vector<float, kFunctorInputDim> pose = state.last_pose.cast<float>();

	// std::cout << pose << std::endl;

	OptimizerHand<float> opt = {};
	OptimizerHandInit(opt);
	OptimizerHandUnpackFromVector(pose.data(), opt);

	eval_hand_with_orientation(opt, state.translations_absolute[state.frame_idx],
	                           state.orientations_absolute[state.frame_idx]);

	// pose = pose_2;



	return time_taken;
}

int times_done = 0;
double sum_of_times = 0;

bool
do_the_thing(struct pgm_state &state)
{
	state.target_jts = state.uis.frames[state.frame_idx].jts;

	times_done += 1;
	sum_of_times += do_it(state);


	do {
		state.frame_idx++;
		if (state.frame_idx > 560) {
			// if (state.frame_idx > 150) {
			return false;
		}

	} while (state.uis.frames[state.frame_idx].num_hands == 0);

	return true;
}



void
draw_axis_at_pt(const sk::vec3 &vec)
{
	// We could take the matrix and plop it into a sk::matrix,
	// But that's a pretty annoying idea.
	// Basically I REALLY don't trust SK's matrix memory layout, and I kinda don't trust mine.
	// So copying is confusing, I don't want to think about it.
	// Instead, I want to go to a representation that's well-known - just the position and orientation in the OpenXR
	// coordinate space. I trust StereoKit to get this right, I don't totally know if my code gets it right, but by
	// the time I'm done it better.

	sk::pose_t pose = sk::pose_identity;
	pose.position = vec;
	draw_axis(pose, 0.01f);
	// text_from_vec3(pose.position, 0.1, name);
}

void
draw_axis_at_pose(const sk::vec3 &vec, float *ori, const char *name)
{
	// We could take the matrix and plop it into a sk::matrix,
	// But that's a pretty annoying idea.
	// Basically I REALLY don't trust SK's matrix memory layout, and I kinda don't trust mine.
	// So copying is confusing, I don't want to think about it.
	// Instead, I want to go to a representation that's well-known - just the position and orientation in the OpenXR
	// coordinate space. I trust StereoKit to get this right, I don't totally know if my code gets it right, but by
	// the time I'm done it better.

	sk::pose_t pose;
	pose.position = vec;

	pose.orientation.w = ori[0];
	pose.orientation.x = ori[1];
	pose.orientation.y = ori[2];
	pose.orientation.z = ori[3];

	draw_axis(pose, 0.01f);
	text_from_vec3(pose.position, 0.1, name);
}


static const char *kine_keys[5][5] = {

    {
        "UHHHHHNO",
        "THMB_MCP",
        "THMB_PXM",
        "THMB_DST",
        "THMB_TIP",
    },

    {
        "INDX_MCP",
        "INDX_PXM",
        "INDX_INT",
        "INDX_DST",
        "INDX_TIP",
    },

    {
        "MIDL_MCP",
        "MIDL_PXM",
        "MIDL_INT",
        "MIDL_DST",
        "MIDL_TIP",
    },

    {
        "RING_MCP",
        "RING_PXM",
        "RING_INT",
        "RING_DST",
        "RING_TIP",
    },

    {
        "LITL_MCP",
        "LITL_PXM",
        "LITL_INT",
        "LITL_DST",
        "LITL_TIP",
    },
};

void
kh4f_disp_to_user(float translations[5][5][3], float orientations[5][5][4])
{
	// draw_axis_at_matrix(&hand.wrist_relation, "WRIST");



	draw_axis_at_pt({0, 0, 0});

	for (int finger_idx = 0; finger_idx < 5; finger_idx++) {

		for (int bone_idx = 0; bone_idx < 5; bone_idx++) {
			if (finger_idx == 0 && bone_idx == 0) {
				// Hidden extra finger joint
				continue;
			}
			// if (!(finger_idx == 0 && bone_idx == 0)) {
			sk::vec3 *pt = (sk::vec3 *)translations[finger_idx][bone_idx];
			float *ori = orientations[finger_idx][bone_idx];
			draw_axis_at_pose(*pt, ori, kine_keys[finger_idx][bone_idx]);
			// }
		}

// Implement later: hand lines.
#if 0
		for (int bone_idx = 0; bone_idx < 4; bone_idx++) {
			if (finger_idx == 0 && bone_idx == 0) {
				// Hidden extra finger joint
				continue;
			}
			sk::vec3 joint0;
			sk::vec3 joint1;

			Eigen::Vector3f j0 = hand.bones[finger_idx][bone_idx].world_pose.translation().cast<float>();
			Eigen::Vector3f j1 =
			    hand.bones[finger_idx][bone_idx + 1].world_pose.translation().cast<float>();

			joint0.x = j0.x();
			joint0.y = j0.y();
			joint0.z = j0.z();

			joint1.x = j1.x();
			joint1.y = j1.y();
			joint1.z = j1.z();


			float hue0 = ((finger_idx * 4) + bone_idx) / 24.0f;
			float hue1 = ((finger_idx * 4) + bone_idx + 1) / 24.0f;
			line_add(joint0, joint1, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
			         color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.0005);

			// if (!(finger_idx == 0 && bone_idx == 0)) {
			// 	draw_axis_at_matrix(&hand.bones[finger_idx][bone_idx].world_pose,
			// 	                    kine_keys[finger_idx][bone_idx]);
			// }
		}
#endif
	}
}

void
render_jts(ui_state *uis, sk::vec3 *jts, sk::pose_t pose, int num = 5)
{

	hierarchy_push(pose_matrix(pose));
	for (int i = 0; i < num; i++) {
		render_add_model(uis->sphere, matrix_trs(jts[i]), color_hsv((float)i / (float)num, 1.0f, 1.0f, .5f));
	}
	hierarchy_pop();
}


void
step(void *ptr)
{
	struct pgm_state &state = *(pgm_state *)ptr;

	sk::matrix forward = sk::matrix_t({0, 0, -0.2});

	sk::hierarchy_push(forward);


	bool should_reinit = state.first_frame;

	if (input_key(sk::key_f) & sk::button_state_just_active) {
		state.uis.disp_gt = !state.uis.disp_gt;
	}


	if (sk::input_key(sk::key_right) & sk::button_state_just_active) {
		size_t &f_idx = state.frame_idx;
		do {
			f_idx++;
			if (f_idx == 600) {
				f_idx = 0;
			}
		} while (state.uis.frames[f_idx].num_hands == 0);
		should_reinit = true;
	}

	if (sk::input_key(sk::key_left) & sk::button_state_just_active) {
		size_t &f_idx = state.frame_idx;
		do {
			f_idx--;
			if (f_idx == 0) {
				f_idx = 599;
			}
		} while (state.uis.frames[f_idx].num_hands == 0);
		should_reinit = true;
	}

	if (should_reinit) {
		U_LOG_E("frame_idx %zu", state.frame_idx);
		state.target_jts = state.uis.frames[state.frame_idx].jts;
	}

	if (sk::input_key(sk::key_down) & sk::button_state_just_inactive) {
		do_it(state);
	}



	kh4f_disp_to_user(state.translations_absolute[state.frame_idx], state.orientations_absolute[state.frame_idx]);


	if (state.uis.disp_gt) {
		render_jts(&state.uis, (sk::vec3 *)state.uis.frames[state.frame_idx].jts, sk::pose_identity, 21);
	}
	sk::hierarchy_pop();
	state.first_frame = false;
}

void
shutdown(void *ptr)
{}

int
main(int argc, char *argv[])
{
	struct pgm_state state;

	// print

	// state.hi.resize(kHandDim);

	do_json_thing(&state.uis);

	// two();
	u_trace_marker_init();
	sk_settings_t settings = {};
	settings.app_name = "StereoKit C";
	settings.assets_folder = "/2/XR/sk-gradient-descent/Assets";
	// settings.display_preference = display_mode_flatscreen;
	settings.display_preference = display_mode_mixedreality;

	settings.overlay_app = true;
	settings.overlay_priority = 1;
	if (!sk_init(settings))
		return 1;

	render_enable_skytex(false);

	do_json_thing(&state.uis);

	state.uis.sp_material = material_copy_id("default/material");

	material_set_transparency(state.uis.sp_material, sk::transparency_add);
	material_set_color(state.uis.sp_material, "color", {.4, .4, .4, 1.0f});

	state.uis.sphere = model_create_mesh(mesh_gen_sphere(0.013f), state.uis.sp_material);


	// state.parameter_vec.setZero();
	// std::cout << state.parameter_vec << std::endl;
	// // std::cout << &state.hi(0) << " " << &state.opt.hand_size() << &state.opt.thumb().rots()(0);
	// U_LOG_E("%p %p %p", &state.parameter_vec(0), &state.opt.hand_size(), &state.opt.thumb().rots()(0));
	// OptimizerHandInit(state.opt);

	// PODHand_init(state.pod);
	// PODCombineWithOpt(state.pod, state.opt);
	// POD_init_world_poses(state.pod);

	const int start_point = 0;

	state.frame_idx = start_point;

	while (do_the_thing(state)) {
		// do_the_thing does the thing, let it do the thing till it is done doing the thing
	}
	state.frame_idx = start_point;

	U_LOG_E("On average, took %f ms per iteration", sum_of_times / (double)times_done);



	float hand_vector[kFunctorInputDim];

	OptimizerHand<float> opt = {};
	OptimizerHandInit(opt);
	// opt.wrist_location()[2] = 1.0f;
	OptimizerHandPackIntoVector(opt, hand_vector);


	eval_hand_with_orientation(opt, state.translations_absolute[state.frame_idx],
	                           state.orientations_absolute[state.frame_idx]);



	// update_pts(state, state.gt);

	// update_pts(state, state.recovered);


	sk_run_data(step, (void *)&state, shutdown, (void *)&state);


	return 0;
}