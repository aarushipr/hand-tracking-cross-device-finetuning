#include <filesystem>
#include <iostream>
#include <sstream>
#include <random>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "math/m_eigen_interop.hpp"

#include "math/m_api.h"
#include "math/m_space.h"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "shls/csv.hpp"


#include "aux/u_random_distribution.h"

#include "util/u_debug.h"
#include "math/m_vec3.h"
#include "math/m_vec2.h"
#include "dataloader_common.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;



struct ElbowResult
{
	xrt_vec2 norm_finite_diff_direction_px_coord;
	float dotprod_elbowdir_wristloc;
};


template <typename... Args>
std::string
string_format(const std::string &format, Args... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size_s <= 0) {
		throw std::runtime_error("Error during formatting.");
	}
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

std::string
leftpadfour(int i)
{
	std::string str = string_format("%04d", i);
	return str;
}



xrt_vec3
center_of_bounding_cuboid(hand26 &joints)
{
	xrt_vec3 min = joints[0].position;
	xrt_vec3 max = joints[0].position;

	for (int i = 0; i < 25; i++) {
		xrt_vec3 &pt = joints[i].position;
		min.x = fmin(pt.x, min.x);
		min.y = fmin(pt.y, min.y);
		min.z = fmin(pt.z, min.z);

		max.x = fmax(pt.x, max.x);
		max.y = fmax(pt.y, max.y);
		max.z = fmax(pt.z, max.z);
	}

	return m_vec3_mul_scalar((min + max), 0.5);
}

void
predict_points(hand26 two_frames_ago, hand26 one_frame_ago, hand26 &out)
{
	for (int i = 0; i < 25; i++) {
		xrt_vec3 from_to = one_frame_ago[i].position - two_frames_ago[i].position;
		out[i].position = one_frame_ago[i].position + from_to;
	}
}

void
predict_points_scaled(hand26 many_frames_ago, hand26 one_frame_ago, int num_frames_between_one_and_many, hand26 &out)
{
	for (int i = 0; i < 25; i++) {
		xrt_vec3 from_to = one_frame_ago[i].position - many_frames_ago[i].position;
		from_to = from_to * (1.0f / num_frames_between_one_and_many);
		out[i].position = one_frame_ago[i].position + from_to;
	}
}


void
rotate_points_into_local(xrt_pose camera_pose, hand26 &global, hand26 &out_local)
{
	xrt_quat camposinv;
	math_quat_invert(&camera_pose.orientation, &camposinv);
	for (int i = 0; i < 26; i++) {
		xrt_vec3 fingerpose_global = global[i].position;

		xrt_vec3 fingerpose_local = fingerpose_global - camera_pose.position;



		math_quat_rotate_vec3(&camposinv, &fingerpose_local, &out_local[i].position);
		math_quat_rotate(&camposinv, &global[i].orientation, &out_local[i].orientation);
	}
}

void
homothety(hand26 &global, float scale)
{
	Eigen::Vector3f center = xrt::auxiliary::math::map_vec3(global[11].position);

	Eigen::Affine3f scale_mat = Eigen::Affine3f::Identity();
	scale_mat.linear() *= scale;

	Eigen::Affine3f translation = Eigen::Affine3f::Identity();

	translation.translation() = center * (1 - scale);

	Eigen::Affine3f transform = scale_mat * translation;

	for (int i = 0; i < 25; i++) {
		xrt::auxiliary::math::map_vec3(global[i].position) =
		    transform * xrt::auxiliary::math::map_vec3(global[i].position);
	}
}

void
add_rel_depth(const hand26 &hand, hand25_2d &out_joints_in_img)
{
	float hand_size = hand_length(hand);
	float midpxm_depth = m_vec3_len(hand[11].position);
	for (int i = 0; i < 25; i++) {
		float jd = m_vec3_len(hand[i].position);
		out_joints_in_img[i].depth_relative_to_midpxm = (jd - midpxm_depth) / hand_size;
	}
}

float
normalf(std::mt19937 &mt, float center, float stddev)
{
	std::normal_distribution<float> rd(center, stddev);
	return rd(mt);
}

void
randommove(hand26 &hand, float variance_overall = 0.05, float variance_per_joint = 0.01)
{
	std::random_device dev;
	std::mt19937 mt = std::mt19937(dev());


	xrt_vec3 overall;

	overall.x = normalf(mt, 0, variance_overall);
	overall.y = normalf(mt, 0, variance_overall);
	overall.z = normalf(mt, 0, variance_overall);

	for (int i = 0; i < 25; i++) {
		hand[i].position.x += overall.x + normalf(mt, 0, variance_per_joint);
		hand[i].position.y += overall.y + normalf(mt, 0, variance_per_joint);
		hand[i].position.z += overall.z + normalf(mt, 0, variance_per_joint);
	}
}

float
pront(hand26 &hand_gt_in_camera)
{
	xrt_vec3 wrist = hand_gt_in_camera[0].position;
	xrt_vec3 elbow = hand_gt_in_camera[25].position;

	xrt_vec3 dir_to_wrist = wrist; // heeehee
	xrt_vec3 dir_wrist_to_elbow = elbow - wrist;

	// These could be replaced iwth probably a well-placed len_sqrd
	// But I don't care.
	math_vec3_normalize(&dir_to_wrist);
	math_vec3_normalize(&dir_wrist_to_elbow);

	return m_vec3_dot(dir_to_wrist, dir_wrist_to_elbow);
}


hand26
copy_to_hand26(py::array_t<float> joints_sequence, int idx_within_sequence)
{
	hand26 out;
	for (int i = 0; i < 26; i++) {
		int root = i * 7;
		out[i].position.x = joints_sequence.at(idx_within_sequence, root + 0);
		out[i].position.y = joints_sequence.at(idx_within_sequence, root + 1);
		out[i].position.z = joints_sequence.at(idx_within_sequence, root + 2);

		out[i].orientation.w = joints_sequence.at(idx_within_sequence, root + 3);
		out[i].orientation.x = joints_sequence.at(idx_within_sequence, root + 4);
		out[i].orientation.y = joints_sequence.at(idx_within_sequence, root + 5);
		out[i].orientation.z = joints_sequence.at(idx_within_sequence, root + 6);
	}
	return out;
}

int
prepare_sample(std::string &img_file,                        //
               std::string &mask_file,                       //
               py::array_t<float> joints_sequence,           //
               py::array_t<float> camera_info_sequence,      //
               int idx_within_sequence,                      //
               py::array_t<float> out_joints_gt,             //
               py::array_t<float> out_joints_pose_predicted, //
               py::array_t<uint8_t> out_image,               //
               py::array_t<uint8_t> out_mask,                //
               py::array_t<float> out_elbow,                 //
               py::array_t<float> out_curls)
{

	single_frame our_frame = {};

	our_frame.camera_pose.position.x = camera_info_sequence.at(idx_within_sequence, 0);
	our_frame.camera_pose.position.y = camera_info_sequence.at(idx_within_sequence, 1);
	our_frame.camera_pose.position.z = camera_info_sequence.at(idx_within_sequence, 2);


	our_frame.camera_pose.orientation.x = camera_info_sequence.at(idx_within_sequence, 3);
	our_frame.camera_pose.orientation.y = camera_info_sequence.at(idx_within_sequence, 4);
	our_frame.camera_pose.orientation.z = camera_info_sequence.at(idx_within_sequence, 5);
	our_frame.camera_pose.orientation.w = camera_info_sequence.at(idx_within_sequence, 6);

	// 7 is the width of a quaternion
	our_frame.camera_info.fx = camera_info_sequence.at(idx_within_sequence, 7 + 0);
	our_frame.camera_info.fy = camera_info_sequence.at(idx_within_sequence, 7 + 1);
	our_frame.camera_info.cx = camera_info_sequence.at(idx_within_sequence, 7 + 2);
	our_frame.camera_info.cy = camera_info_sequence.at(idx_within_sequence, 7 + 3);

	bool must_not_predict = idx_within_sequence == 0;
	bool can_predict_1 = idx_within_sequence >= 1;
	bool can_predict_2 = idx_within_sequence >= 2;

	bool can_predict_25 = idx_within_sequence >= 25;

	our_frame.move_overall_variance = 0.015f;
	our_frame.move_per_joint_variance = 0.01f;
	float homothety_min =
	    0.88f; // Intentionally smaller than the other side, to account for per_joint_variance making us bigger
	float homothety_max = 1.1f;



	bool use_predict = u_random_distribution_get_sample_float(0.0f, 1.0f) < 0.8;
	use_predict = use_predict && !(must_not_predict);

	int prediction_type = 0;

	hand26 ground_truth_global = copy_to_hand26(joints_sequence, idx_within_sequence);
	hand26 predicted_global;

	if (use_predict) {
		if (can_predict_25 && (u_random_distribution_get_sample_float(0.0f, 1.0f) < 0.05)) {

			predict_points_scaled(copy_to_hand26(joints_sequence, idx_within_sequence - 25),
			                      copy_to_hand26(joints_sequence, idx_within_sequence - 1), 24,
			                      predicted_global);
			prediction_type = 3;
		} else if (can_predict_2) {
			predict_points(copy_to_hand26(joints_sequence, idx_within_sequence - 2),
			               copy_to_hand26(joints_sequence, idx_within_sequence - 1), predicted_global);
			prediction_type = 2;
		} else if (can_predict_1) {
			predicted_global = copy_to_hand26(joints_sequence, idx_within_sequence - 1);
			prediction_type = 1;
		}
	} else {
		prediction_type = 0;
		predicted_global = ground_truth_global;
		our_frame.move_overall_variance = 0.04f;
		our_frame.move_per_joint_variance = 0.03f;
		homothety_max = 1.3f;
	}

	hand26 ground_truth_local;
	hand26 predicted_local;

	rotate_points_into_local(our_frame.camera_pose, ground_truth_global, ground_truth_local);
	rotate_points_into_local(our_frame.camera_pose, predicted_global, predicted_local);


	float homothety_scale = u_random_distribution_get_sample_float(homothety_min, homothety_max);
	homothety(predicted_local, homothety_scale);

	float pja_1 = u_random_distribution_get_sample_float(0.5, 1.1);
	float pja_2 = u_random_distribution_get_sample_float(0.5, 1.0);
	float pja_3 = u_random_distribution_get_sample_float(0.5, 1.0);
	float per_joint_amt = fmaxf(fmaxf(pja_1, pja_2), pja_3);

	our_frame.move_per_joint_variance *= per_joint_amt;
	our_frame.move_overall_variance *= u_random_distribution_get_sample_float(0.8, 1.0);

	randommove(predicted_local, our_frame.move_overall_variance, our_frame.move_per_joint_variance);

	xrt_vec3 direction = center_of_bounding_cuboid(predicted_local);


	float twist = u_random_distribution_get_sample_float(-M_PI, M_PI);

	float expand_val_c = 1.65;
	float expand_val_r = 0.2;
	float sg_expand_val_change_from_average =
	    u_random_distribution_get_sample_float(expand_val_c - expand_val_r, expand_val_c + expand_val_r) /
	    expand_val_c;

	float sg_expand_val = 1.65 * sg_expand_val_change_from_average;
	float sg_palm_expand_val = 2.2 * sg_expand_val_change_from_average;

	// U_LOG_E("%f %f %f", sg_expand_val_change_from_average, sg_expand_val, sg_palm_expand_val);

	ElbowResult elbow_result;
	elbow_result.dotprod_elbowdir_wristloc = pront(ground_truth_local);


	std::vector<cv::Mat> color_alpha_in = {};
	std::vector<cv::Mat> color_alpha_out = {};

	color_alpha_out.push_back(cv::Mat());
	color_alpha_out.push_back(cv::Mat());

	////

	cv::Mat mat = cv::imread(img_file);
	color_alpha_in.push_back(mat);
	///

	cv::Mat maybe_alpha = {}; // nervous about scoping stuff so aaa
	bool alphaexists = mask_file.length() != 0;

	if (alphaexists) {
		maybe_alpha = cv::imread(mask_file);
		color_alpha_in.push_back(maybe_alpha);
	}

	hand25_2d px_coord_gt = {};
	hand25_2d px_coord_pred = {};


	float stereographic_radius;

	///
	distort_image(color_alpha_in,                                   //
	              direction,                                        //
	              cv::Scalar{0, 255, 0},                            //
	              our_frame.camera_info,                            //
	              twist,                                            //
	              sg_expand_val,                                    //
	              sg_palm_expand_val,                               //
	              ground_truth_local,                               //
	              predicted_local,                                  //
	              px_coord_gt,                                      //
	              px_coord_pred,                                    //
	              elbow_result.norm_finite_diff_direction_px_coord, //
	              stereographic_radius,                             //
	              color_alpha_out);



	U_LOG_D("%f %f %f", elbow_result.norm_finite_diff_direction_px_coord.x,
	        elbow_result.norm_finite_diff_direction_px_coord.y, elbow_result.dotprod_elbowdir_wristloc);

	float tglenxy = sqrtf(1.0f - (elbow_result.dotprod_elbowdir_wristloc * elbow_result.dotprod_elbowdir_wristloc));

	float actlenxy = m_vec2_len(elbow_result.norm_finite_diff_direction_px_coord);


	elbow_result.norm_finite_diff_direction_px_coord =
	    m_vec2_mul_scalar(elbow_result.norm_finite_diff_direction_px_coord, tglenxy / actlenxy);

	U_LOG_D("%f %f %f", elbow_result.norm_finite_diff_direction_px_coord.x,
	        elbow_result.norm_finite_diff_direction_px_coord.y, elbow_result.dotprod_elbowdir_wristloc);
	float len =
	    elbow_result.norm_finite_diff_direction_px_coord.x * elbow_result.norm_finite_diff_direction_px_coord.x + //
	    elbow_result.norm_finite_diff_direction_px_coord.y * elbow_result.norm_finite_diff_direction_px_coord.y + //
	    elbow_result.dotprod_elbowdir_wristloc * elbow_result.dotprod_elbowdir_wristloc;
	U_LOG_D("%f", len);



	add_rel_depth(ground_truth_local, px_coord_gt);
	add_rel_depth(predicted_local, px_coord_pred);

	std::array<float, 5> curls = {};

	hand_curls(ground_truth_local, curls);


	for (int v = 0; v < 128; v++) {
		for (int u = 0; u < 128; u++) {
			out_image.mutable_at(v, u) = color_alpha_out[0].at<uint8_t>(v, u);
			if (alphaexists) {
				out_mask.mutable_at(v, u) = color_alpha_out[1].at<uint8_t>(v, u);
			}
		}
	}

	for (int i = 0; i < 25; i++) {
		out_joints_gt.mutable_at(i, 0) = px_coord_gt[i].x;
		out_joints_gt.mutable_at(i, 1) = px_coord_gt[i].y;
		out_joints_gt.mutable_at(i, 2) = px_coord_gt[i].depth_relative_to_midpxm;

		out_joints_pose_predicted.mutable_at(i, 0) = px_coord_pred[i].x;
		out_joints_pose_predicted.mutable_at(i, 1) = px_coord_pred[i].y;
		out_joints_pose_predicted.mutable_at(i, 2) = px_coord_pred[i].depth_relative_to_midpxm;
	}

	out_elbow.mutable_at(0) = elbow_result.norm_finite_diff_direction_px_coord.x;
	out_elbow.mutable_at(1) = elbow_result.norm_finite_diff_direction_px_coord.y;
	out_elbow.mutable_at(2) = elbow_result.dotprod_elbowdir_wristloc;


	for (int i = 0; i < 5; i++) {
		out_curls.mutable_at(i) = curls[i];
	}

	return 0;
}

PYBIND11_MODULE(ad4_stereographic_projection, m)
{
	m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("prepare_sample", &prepare_sample, "Prepare sample");
}