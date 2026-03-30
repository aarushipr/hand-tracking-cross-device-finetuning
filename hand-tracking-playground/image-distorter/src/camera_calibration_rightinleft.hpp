#include <cmath>
#include <stdio.h>

#include <unistd.h>
#include "math/m_api.h"
#include "math/m_vec3.h"

#include "util/u_time.h"

#include "xrt/xrt_defines.h"
#include "math/m_space.h"
#include <filesystem>
#include <fstream>
#include "os/os_time.h"
#include "util/u_logging.h"
#include "tracking/t_tracking.h"

#include "tracking/t_calibration_opencv.hpp"

#include <opencv2/opencv.hpp>
// #include <format>
#include <iostream>

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

static std::string
format_pose(xrt_pose &in)
{
	return string_format("{\"pos\": [%f, %f, %f], \"rot\": [%f, %f, %f, %f]}\n", in.position.x, in.position.y,
	                     in.position.z, in.orientation.x, in.orientation.y, in.orientation.z, in.orientation.w);
}

static void
center_cameras_bad(xrt_pose &left_in_right, xrt_pose &out_half_left_in_right)
{

	xrt_pose ident = XRT_POSE_IDENTITY;

	math_pose_interpolate(&ident, &left_in_right, 0.5, &out_half_left_in_right);
}

static void
center_cameras(xrt_pose &right_in_left, xrt_pose &out_half_left_in_right)
{

	xrt_vec3 center_pos = right_in_left.position * 0.5;

	xrt_vec3 x_axis = right_in_left.position * 1;

	math_vec3_normalize(&x_axis);

	xrt_vec3 unitz = XRT_VEC3_UNIT_Z;

	xrt_vec3 z_axis_l = unitz;

	xrt_vec3 z_axis_r;

	math_quat_rotate_vec3(&right_in_left.orientation, &unitz, &z_axis_r);

	xrt_pose between_cameras_pose;

	between_cameras_pose.position = center_pos;

	xrt_vec3 z_axis = m_vec3_orthonormalize(x_axis, z_axis_l + z_axis_r);

	math_quat_from_plus_x_z(&x_axis, &z_axis, &between_cameras_pose.orientation);

	math_pose_invert(&between_cameras_pose, &out_half_left_in_right);
}

static void
get_camera_extrinsics(struct t_stereo_camera_calibration *calib,
                      std::string &out_right_in_left,
											std::string &out_left_camera_in_center,
                      xrt_pose &out_left_in_right)
{
	// struct t_stereo_camera_calibration *calib = NULL;
	// t_stereo_camera_calibration_load("/3/INDEX_CALIBRATION.json", &calib);

	xrt::auxiliary::tracking::StereoCameraCalibrationWrapper wrap(calib);


	xrt_matrix_3x3 s;
	s.v[0] = wrap.camera_rotation_mat(0, 0);
	s.v[1] = wrap.camera_rotation_mat(1, 0);
	s.v[2] = wrap.camera_rotation_mat(2, 0);

	s.v[3] = wrap.camera_rotation_mat(0, 1);
	s.v[4] = wrap.camera_rotation_mat(1, 1);
	s.v[5] = wrap.camera_rotation_mat(2, 1);

	s.v[6] = wrap.camera_rotation_mat(0, 2);
	s.v[7] = wrap.camera_rotation_mat(1, 2);
	s.v[8] = wrap.camera_rotation_mat(2, 2);

	// xrt_pose left_in_right;
	out_left_in_right.position.x = wrap.camera_translation_mat(0);
	out_left_in_right.position.y = wrap.camera_translation_mat(1);
	out_left_in_right.position.z = wrap.camera_translation_mat(2);

	math_quat_from_matrix_3x3(&s, &out_left_in_right.orientation);
	out_left_in_right.orientation.x = -out_left_in_right.orientation.x;
	out_left_in_right.position.y = -out_left_in_right.position.y;
	out_left_in_right.position.z = -out_left_in_right.position.z;

	// out_left_in_right =



	xrt_pose right_in_left;

	math_pose_invert(&out_left_in_right, &right_in_left);

	out_right_in_left = format_pose(right_in_left);

	std::string tmp_left_in_right = format_pose(out_left_in_right);

	xrt_pose half;


	center_cameras(right_in_left, half);
	out_left_camera_in_center = format_pose(half);
	U_LOG_E("Half: %s", out_left_camera_in_center.c_str());

	U_LOG_E("Left in right: %s", tmp_left_in_right.c_str());
	U_LOG_E("Right in left: %s", out_right_in_left.c_str());
}
