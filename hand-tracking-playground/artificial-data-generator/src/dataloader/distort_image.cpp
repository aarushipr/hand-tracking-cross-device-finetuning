#pragma once
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>

#include <string>
#include <unistd.h>
#include "math/m_vec3.h"
#include "math/m_vec2.h"

#include "util/u_time.h"

#include "xrt/xrt_defines.h"
#include "math/m_space.h"
#include <filesystem>
#include <fstream>
#include "os/os_time.h"
#include "util/u_logging.h"
#include "tracking/t_tracking.h"

#include "tracking/t_calibration_opencv.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

// #include <Eigen/Core>
// #include <Eigen/Geometry>
#define EIGEN_STACK_ALLOCATION_LIMIT 1000000000

#include "math/m_eigen_interop.hpp"
#include "dataloader_common.hpp"

// #include "pinhole_camera.hpp"


constexpr int in_size = 768;

constexpr int wsize = 128;



struct global_state_magic
{
	int mouse_x;
	int mouse_y;
	cv::Mat input_image;
	cv::Mat distorted_image;
	Eigen::Matrix<uint8_t, in_size, in_size> input_image_eigen;
	Eigen::Matrix<uint8_t, wsize, wsize> distorted_image_eigen;
	cam_info dist;

	float stereographic_radius = 1.0;
	Eigen::Quaternionf rot_quat;
	// basalt::PinholeCamera<float> phc;
};


// float bound = 1;

void
mouse_callback(int event, int x, int y, int flag, void *param)
{
	global_state_magic *mg = (global_state_magic *)param;
	if (event == cv::EVENT_MOUSEMOVE) {
		mg->mouse_x = x;
		mg->mouse_y = y;
	}
}

Eigen::Vector3f
stereographic_to_direction(float sg_x, float sg_y)
{
	float X = sg_x;
	float Y = sg_y;

	float denom = (1 + X * X + Y * Y);

	float x = (2 * X) / denom;
	float y = (2 * Y) / denom;
	float z = (-1 + X * X + Y * Y) / denom;

	// forward is -z
	return {x, y, z};
	// return {x / -z, y / -z};
}

Eigen::Vector3f
stereographic_to_direction_no_normalized(float sg_x, float sg_y)
{
	float X = sg_x;
	float Y = sg_y;


	float x = (2 * X);
	float y = (2 * Y);
	float z = (-1 + X * X + Y * Y);

	return {x, y, z};
}


template <typename T>
T
map_ranges(T value, T from_low, T from_high, T to_low, T to_high)
{
	return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
}

template <typename M, typename T>
M
map_ranges_arr(M value, T from_low, T from_high, T to_low, T to_high)
{
	return (value.array() - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
}

// template <typename T>
// inline void
// UnitQuaternionRotatePoint(const Quat<T> &q, const Vec3<T> &pt, Vec3<T> &result)
// {
// 	// clang-format off
//   T uv0 = q.y * pt.z - q.z * pt.y;
//   T uv1 = q.z * pt.x - q.x * pt.z;
//   T uv2 = q.x * pt.y - q.y * pt.x;
//   uv0 += uv0;
//   uv1 += uv1;
//   uv2 += uv2;
//   result.x = pt.x + q.w * uv0;
//   result.y = pt.y + q.w * uv1;
//   result.z = pt.z + q.w * uv2;
//   result.x += q.y * uv2 - q.z * uv1;
//   result.y += q.z * uv0 - q.x * uv2;
//   result.z += q.x * uv1 - q.y * uv0;
// 	// clang-format on
// }

void
naive_remap(Eigen::Matrix<int16_t, wsize, wsize> &image_x,
            Eigen::Matrix<int16_t, wsize, wsize> &image_y,
            Eigen::Matrix<uint8_t, in_size, in_size> &input,
            Eigen::Matrix<uint8_t, wsize, wsize> &output)
{
	output.array() = 0;

	for (int y = 0; y < wsize; y++) {
		for (int x = 0; x < wsize; x++) {
			if (image_y(y, x) < 0) {
				continue;
			}
			if (image_y(y, x) >= in_size) {
				continue;
			}
			if (image_x(y, x) < 0) {
				continue;
			}
			if (image_x(y, x) >= in_size) {
				continue;
			}
			output(x, y) = input(image_x(y, x), image_y(y, x));
		}
	}
}

void
distort_0(global_state_magic &mi)
{


	Eigen::Matrix<float, wsize, wsize> sg_x;
	Eigen::Matrix<float, wsize, wsize> sg_y;

#if 0
	// Please vectorize me?
	for (int x = 0; x < wsize; ++x) {
		sg_x.col(x).setConstant(x);
	}
	// Ditto?
	for (int y = 0; y < wsize; ++y) {
		sg_y.row(y).setConstant(y);
	}

	sg_x.array() = map_ranges_arr(sg_x, 0.0f, (float)wsize, (float)-mi.stereographic_radius, (float)mi.stereographic_radius);
	sg_y.array() = map_ranges_arr(sg_y, 0.0f, (float)wsize, (float)mi.stereographic_radius, (float)-mi.stereographic_radius);
#else
	// Please vectorize me?
	for (int x = 0; x < wsize; ++x) {
		sg_x.col(x).setConstant(map_ranges<float>((float)x, 0.0f, (float)wsize, (float)-mi.stereographic_radius,
		                                          (float)mi.stereographic_radius));
	}
	// Ditto?
	for (int y = 0; y < wsize; ++y) {
		sg_y.row(y).setConstant(map_ranges<float>((float)y, 0.0f, (float)wsize, (float)mi.stereographic_radius,
		                                          (float)-mi.stereographic_radius));
	}

#endif


	Eigen::Matrix<float, wsize, wsize> dir_x;
	Eigen::Matrix<float, wsize, wsize> dir_y;
	Eigen::Matrix<float, wsize, wsize> dir_z;


#if 0
	dir_x.array() = sg_x.array() * 2;
	dir_y.array() = sg_y.array() * 2;
#else
	dir_x.array() = sg_x.array() + sg_x.array();
	dir_y.array() = sg_y.array() + sg_y.array();
#endif

	dir_z.array() = (sg_x.array() * sg_x.array()) + (sg_y.array() * sg_y.array()) - 1;


	// QUATERNION ROTATING VECTOR
	Eigen::Matrix<float, wsize, wsize> rot_dir_x;
	Eigen::Matrix<float, wsize, wsize> rot_dir_y;
	Eigen::Matrix<float, wsize, wsize> rot_dir_z;

	Eigen::Matrix<float, wsize, wsize> uv0;
	Eigen::Matrix<float, wsize, wsize> uv1;
	Eigen::Matrix<float, wsize, wsize> uv2;

	Eigen::Quaternionf q = mi.rot_quat;

	uv0.array() = q.y() * dir_z.array() - q.z() * dir_y.array();
	uv1.array() = q.z() * dir_x.array() - q.x() * dir_z.array();
	uv2.array() = q.x() * dir_y.array() - q.y() * dir_x.array();

#if 0
	uv0.array() *= 2;
	uv1.array() *= 2;
	uv2.array() *= 2;
#else

	uv0.array() += uv0.array();
	uv1.array() += uv1.array();
	uv2.array() += uv2.array();
#endif

	rot_dir_x.array() = dir_x.array() + q.w() * uv0.array();
	rot_dir_y.array() = dir_y.array() + q.w() * uv1.array();
	rot_dir_z.array() = dir_z.array() + q.w() * uv2.array();

	rot_dir_x.array() += q.y() * uv2.array() - q.z() * uv1.array();
	rot_dir_y.array() += q.z() * uv0.array() - q.x() * uv2.array();
	rot_dir_z.array() += q.x() * uv1.array() - q.y() * uv0.array();
	// END QUATERNION ROTATING VECTOR



	Eigen::Matrix<int16_t, wsize, wsize> image_x;
	Eigen::Matrix<int16_t, wsize, wsize> image_y;

#if 0

	// DIRECTION VECTOR TO IMAGE COORDINATES
	Eigen::Matrix<float, wsize, wsize> tanangle_x;
	Eigen::Matrix<float, wsize, wsize> tanangle_y;

	tanangle_x.array() = rot_dir_x.array() / -rot_dir_z.array();
	tanangle_y.array() = rot_dir_y.array() / -rot_dir_z.array();



	float bleb = 1.0;
	image_x = map_ranges_arr(tanangle_x, -bleb, bleb, 0.0f, (float)in_size).cast<int16_t>();

	// Top corresponds to +Y, bottom corresponds to -Y
	image_y = map_ranges_arr(tanangle_y, -bleb, bleb, (float)in_size, 0.0f).cast<int16_t>();
	// END DIRECTION VECTOR TO IMAGE COORDINATES
#elif 1
	// DIRECTION VECTOR TO IMAGE COORDINATES

	image_x.array() = ((mi.dist.fx * rot_dir_x.array() / -rot_dir_z.array()) + mi.dist.cx).cast<int16_t>();
	image_y.array() = ((mi.dist.fy * rot_dir_y.array() / rot_dir_z.array()) + mi.dist.cy).cast<int16_t>();


	// END DIRECTION VECTOR TO IMAGE COORDINATES
#else

	for (int x = 0; x < wsize; x++) {
		for (int y = 0; y < wsize; y++) {
			Eigen::Vector3f dir = {rot_dir_x(x, y), -rot_dir_y(x, y), -rot_dir_z(x, y)};
			Eigen::Vector2f img;
			mi.phc.project(dir, img);
			image_x(x, y) = img.x();
			image_y(x, y) = img.y();
		}
	}

#endif



	// uint64_t time_start = os_monotonic_get_ns();
	naive_remap(image_x, image_y, mi.input_image_eigen, mi.distorted_image_eigen);

	// uint64_t time_end = os_monotonic_get_ns();

	// double diff = time_end - time_start;
	// U_LOG_E("remap: %f ms", diff / U_TIME_1MS_IN_NS);
}



cv::Point
slow(global_state_magic &mi, float x, float y)
{
	float sg_x = map_ranges<float>(x, 0, wsize, -mi.stereographic_radius, mi.stereographic_radius);

	// bottom of our image has higher
	float sg_y = map_ranges<float>(y, 0, wsize, mi.stereographic_radius, -mi.stereographic_radius);
	// We are in OpenXR coordinates!
	// float sg_x = map_ranges<float>(x, 0, wsize, -mi.stereographic_radius, mi.stereographic_radius);

	// // bottom of our image has higher
	// float sg_y = map_ranges<float>(y, 0, wsize, mi.stereographic_radius, -mi.stereographic_radius);

	Eigen::Vector3f dir = stereographic_to_direction_no_normalized(sg_x, sg_y);

	dir = mi.rot_quat * dir;

	float x_in_image = (mi.dist.fx * dir.x() / -dir.z()) + mi.dist.cx;

	// Note the lack of negation here. This is because of a double negative: negative Z coordinate,
	// and image-space coordinates have Y increasing as you go down.
	float y_in_image = (mi.dist.fy * dir.y() / dir.z()) + mi.dist.cy;


	return cv::Point2i{(int)x_in_image, (int)y_in_image};
}

void
draw_boundary(global_state_magic &mi, cv::Scalar color, cv::Mat img)
{
	std::vector<cv::Point> line_vec = {};
	for (int y = 0; y < wsize; y++) {
		int x = 0;
		cv::Point e = slow(mi, x, y);
		line_vec.push_back(e);
	}
	cv::polylines(img, line_vec, false, color);

	line_vec.clear();
	for (int y = 0; y < wsize; y++) {
		int x = wsize;
		cv::Point e = slow(mi, x, y);
		line_vec.push_back(e);
	}
	cv::polylines(img, line_vec, false, color);

	line_vec.clear();
	for (int x = 0; x < wsize; x++) {
		int y = 0;
		cv::Point e = slow(mi, x, y);
		line_vec.push_back(e);
	}
	cv::polylines(img, line_vec, false, color);

	line_vec.clear();
	for (int x = 0; x < wsize; x++) {
		int y = wsize;
		cv::Point e = slow(mi, x, y);
		line_vec.push_back(e);
	}
	cv::polylines(img, line_vec, false, color);
}



void
project_25_points_unscaled(hand26 &joints_local, Eigen::Quaternionf rot_quat, hand25_2d &out_joints)
{
	// Eigen::Vector3f dir = xrt::auxiliary::math::map_vec3(direction_3d);

	// // xrt_quat bleh = {};

	// Eigen::Quaternionf rot_quat = Eigen::Quaternionf().setFromTwoVectors(-Eigen::Vector3f::UnitZ(), dir);

	for (int i = 0; i < 25; i++) {
		xrt_vec3 d = m_vec3_normalize(joints_local[i].position);
		Eigen::Vector3f direction = {d.x, d.y, d.z};
		direction = rot_quat.conjugate() * direction;
		float denom = 1 - direction.z();
		float sg_x = direction.x() / denom;
		float sg_y = direction.y() / denom;
		// sg_x *= mi.stereographic_radius;
		// sg_y *= mi.stereographic_radius;

		out_joints[i].x = sg_x;
		out_joints[i].y = sg_y;
	}
}

template <typename V2>
void
project_point_scaled(global_state_magic &mi, xrt_vec3 local_pt, V2 &out_img_pt)
{
	xrt_vec3 d = m_vec3_normalize(local_pt);
	Eigen::Vector3f direction = {d.x, d.y, d.z};
	direction = mi.rot_quat.conjugate() * direction;
	float denom = 1 - direction.z();
	float sg_x = direction.x() / denom;
	float sg_y = direction.y() / denom;
	// sg_x *= mi.stereographic_radius;
	// sg_y *= mi.stereographic_radius;

	out_img_pt.x = map_ranges<float>(sg_x, -mi.stereographic_radius, mi.stereographic_radius, 0, wsize);
#if 0
			out_joints_in_img[i].y = map_ranges<float>(sg_y, -mi.stereographic_radius, mi.stereographic_radius, wsize, 0);
#else
	out_img_pt.y = map_ranges<float>(sg_y, mi.stereographic_radius, -mi.stereographic_radius, 0, wsize);
#endif
}

void
project_25_points_scaled(global_state_magic &mi, hand26 &joints_local, hand25_2d &out_joints_in_img)
{
	for (int i = 0; i < 25; i++) {
		project_point_scaled(mi, joints_local[i].position, out_joints_in_img[i]);
#if 0
		xrt_vec3 d = m_vec3_normalize(joints_local[i].position);
		Eigen::Vector3f direction = {d.x, d.y, d.z};
		direction = mi.rot_quat.conjugate() * direction;
		float denom = 1 - direction.z();
		float sg_x = direction.x() / denom;
		float sg_y = direction.y() / denom;
		// sg_x *= mi.stereographic_radius;
		// sg_y *= mi.stereographic_radius;

		out_joints_in_img[i].x =
		    map_ranges<float>(sg_x, -mi.stereographic_radius, mi.stereographic_radius, 0, wsize);
#if 0
			out_joints_in_img[i].y = map_ranges<float>(sg_y, -mi.stereographic_radius, mi.stereographic_radius, wsize, 0);
#else
		out_joints_in_img[i].y =
		    map_ranges<float>(sg_y, mi.stereographic_radius, -mi.stereographic_radius, 0, wsize);

#endif
#endif
	}
}



Eigen::Quaternionf
directon(Eigen::Vector3f dir, float twist)
{
	Eigen::Quaternionf one = Eigen::Quaternionf().setFromTwoVectors(-Eigen::Vector3f::UnitZ(), dir);

	Eigen::Quaternionf two;
	two = Eigen::AngleAxisf(twist, -Eigen::Vector3f::UnitZ());
	return one * two;
}

xrt_vec2
finite_differences(global_state_magic &mi, xrt_vec3 wrist_location, xrt_vec3 forearm_location)
{
	xrt_vec2 retval = {};

	xrt_vec3 direction_to_forearm = forearm_location - wrist_location;

	direction_to_forearm *= 0.05;

	xrt_vec3 finite_forearm_pt = wrist_location + direction_to_forearm;

	xrt_vec2 ffp_img = {};
	project_point_scaled(mi, finite_forearm_pt, ffp_img);

	xrt_vec2 wrist_img = {};
	project_point_scaled(mi, wrist_location, wrist_img);

	retval = ffp_img - wrist_img;
	m_vec2_normalize(&retval);

	return retval;
}

xrt_vec2
differential(global_state_magic &mi, xrt_vec3 wrist_location, xrt_vec3 forearm_location)
{
	xrt_vec2 retval = {};

	Eigen::Vector3f wrist = {wrist_location.x, wrist_location.y, wrist_location.z};
	wrist = mi.rot_quat.conjugate() * wrist;

	Eigen::Vector3f forearm = {forearm_location.x, forearm_location.y, forearm_location.z};
	forearm = mi.rot_quat.conjugate() * forearm;

	Eigen::Vector3f direction_to_forearm = forearm - wrist;

	direction_to_forearm.normalize();
	wrist.normalize();

	// xrt_vec2 wrist_img = {};
	// project_point_scaled(mi, wrist_location, wrist_img);

	float x = wrist.x();
	float y = wrist.y();
	float z = wrist.z();

	float sr = mi.stereographic_radius;

	float du_dx = -64 / (sr * (z - 1));
	float du_dy = 0;
	float du_dz = 64 * x / (sr * pow(z - 1, 2));

	float dv_dx = 0;
	float dv_dy = 64 / (sr * (z - 1));
	float dv_dz = -64 * y / (sr * pow(z - 1, 2));


	float du = (du_dx * direction_to_forearm.x()) + (du_dy * direction_to_forearm.y()) +
	           (du_dz * direction_to_forearm.z());
	float dv = (dv_dx * direction_to_forearm.x()) + (dv_dy * direction_to_forearm.y()) +
	           (dv_dz * direction_to_forearm.z());

	retval.x = du;
	retval.y = dv;

	m_vec2_normalize(&retval);

	return retval;
}

void
distort_image(std::vector<cv::Mat> input_images,
              xrt_vec3 direction_3d,
              cv::Scalar color,
              cam_info dist,
              float twist,
              float expand_val,
              hand26 &joints_gt,
              hand26 &joints_predicted, // These may or may not actually be predicted.
              hand25_2d &out_joints_in_img_gt,
              hand25_2d &out_joints_in_img_predicted,
              xrt_vec2 &norm_finite_diff_direction_px_coord,
              float &out_stereographic_radius,
              std::vector<cv::Mat>& out)
{
	global_state_magic mi = {};
	mi.dist = dist;

	mi.input_image =
	    cv::Mat(cv::Size(in_size, in_size), CV_8U, mi.input_image_eigen.data(), in_size * sizeof(uint8_t));
	mi.distorted_image =
	    cv::Mat(cv::Size(wsize, wsize), CV_8U, mi.distorted_image_eigen.data(), wsize * sizeof(uint8_t));


	math_vec3_normalize(&direction_3d);

	Eigen::Vector3f dir = xrt::auxiliary::math::map_vec3(direction_3d);

	// xrt_quat bleh = {};

	mi.rot_quat = directon(dir, twist);

	Eigen::Vector3f old_direction = dir;


	// Meow meow empirically tested on Dec 7: This converges in 4 iterations max, usually 2.
	for (int i = 0; i < 8; i++) {
		hand25_2d pts_sgo = {};
		project_25_points_unscaled(joints_predicted, mi.rot_quat, pts_sgo);

		xrt_vec2 min = vv(pts_sgo[0]);
		xrt_vec2 max = vv(pts_sgo[0]);

		for (int i = 0; i < 25; i++) {
			xrt_vec2 pt = vv(pts_sgo[i]);
			min.x = fmin(pt.x, min.x);
			min.y = fmin(pt.y, min.y);

			max.x = fmax(pt.x, max.x);
			max.y = fmax(pt.y, max.y);
		}


		xrt_vec2 center = m_vec2_mul_scalar(min + max, 0.5);

		float r = fmax(center.x - min.x, center.y - min.y);
		mi.stereographic_radius = r;

		Eigen::Vector3f new_direction = stereographic_to_direction(center.x, center.y);

		new_direction = mi.rot_quat * new_direction;

		mi.rot_quat = directon(new_direction, twist);


		if ((old_direction - dir).norm() < 0.0001) {
			// We converged
			break;
		}
		old_direction = dir;
	}


	float old_stereographic_radius = mi.stereographic_radius;
	mi.stereographic_radius *= expand_val;
	out_stereographic_radius = mi.stereographic_radius;


	project_25_points_scaled(mi, joints_gt, out_joints_in_img_gt);
	project_25_points_scaled(mi, joints_predicted, out_joints_in_img_predicted);

	// hand26 joints_gt_copy = {};

	// // std::array doesn't have a "copy" ??????

	// for (int i = 0; i < 26; i++) {
	// 	joints_gt_copy[i] = joints_gt[i];
	// }


	// norm_finite_diff_direction_px_coord = finite_differences(mi, joints_gt[0].position, joints_gt[25].position);
	norm_finite_diff_direction_px_coord = differential(mi, joints_gt[0].position, joints_gt[25].position);

	// U_LOG_E("finite: %f %f", norm_finite_diff_direction_px_coord.x, norm_finite_diff_direction_px_coord.y);
	// U_LOG_E("differ: %f %f", other.x, other.y);

	for (size_t i = 0; i < input_images.size(); i++) {
#if 0
	input_image.copyTo(mi.input_image);
#else
		cv::cvtColor(input_images[i], mi.input_image, cv::COLOR_BGR2GRAY);
#endif

		draw_boundary(mi, color, input_images[i]);

		distort_0(mi);

		draw_boundary(mi, color, input_images[i]);

		std::ostringstream str;
		str << mi.stereographic_radius;
		cv::putText(input_images[i], str.str().c_str(), {40, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255, 255, 0});
		// out.push_back(cv::Mat());
		mi.distorted_image.copyTo(out[i]);
		// out.push_back(mi.distorted_image);

	}
	// This is slow and I am being lazy. Pls fix
	// input_image.copyTo(input_image);
}
