#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>

#include <string>
#include <unistd.h>
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
#include "distort_image.hpp"

struct rotate_command
{
	bool should_do = false;
	xrt_vec3 axis;
	float angle;
};

struct distort_one_view
{
	cv::Mat map_x;
	cv::Mat map_y;
};

struct distort_views
{
	// Left, right. Then, left right top bottom.
	struct distort_one_view views[2][5];
};

struct image_distort_data
{
	struct t_stereo_camera_calibration *calib;
	distort_views views_distort;
};



void
rotate(cv::Point2f &in, struct rotate_command com)
{

	struct xrt_vec3 vec = {in.x, in.y, 1};

	vec = m_vec3_normalize(vec);



	xrt_vec3 rotate_on_z = com.axis;
	xrt_quat bleh;
	math_quat_from_angle_vector(com.angle, &rotate_on_z, &bleh);

	math_quat_rotate_vec3(&bleh, &vec, &vec);

	if (vec.z < 0) {
		in.x = NAN;
		in.y = NAN;
		return;
	}

	in.x = vec.x / vec.z;
	in.y = vec.y / vec.z;
}

void
init_distort_map(cv::Size in_size,
                 cv::Size out_size,
                 xrt::auxiliary::tracking::CameraCalibrationWrapper wrap,
                 rotate_command com,
                 distort_one_view &view)

//  cv::Mat &map_x,
//  cv::Mat &map_y)
{
	cv::Mat &map_x = view.map_x;
	cv::Mat &map_y = view.map_y;
	// auto in_size = cv::Size(img.cols, img.rows);


	map_x = cv::Mat(out_size, CV_32FC1);
	map_y = cv::Mat(out_size, CV_32FC1);

	std::vector<cv::Point2f> pts_ud;
	std::vector<cv::Point2f> pts_distort;

	for (int y = 0; y < out_size.height; ++y)
		for (int x = 0; x < out_size.width; ++x)
			pts_distort.emplace_back(x, y);

	// std::cout << "eh " << wrap.intrinsics_mat << "\n" << wrap.distortion_fisheye_mat << "\n";

	cv::fisheye::undistortPoints(pts_distort, pts_ud, wrap.intrinsics_mat, wrap.distortion_fisheye_mat);

	for (auto &pt : pts_ud) {
		if (com.should_do) {
			rotate(pt, com);
		}
		float iwidth_2 = float(in_size.width) / 2;
		float owidth_2 = float(out_size.width) / 2;
		float iheight_2 = float(in_size.height) / 2;
		float oheight_2 = float(out_size.height) / 2;

		pt.x *= iwidth_2;
		pt.x += iwidth_2 - .5;

		pt.y *= iwidth_2;
		pt.y += iheight_2 - .5;
	}

	for (int y = 0; y < out_size.height; ++y) {
		float *ptr1 = map_x.ptr<float>(y);
		float *ptr2 = map_y.ptr<float>(y);
		for (int x = 0; x < out_size.width; ++x) {
			const auto &pt = pts_ud[y * out_size.width + x];
			ptr1[x] = pt.x;
			ptr2[x] = pt.y;
		}
	}
	// std::cout << map_x.size << std::endl;
	// std::cout << view.map_x.size << std::endl;
	// cv::Mat image_distort;
	// cv::remap(img, image_distort, map_x, map_y, cv::INTER_LINEAR);
	// return image_distort;
}



void
init_distort_image(struct image_distort_data **out_data, t_stereo_camera_calibration *calib)
{
	image_distort_data *data = new image_distort_data();
	t_stereo_camera_calibration_reference(&data->calib, calib);
	// data->left_in_right = get_left_in_right(calib);
	// data->identity = XRT_POSE_IDENTITY;
	for (int view = 0; view < 2; view++) {
		xrt::auxiliary::tracking::CameraCalibrationWrapper wrap(calib->view[view]);
		cv::Size in_size = {512, 512};
		cv::Size out_size = {calib->view[0].image_size_pixels.w, calib->view[0].image_size_pixels.h};

		init_distort_map(in_size, out_size, wrap, {false, {}, {}},              // Front
		                 data->views_distort.views[view][0]);                   //
		                                                                        //
		init_distort_map(in_size, out_size, wrap, {true, {0, 1, 0}, M_PI / 2},  // Left
		                 data->views_distort.views[view][1]);                   //
		                                                                        //
		init_distort_map(in_size, out_size, wrap, {true, {0, 1, 0}, -M_PI / 2}, // Right
		                 data->views_distort.views[view][2]);                   //
		                                                                        //
		init_distort_map(in_size, out_size, wrap, {true, {1, 0, 0}, -M_PI / 2}, // Top
		                 data->views_distort.views[view][3]);                   //
		                                                                        //
		init_distort_map(in_size, out_size, wrap, {true, {1, 0, 0}, M_PI / 2},  // Bottom
		                 data->views_distort.views[view][4]);                   //
	}
	*out_data = data;
}

cv::Mat
remapw(cv::Mat &src, distort_one_view view)
{
	cv::Mat dst;
	cv::remap(src, dst, view.map_x, view.map_y, cv::INTER_LINEAR);
	return dst;
}

cv::Mat
distort_image(struct image_distort_data *data, cubemap_view &in, int camera_idx)
{
	cv::Mat the_out;

	// the_out = cv::Scalar(0,0,0);

	the_out = remapw(in.views[cubemap_view_idx::FORWARD], data->views_distort.views[camera_idx][0]);
	the_out += remapw(in.views[cubemap_view_idx::LEFT], data->views_distort.views[camera_idx][1]);
	the_out += remapw(in.views[cubemap_view_idx::RIGHT], data->views_distort.views[camera_idx][2]);
	the_out += remapw(in.views[cubemap_view_idx::TOP], data->views_distort.views[camera_idx][3]);
	the_out += remapw(in.views[cubemap_view_idx::BOTTOM], data->views_distort.views[camera_idx][4]);

	return the_out;
}
