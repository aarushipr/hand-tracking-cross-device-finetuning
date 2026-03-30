#pragma once
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
#include <iostream>
#include <opencv2/opencv.hpp>

// Oh no lol
struct vec2_5
{
	float x;
	float y;
	float depth_relative_to_midpxm;

	float confidence_xy;
	float confidence_depth;

	xrt_vec3 &
	ref_xrt_vec3()
	{
		return *(xrt_vec3 *)&this->x;
	}
	xrt_vec2 &
	ref_xrt_vec2()
	{
		return *(xrt_vec2 *)&this->x;
	}
};

static xrt_vec2
vv(vec2_5 v)
{
	return {v.x, v.y};
}

using hand26 = std::array<xrt_pose, 26>;
using hand25_2d = std::array<vec2_5, 25>;
namespace fs = std::filesystem;


struct cam_info
{
	float fx;
	float fy;
	float cx;
	float cy;
};



struct single_frame
{
	xrt_pose camera_pose;
	hand26 hand_landmarks;
	hand26 noise_free_pose_predicted_hand_landmarks;
	cam_info camera_info;

	float homothety_scale;
	float move_overall_variance;
	float move_per_joint_variance;
};

// struct past_frames_info
// {
// 	hand26 one_frame_ago;
// 	hand26
// }

struct sequence
{
	fs::path path = {};
	std::vector<single_frame> frames = {};
};



void
distort_image(std::vector<cv::Mat> input_images,
              xrt_vec3 direction_3d,
              cv::Scalar color,
              cam_info dist,
              float twist,
              float expand_val,
              float expand_val_palm,
              hand26 &joints_gt,
              hand26 &joints_predicted, // These may or may not actually be predicted.
              hand25_2d &out_joints_in_img_gt,
              hand25_2d &out_joints_in_img_predicted,
              xrt_vec2 &norm_finite_diff_direction_px_coord,
              float &out_stereographic_radius,
              std::vector<cv::Mat> &out);

void
hand_curls(const hand26 &gt, std::array<float, 5> &curls_out);

static float
hand_length(const hand26 &hand)
{
	return m_vec3_len(hand[0].position - hand[11].position);
}
