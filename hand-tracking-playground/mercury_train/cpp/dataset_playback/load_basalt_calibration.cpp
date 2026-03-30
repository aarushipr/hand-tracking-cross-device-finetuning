// Inspired by:
// mercury_steamvr_driver
// t_hand_tracking_async

#include "hg_debug_instrumentation.hpp"
#include "os/os_time.h"
#include "util/u_time.h"
#include "xrt/xrt_defines.h"
#include "math/m_api.h"
#include "tracking/t_tracking.h"
#include "util/u_logging.h"
#include "sync_runner.hpp"
#include "vive/vive_config.h"
#include "util/u_file.h"
#include <fstream>
#include "assert.h"
#include "util/u_json.hpp"
#include <iostream>

using xrt::auxiliary::util::json::JSONNode;

static std::string
read_file(std::string_view path)
{
	constexpr auto read_size = std::size_t(4096);
	auto stream = std::ifstream(path.data());
	stream.exceptions(std::ios_base::badbit);

	std::string out = std::string();
	auto buf = std::string(read_size, '\0');
	while (stream.read(&buf[0], read_size)) {
		out.append(buf, 0, stream.gcount());
	}
	out.append(buf, 0, stream.gcount());
	return out;
}

void
assert_leftcam_0(JSONNode &json)
{
	JSONNode left_cam_transform = json["value0"]["T_imu_cam"][0];
	assert(left_cam_transform["px"].asDouble() == 0);
	assert(left_cam_transform["py"].asDouble() == 0);
	assert(left_cam_transform["pz"].asDouble() == 0);

	assert(left_cam_transform["qx"].asDouble() == 0);
	assert(left_cam_transform["qy"].asDouble() == 0);
	assert(left_cam_transform["qz"].asDouble() == 0);
	// assert(left_cam_transform["qw"].asDouble() == 0);
}

void
get_camera_translation(JSONNode &json, t_stereo_camera_calibration &calib)
{
	JSONNode left_to_right_j = json["value0"]["T_imu_cam"][1];

	xrt_pose left_to_right_p = {};
	left_to_right_p.position.x = left_to_right_j["px"].asDouble();
	left_to_right_p.position.y = left_to_right_j["py"].asDouble();
	left_to_right_p.position.z = left_to_right_j["pz"].asDouble();

	left_to_right_p.orientation.w = left_to_right_j["qw"].asDouble();
	left_to_right_p.orientation.x = left_to_right_j["qx"].asDouble();
	left_to_right_p.orientation.y = left_to_right_j["qy"].asDouble();
	left_to_right_p.orientation.z = left_to_right_j["qz"].asDouble();

	xrt_pose right_to_left_p;

	#if 1

	math_pose_invert(&left_to_right_p, &right_to_left_p);

	#else
	right_to_left_p = left_to_right_p;
	#endif


	xrt_matrix_3x3 bl = {};

	math_matrix_3x3_from_quat(&right_to_left_p.orientation, &bl);

	#if 0

	calib.camera_rotation[0][0] = bl.v[0];
	calib.camera_rotation[1][0] = bl.v[1];
	calib.camera_rotation[2][0] = bl.v[2];

	calib.camera_rotation[0][1] = bl.v[3];
	calib.camera_rotation[1][1] = bl.v[4];
	calib.camera_rotation[2][1] = bl.v[5];

	calib.camera_rotation[0][2] = bl.v[6];
	calib.camera_rotation[1][2] = bl.v[7];
	calib.camera_rotation[2][2] = bl.v[8];

	#else


	calib.camera_rotation[0][0] = bl.v[0];
	calib.camera_rotation[0][1] = bl.v[1];
	calib.camera_rotation[0][2] = bl.v[2];

	calib.camera_rotation[1][0] = bl.v[3];
	calib.camera_rotation[1][1] = bl.v[4];
	calib.camera_rotation[1][2] = bl.v[5];

	calib.camera_rotation[2][0] = bl.v[6];
	calib.camera_rotation[2][1] = bl.v[7];
	calib.camera_rotation[2][2] = bl.v[8];

	#endif

	calib.camera_translation[0] = right_to_left_p.position.x;
	calib.camera_translation[1] = right_to_left_p.position.y;
	calib.camera_translation[2] = right_to_left_p.position.z;
}

void
get_camera_params_kb4(JSONNode camera, t_camera_calibration &calib) {

	assert(camera["camera_type"].asString() == "kb4");

	JSONNode intrinsics = camera["intrinsics"];

	calib.intrinsics[0][0] = intrinsics["fx"].asDouble();
	calib.intrinsics[1][1] = intrinsics["fy"].asDouble();

	calib.intrinsics[0][2] = intrinsics["cx"].asDouble();
	calib.intrinsics[1][2] = intrinsics["cy"].asDouble();

	calib.kb4.k1 = intrinsics["k1"].asDouble();
	calib.kb4.k2 = intrinsics["k2"].asDouble();
	calib.kb4.k3 = intrinsics["k3"].asDouble();
	calib.kb4.k4 = intrinsics["k4"].asDouble();
}

void
load_basalt_calibration(const char *path, t_stereo_camera_calibration **out_calib)
{

	t_stereo_camera_calibration_alloc(out_calib, T_DISTORTION_FISHEYE_KB4);

	t_stereo_camera_calibration &calib_ref = **out_calib;
	JSONNode json = JSONNode::loadFromFile(path);

	JSONNode left_cam_transform = json["value0"]["T_imu_cam"][0];

	assert_leftcam_0(json);

	get_camera_translation(json, calib_ref);

	get_camera_params_kb4(json["value0"]["intrinsics"][0], calib_ref.view[0]);
	get_camera_params_kb4(json["value0"]["intrinsics"][1], calib_ref.view[1]);

	JSONNode r_arr = json["value0"]["resolution"];

	calib_ref.view[0].image_size_pixels.w = r_arr[0][0].asDouble();
	calib_ref.view[0].image_size_pixels.h = r_arr[0][1].asDouble();

	calib_ref.view[1].image_size_pixels.w = r_arr[1][0].asDouble();
	calib_ref.view[1].image_size_pixels.h = r_arr[1][1].asDouble();



	std::cout << left_cam_transform.toString() << std::endl;
}
