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
#include "csv.h"

struct rotate_command
{
	bool should_do = false;
	xrt_vec3 axis;
	float angle;
};


xrt_vec3 bads[21] = {};
#if 1

struct xrt_pose
get_left_in_right(xrt::auxiliary::tracking::StereoCameraCalibrationWrapper wrap)
{



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

	xrt_pose left_in_right;
	left_in_right.position.x = wrap.camera_translation_mat(0);
	left_in_right.position.y = wrap.camera_translation_mat(1);
	left_in_right.position.z = wrap.camera_translation_mat(2);

	math_quat_from_matrix_3x3(&s, &left_in_right.orientation);
	left_in_right.orientation.x = -left_in_right.orientation.x;
	left_in_right.position.y = -left_in_right.position.y;
	left_in_right.position.z = -left_in_right.position.z;

	return left_in_right;


	// U_LOG_E("OpenXR: %f %f %f   %f %f %f %f", left_in_right.position.x, left_in_right.position.y,
	//         left_in_right.position.z, left_in_right.orientation.x, left_in_right.orientation.y,
	//         left_in_right.orientation.z, left_in_right.orientation.w);
}

template <typename T>
int
sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}

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

cv::Mat
distort(cv::Mat img, xrt::auxiliary::tracking::CameraCalibrationWrapper wrap, rotate_command com)
{
	auto in_size = cv::Size(img.cols, img.rows);
	auto out_size = cv::Size(960, 960);


	cv::Mat map_x = cv::Mat(out_size, CV_32FC1);
	cv::Mat map_y = cv::Mat(out_size, CV_32FC1);

	std::vector<cv::Point2f> pts_ud;
	std::vector<cv::Point2f> pts_distort;

	for (int y = 0; y < out_size.height; ++y)
		for (int x = 0; x < out_size.width; ++x)
			pts_distort.emplace_back(x, y);

	std::cout << "eh " << wrap.intrinsics_mat << "\n" << wrap.distortion_fisheye_mat << "\n";

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
	cv::Mat image_distort;
	cv::remap(img, image_distort, map_x, map_y, cv::INTER_LINEAR);
	return image_distort;
}

static cv::Scalar
hsv2rgb(float fH, float fS, float fV)
{
	const float fC = fV * fS; // Chroma
	const float fHPrime = fmod(fH / 60.0, 6);
	const float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	const float fM = fV - fC;

	float fR, fG, fB;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	} else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	} else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	} else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	} else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	} else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	} else {
		fR = 0;
		fG = 0;
		fB = 0;
	}

	fR += fM;
	fG += fM;
	fB += fM;
	return {fR * 255.0f, fG * 255.0f, fB * 255.0f};
}

static void
handDot(cv::Mat &mat, xrt_vec2 place, float radius, float hue, float intensity, int type)
{
	cv::circle(mat, {(int)place.x, (int)place.y}, radius, hsv2rgb(hue * 360.0f, intensity, intensity), type);
}


static void
project(cv::Mat &frame, xrt::auxiliary::tracking::CameraCalibrationWrapper wrap, xrt_pose move_amount)
{

	std::vector<cv::Point3d> pts_relative_to_camera(21);


	for (int i = 0; i < 21; i++) {
		xrt_vec3 tmp;
		math_quat_rotate_vec3(&move_amount.orientation, &bads[i], &tmp);
		pts_relative_to_camera[i].x = tmp.x + move_amount.position.x;
		pts_relative_to_camera[i].y = tmp.y + move_amount.position.y;
		pts_relative_to_camera[i].z = tmp.z + move_amount.position.z;

		// pts_relative_to_camera[i].x = bads[i].x;
		// pts_relative_to_camera[i].y = bads[i].y;
		// pts_relative_to_camera[i].z = bads[i].z;
		pts_relative_to_camera[i].y *= -1;
		pts_relative_to_camera[i].z *= -1;
	}


	std::vector<cv::Point2d> out(21);
	cv::Affine3f aff = cv::Affine3f::Identity();
	cv::fisheye::projectPoints(pts_relative_to_camera, out, aff, wrap.intrinsics_mat, wrap.distortion_fisheye_mat);



	for (int i = 0; i < 21; i++) {
		xrt_vec2 loc;
		loc.x = out[i].x;
		loc.y = out[i].y;
		handDot(frame, loc, 2, (float)(i) / 26.0, 1, 2);
	}
}

void
aaaacsv(int frame)
{
	io::CSVReader<3> the_csv("/3/inshallah1/" + std::to_string(frame) + ".csv");
	the_csv.read_header(io::ignore_extra_column, "0", "1", "2");

	for (int i = 0; i < 21; i++) {
		xrt_vec3 &bad = bads[i];
		bool ret = the_csv.read_row(bad.x, bad.y, bad.z);
		assert(ret != false);
	}
}

#if 0
static void
draw_epipolars(cv::Mat const &frame0,
               cv::Mat const &frame1,
               xrt::auxiliary::tracking::StereoCameraCalibrationWrapper wrap)
{
	// Okay so this does the wrong thing, I don't have time to investigate it. Too bad.

	std::cout << frame0.size() << std::endl;
	cv::Matx33d R[2];

	cv::Matx34d P1;
	cv::Matx34d P2;

	cv::Matx44d Q;

	// We only want R1 and R2, we don't care about anything else
	cv::fisheye::stereoRectify(wrap.view[0].intrinsics_mat,         // cameraMatrix1
	                           wrap.view[0].distortion_fisheye_mat, // distCoeffs1
	                           wrap.view[1].intrinsics_mat,         // cameraMatrix2
	                           wrap.view[1].distortion_fisheye_mat, // distCoeffs2
	                           wrap.view[0].image_size_pixels_cv,   // imageSize*
	                           wrap.camera_rotation_mat,            // R
	                           wrap.camera_translation_mat,         // T
	                           R[0],                                // R1
	                           R[1],                                // R2
	                           P1,                                  // P1
	                           P2,                                  // P2
	                           Q,                                   // Q
	                           0,                                   // flags
	                           cv::Size()                           // newImageSize
	);


	for (int view = 0; view < 2; view++) {
		std::vector<cv::Point3d> in;
		std::vector<cv::Point2d> out;
		for (int y = -10; y < 10; y++) {
			for (int x = -10; x < 10; x++) {
				xrt_vec2 dir = {x / 5.0f, y / 5.0f};

				in.push_back({dir.x, dir.y, 1.0});
			}
		}
		// very wrong: R[view]
		// less wrong: R[view].inv()
		// even less wrong: R[1-view]
		// more wrong: R[1-view].inv
		// Yeah I am confused.
		cv::fisheye::projectPoints(in, out, R[1-view].inv(), wrap.view[view].intrinsics_mat,
		                           wrap.view[view].distortion_fisheye_mat);
		int acc_idx = 0;
		for (int y = -10; y < 10; y++) {
			for (int x = -10; x < 10; x++) {
				cv::Point2d &start = out[acc_idx++];
				cv::Point2d &next = out[acc_idx];
				if (x == 9) {
					continue;
				}
				// sighhhhhh
				cv::Mat const &frame = (view == 0) ? frame0 : frame1;
				cv::line(frame, {(int)start.x, (int)start.y}, {(int)next.x, (int)next.y},
				         cv::Scalar((-y + 10) * 10, (y + 10) * 10, 255), 1);
			}
		}
	}
}
#endif

int
main()
{
	struct t_stereo_camera_calibration *calib = NULL;
	t_stereo_camera_calibration_load("/3/INDEX_CALIBRATION.json", &calib);


	xrt_pose left_in_right = get_left_in_right(calib);

	xrt_pose identity = XRT_POSE_IDENTITY;



	cv::Mat frames[2] = {};

	int frame = 0;


	while (true) {

		aaaacsv(frame);

		// wrap.



		// cv::imshow("h", img);
		// cv::imshow("he", image_distort);

		for (int i = 0; i < 21; i++) {
			xrt_vec3 bad = bads[i];
			bads[i].x = bad.x;
			bads[i].y = bad.z;
			bads[i].z = -bad.y;
			U_LOG_E("%f %f %f", bads[i].x, bads[i].y, bads[i].z);
		}


		const char *names[] = {"l", "r"};
		for (int view = 0; view < 2; view++) {
			// cv::Mat frame = cv::imread("/3/IMAGES/IMG_" + std::to_string(view) + ".jpg");
			// project(frame, wrap);
			// cv::imshow(names[i], frame);



			std::string file_pfix = "/3/inshallah1/";
			std::string frame_string = std::to_string(frame) + "_";
			std::string view_string = std::to_string(view) + "_";

			std::string pfix = file_pfix + frame_string + view_string;

			std::string file_ext = ".jpg";

			std::cout << pfix + "color_forward" + file_ext;

			U_LOG_E("aaaaaAAAAAAAAAAAAAA");

			cv::Mat img_f = cv::imread(pfix + "color_forward" + file_ext);
			cv::Mat img_l = cv::imread(pfix + "color_left" + file_ext);
			cv::Mat img_r = cv::imread(pfix + "color_right" + file_ext);
			cv::Mat img_t = cv::imread(pfix + "color_top" + file_ext);
			cv::Mat img_b = cv::imread(pfix + "color_bottom" + file_ext);

			xrt::auxiliary::tracking::CameraCalibrationWrapper wrap(calib->view[view]);


			frames[view] = distort(img_f, wrap, {false, {}, {}});
			frames[view] += distort(img_l, wrap, {true, {0, 1, 0}, M_PI / 2});
			frames[view] += distort(img_r, wrap, {true, {0, 1, 0}, -M_PI / 2});
			frames[view] += distort(img_t, wrap, {true, {1, 0, 0}, -M_PI / 2});
			frames[view] += distort(img_b, wrap, {true, {1, 0, 0}, M_PI / 2});

			xrt_pose the_pose;

			if (view == 0) {
				the_pose = identity;
			} else {
				the_pose = left_in_right;
			}

			project(frames[view], wrap, the_pose);
		}
		xrt::auxiliary::tracking::StereoCameraCalibrationWrapper wrap(calib);

		// for (int view = 0; view < 2; view++) {
		// 	cv::imshow(names[view], frames[view]);
		// }



		cv::waitKey(1);

		cv::Mat out_img;
		cv::hconcat(frames, 2, out_img);
		cv::imshow("h", out_img);

		cv::imwrite("/3/inshallah2/" + std::to_string(frame) + ".jpg", out_img);


		frame++;
	}


	return 0;
}

#else

int
main()
{
	{
		cv::Point2f front_left = {1, 0.6};

		rotate_left(front_left);

		std::cout << front_left << std::endl;
	}
	{
		cv::Point2f front_left = {1000000, 0.1};

		rotate_left(front_left);

		std::cout << front_left << std::endl;
	}
}

#endif