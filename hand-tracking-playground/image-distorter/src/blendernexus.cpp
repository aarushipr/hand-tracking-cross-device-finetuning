#include "blendernexus.hpp"

#include <stdio.h>
#include <unistd.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "math/m_vec3.h"
#include "math/m_vec2.h"

#include "util/u_time.h"

#include "xrt/xrt_defines.h"
#include "math/m_space.h"

#include "os/os_time.h"
#include "util/u_logging.h"
#include "tracking/t_tracking.h"

#include "tracking/t_calibration_opencv.hpp"

#include <opencv2/opencv.hpp>
#include "csv.hpp"
#include "subprocess.hpp"

#include "camera_calibration_rightinleft.hpp"
#include "distort_image.hpp"
#include "u_uniform_distribution.h"


namespace fs = std::experimental::filesystem;


fs::path poses_csvname_openxr = "poses_openxr.csv";
fs::path poses_csvname_opencv = "poses_opencv.csv";


fs::path color_cubemap_name = "cubemaps_color";
fs::path alpha_cubemap_name = "cubemaps_alpha";

fs::path final_images_name = "imgs";

struct joints_21kp
{
	struct xrt_vec3 kps[21];
};

struct captured_state
{
	fs::path root_dir; // = "/3/inshallah4";


	fs::path csvpath_openxr;
	fs::path csvpath_opencv;


	int num_frames = 80;

	int wristpose_num_frames;
	int fingerpose_num_frames;

	// float wristpose_framerate = 120.0f;

	// // Sorta wrong - our euroc recorder was dropping some frames for some reason
	// // so it'd be some amount less
	// float fingerpose_framerate = 54.0f;
	// float out_framerate = 30.0f;

	image_distort_data *distdata = NULL;

	uniform_distribution *uniform_distribution_data = NULL;


	std::vector<joints_21kp> joints_all_frames = {};


	// CSV file we are writing.
	// Output spot: Root/seq{idx}/valid_samples.csv
	// Columns: Frame index, camera index. Nominally there'd be num_frames*2 rows and this would be obviated,
	// but due to the way we generate data, some of the hands will be outside the image so we need it.
	std::ofstream out_valid_samples;
	int num_valid_samples;

	// +Z forward, +X right, +Y down.
	std::ofstream out_poses_opencv;
};



void
wait_for_file_to_exist(fs::path &path)
{
	std::error_code ec;
	while (true) {
		int exists = fs::exists(path, ec);
		os_nanosleep(U_TIME_1MS_IN_NS * 50);
		// std::cout << "aaaa " << path << " " << exists << " " << ec.message() << std::endl;
		if (exists) {
			return;
		}
	}
}

void
wait_for_files_to_exist(const std::vector<fs::path> paths)
{
	bool ok = true;
	while (true) {
		for (const fs::path &path : paths) {
			ok = ok && fs::exists(path);
		}
		if (ok) {
			return;
		}
	}
}



std::string
leftpadfour(int i)
{
	std::string str = string_format("%04d", i);
	return str;
}

size_t
get_csv_length(fs::path filename)
{
	csv::CSVReader reader(filename.string());

	size_t n_rows = 0;

	for (csv::CSVRow &row : reader) {
		n_rows++;
	}

	return n_rows;
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

// Note, this is for some reason in OpenXR coordinate system. See the conversion when we load the CSV file.
static bool
project(cv::Mat &frame, xrt::auxiliary::tracking::CameraCalibrationWrapper wrap, xrt_pose move_amount, xrt_vec3 *joints)
{

	std::vector<cv::Point3d> pts_relative_to_camera(21);


	bool this_is_behind_camera[21] = {};
	for (int i = 0; i < 21; i++) {
		xrt_vec3 tmp;
		math_quat_rotate_vec3(&move_amount.orientation, &joints[i], &tmp);
		pts_relative_to_camera[i].x = tmp.x + move_amount.position.x;
		pts_relative_to_camera[i].y = tmp.y + move_amount.position.y;
		pts_relative_to_camera[i].z = tmp.z + move_amount.position.z;

		// pts_relative_to_camera[i].x = bads[i].x;
		// pts_relative_to_camera[i].y = bads[i].y;
		// pts_relative_to_camera[i].z = bads[i].z;
		pts_relative_to_camera[i].y *= -1;
		pts_relative_to_camera[i].z *= -1;
		// U_LOG_E("%f %f %f", pts_relative_to_camera[i].x, pts_relative_to_camera[i].y,
		//         pts_relative_to_camera[i].z);
		if (pts_relative_to_camera[i].z < 0) {
			this_is_behind_camera[i] = true;
		}
	}


	std::vector<cv::Point2d> out(21);
	cv::Affine3f aff = cv::Affine3f::Identity();
	cv::fisheye::projectPoints(pts_relative_to_camera, out, aff, wrap.intrinsics_mat, wrap.distortion_fisheye_mat);

	int num_outside = 0;

	for (int i = 0; i < 21; i++) {
		xrt_vec2 loc;
		loc.x = out[i].x;
		loc.y = out[i].y;
		xrt_vec2 center_px = {wrap.image_size_pixels.w / 2.0f, wrap.image_size_pixels.h / 2.0f};
		xrt_vec2 diff = center_px - loc;
		float dist_px = m_vec2_len(diff);

#if 0
		// Hardcoded for Index
		const float max_radius = 480.0f;
#else
		// Should work _okay_ but be kinda weird on non-square aspect ratios. Doesn't create false positives
		// tho.
		float max_radius = (wrap.image_size_pixels.w + wrap.image_size_pixels.h) / 4;
#endif

		bool outside_view = (dist_px > max_radius) || this_is_behind_camera[i];
		handDot(frame, loc, 2, outside_view ? 1.0 : (float)(i) / 26.0, 1, 2);
		if (outside_view) {
			num_outside++;
		}
	}
	return num_outside < 6;
}


void
add_vignettes(fs::path color, fs::path alpha, bool use_alpha)
{
	std::vector<std::string> args = {"python3",
	                                 "/3/epics/artificial_data_3/image-distorter/py/add_index_vignette.py"};


	std::cout << color << " " << alpha << std::endl;
	subprocess::environment env = //
	    subprocess::environment{{
	        {"IMG_PATH_COLOR", color.string()}, //
	        {"IMG_PATH_ALPHA", alpha.string()}, //
	        {"USE_ALPHA", alpha.string()}       //
	    }};


	// Todo what actually happens here? I wrote it weirdly so that the template deducer would be
	// happy, but I'm hoping we're just passing the original thing by ...value? in both cases1

	subprocess::Popen p = subprocess::Popen( //
	    std::vector<std::string>(args),      //
	    subprocess::environment(env)         //
	);
	p.wait();
}


// Check the artificial_data_2 version of this to see how I made a euroc dataset ;)
// Only used in validation dataset codepath
void
make_img_with_background(fs::path color, fs::path alpha, fs::path out)
{
	std::vector<std::string> args = {"python3", "/3/epics/artificial_data_3/image-distorter/py/uhhhh.py"};


	std::cout << color << " " << alpha << std::endl;
	subprocess::environment env = //
	    subprocess::environment{{
	        {"IMG_PATH_COLOR", color.string()}, //
	        {"IMG_PATH_ALPHA", alpha.string()}, //
	        {"IMG_PATH_OUT", out.string()},     //
	    }};


	// Todo what actually happens here? I wrote it weirdly so that the template deducer would be
	// happy, but I'm hoping we're just passing the original thing by ...value? in both cases1

	subprocess::Popen p = subprocess::Popen( //
	    std::vector<std::string>(args),      //
	    subprocess::environment(env)         //
	);
	p.wait();
}

void
do_py_validation_img(fs::path color, fs::path alpha)
{
	make_img_with_background(color, alpha, color);

	fs::remove(alpha);
}

template <typename T>
void
read_assign(csv::CSVRow &row, int idx, T *out)
{
	csv::CSVField field = row[idx];
	*out = field.get<T>();
}

void
uh_loop_thing(create_dataset_info &info,
              struct captured_state &st,
              int frame_idx,
              int camera_idx,
              std::string color_or_alpha_suffix,
              cv::Mat &out_mat,
              fs::path &out_finalname)
{
	std::vector<std::string> directions = {"forward", "left", "right", "top", "bottom"};
	cubemap_view cubemap = {};
	for (int cubemap_direction_idx = 0; cubemap_direction_idx < directions.size(); cubemap_direction_idx++) {
		std::string direction = directions[cubemap_direction_idx];
		// char idx[5];
		// snprintf(idx, 5, "%04d", frame_idx);
		// std::cout << idx << std::endl;

		std::string filename =
		    "Image" + leftpadfour(frame_idx) + "_" + std::to_string(camera_idx) + "_" + direction + ".jpg";
		// std::cout << filename << std::endl;
		// std::string filename =
		fs::path path = info.out_dir / (std::string("cubemaps_") + color_or_alpha_suffix) / filename;

		// std::cout << path << std::endl;

		std::cout << "Waiting for " << path << " to exist" << std::endl;

		wait_for_file_to_exist(path);

		cubemap.views[cubemap_direction_idx] = cv::imread(path);
		if (cubemap.views[cubemap_direction_idx].empty()) {
			std::cout << "Uh oh, couldn't load " << path << std::endl;
			abort();
		}
	}
	out_mat = distort_image(st.distdata, cubemap, camera_idx);

	std::string outfilename = "frame" + leftpadfour(frame_idx) + "_camera" + std::to_string(camera_idx) + "_" +
	                          color_or_alpha_suffix + ".jpg";

	fs::path finalname = info.out_dir / final_images_name / outfilename;

	out_finalname = finalname;

	cv::imwrite(finalname, out_mat);
}

// once upon a time, /3/whatever/three.csv and /3/whatever/left_smoothed.csv
int
create_dataset(create_dataset_info info)
{

	struct captured_state st = {};

	st.csvpath_opencv = info.out_dir / poses_csvname_opencv;
	st.csvpath_openxr = info.out_dir / poses_csvname_openxr;

	u_random_distribution_create(&st.uniform_distribution_data);

	st.wristpose_num_frames = get_csv_length(info.wristpose);
	st.fingerpose_num_frames = get_csv_length(info.fingerpose);
	std::cout << "fingerpose " << st.fingerpose_num_frames << " wristpose " << st.wristpose_num_frames << std::endl;

	int unusuable_end_frames_wrist = ceil(info.num_frames * (info.wristpose_framerate / info.out_framerate));
	int unusuable_end_frames_finger = ceil(info.num_frames * (info.fingerpose_framerate / info.out_framerate));

	std::cout << unusuable_end_frames_wrist << " " << unusuable_end_frames_finger << std::endl;

	int latest_start_time_wrist = st.wristpose_num_frames - unusuable_end_frames_wrist;
	int latest_start_time_finger = st.fingerpose_num_frames - unusuable_end_frames_finger;



	int wristpose_start_idx =
	    u_random_distribution_get_sample_int(st.uniform_distribution_data, 0, latest_start_time_wrist);
	int fingerpose_start_idx =
	    u_random_distribution_get_sample_int(st.uniform_distribution_data, 0, latest_start_time_finger);

	std::cout << wristpose_start_idx << " " << fingerpose_start_idx << std::endl;


	std::string calibration_names[] = {"moses_index.json", "jakob_elp.json"};

	struct t_stereo_camera_calibration *calib = NULL;

#if 0
	std::string calibration_name = "/3/whatever2/camera_calibrations/" + calibration_names[1];
#else
	std::string calibration_name =
	    "/3/whatever2/camera_calibrations/" +
	    calibration_names[u_random_distribution_get_sample_int(st.uniform_distribution_data, 0, 2)];

#endif

	t_stereo_camera_calibration_load(calibration_name.c_str(), &calib);

	xrt::auxiliary::tracking::StereoCameraCalibrationWrapper wrap(calib);

	std::string calibration_string_right_in_left;
	std::string calibration_string_left_in_center;
	xrt_pose calibration_left_in_right;

	get_camera_extrinsics(calib, calibration_string_right_in_left, calibration_string_left_in_center, calibration_left_in_right);
	//  = get_right_in_left(calib);

	std::cout << calibration_string_right_in_left << std::endl;

	bool bleh = fs::remove_all(info.out_dir);

	bool blorg = fs::create_directories(info.out_dir);

	U_LOG_E("Creating dataset at %s out of %s and %s", info.out_dir.c_str(), info.fingerpose.c_str(),
	        info.wristpose.c_str());

	std::vector<std::string> args = {"blender", //
	                                 "-P",      //
	                                 "/3/epics/artificial_data_3/image-distorter/py/magic2.py"};



	subprocess::environment env =
	    subprocess::environment{{{"WRISTPOSE_CSV", info.wristpose.string()},
	                             {"WRISTPOSE_START_IDX", std::to_string(wristpose_start_idx)},

	                             {"FINGERPOSE_CSV", info.fingerpose.string()},
	                             {"FINGERPOSE_START_IDX", std::to_string(fingerpose_start_idx)},

	                             {"USE_EXR_BACKGROUND", std::to_string(int(info.use_exr_background))},
	                             {"RENDER_ALPHA", std::to_string(int(info.render_alpha))},

	                             {"NUM_FRAMES", std::to_string(info.num_frames)},
	                             {"CAMERA_RIGHT_IN_LEFT", calibration_string_right_in_left},
	                             {"CAMERA_LEFT_IN_CENTER", calibration_string_left_in_center},
	                             {"OUTPUT_CSV_OPENXR", st.csvpath_openxr},
	                             {"OUTPUT_CSV_OPENCV", st.csvpath_opencv},
	                             {"OUTPUT_COLOR_BASE", info.out_dir / color_cubemap_name},
	                             {"OUTPUT_ALPHA_BASE", info.out_dir / alpha_cubemap_name},
	                             {"HAND_MODEL_INDEX", std::to_string(int(info.hand_model_index))},
	                             {"DONT_EXIT_IMMEDIATELY", std::to_string(int(info.dont_exit_blender_at_end))},
	                             {"DONT_RENDER", std::to_string(int(info.dont_render))}

	    }};


	// Todo what actually happens here? I wrote it weirdly so that the template deducer would be
	// happy, but I'm hoping we're just passing the original thing by ...value? in both cases1

	subprocess::Popen p = subprocess::Popen( //
	    std::vector<std::string>(args),      //
	    subprocess::environment(env)         //
	);

	init_distort_image(&st.distdata, calib);

	wait_for_file_to_exist(st.csvpath_openxr);
	wait_for_file_to_exist(st.csvpath_opencv);

	csv::CSVReader reader(st.csvpath_openxr.string());

	st.joints_all_frames.clear();

	for (csv::CSVRow &row : reader) { // Input iterator
		joints_21kp joints;
		for (int joint_idx = 0; joint_idx < 21; joint_idx++) {
			xrt_vec3 tmp;
			read_assign<float>(row, (joint_idx * 3) + 0, &tmp.x);
			read_assign<float>(row, (joint_idx * 3) + 1, &tmp.y);
			read_assign<float>(row, (joint_idx * 3) + 2, &tmp.z);

			// Blender to OpenXR conversion!
			xrt_vec3 *out = &joints.kps[joint_idx];

			out->x = tmp.x;
			out->y = tmp.y;
			out->z = tmp.z;
		}
		st.joints_all_frames.push_back(joints);
		// std::cout << std::endl;
	}

	fs::create_directories(info.out_dir / final_images_name);

	st.out_valid_samples.open(info.out_dir / "valid_samples.csv");
	st.out_valid_samples << "frame_idx, camera_idx" << std::endl;

	st.num_valid_samples = 0;


	for (int frame_idx = 0; frame_idx < info.num_frames; frame_idx++) {
		cv::Mat mats[2];
		std::cout << "Waiting for frame " << std::to_string(frame_idx) << " in " << info.out_dir << " to exist!"
		          << std::endl;
		std::vector<fs::path> paths(10);
		// struct two_cubemap_views two_cubemaps;
		// size_t acc_idx = 0;
		fs::path color_img_path;
		fs::path alpha_img_path;
		for (int camera_idx = 0; camera_idx < 2; camera_idx++) {
			// std::string color_and_alpha[2] = {, "color"};
			if (info.render_alpha) {
				uh_loop_thing(info, st, frame_idx, camera_idx, "alpha", mats[camera_idx],
				              alpha_img_path);
			}
			uh_loop_thing(info, st, frame_idx, camera_idx, "color", mats[camera_idx], color_img_path);
			// Note! this is going one color and one alpha image at a time.
			add_vignettes(color_img_path, alpha_img_path, info.render_alpha);

			//!@todo probably wrong now that we have hdr backgrounds
			if (info.validation_dataset) {
				do_py_validation_img(color_img_path, alpha_img_path);
			}

			xrt::auxiliary::tracking::CameraCalibrationWrapper wrap(calib->view[camera_idx]);
			xrt_pose ident = XRT_POSE_IDENTITY;
			bool is_valid =
			    project(mats[camera_idx], wrap, camera_idx == 0 ? ident : calibration_left_in_right,
			            st.joints_all_frames[frame_idx].kps);

			if (is_valid) {
				st.num_valid_samples++;
				st.out_valid_samples << frame_idx << ", " << camera_idx << std::endl;
			}
		}


#if 1
		cv::imshow("left", mats[0]);
		cv::imshow("right", mats[1]);
		cv::waitKey(1);
#endif
		st.out_valid_samples.flush();
	}

	st.out_valid_samples.close();

	{
		fs::path num_valid_indices_path = info.out_dir / fs::path("num_valid_samples");

		std::string system_string =
		    "echo " + std::to_string(st.num_valid_samples) + " > " + num_valid_indices_path.string();

		system(system_string.c_str());
	}

	fs::path out_path = info.out_dir / "camera_calibration.json";

	t_stereo_camera_calibration_save(out_path.string().c_str(), calib);

// Note: if Blender starts acting up again, change this to p.kill()
#if 0
	p.wait();
#else
	p.kill();
#endif

	std::cout << "yay" << std::endl;

	return 0;
}
