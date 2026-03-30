#include <cmath>
#include <string>
#include <iostream>
// #include <filesystem>
// #include <experimental/bits/fs_ops.h>
#include <experimental/filesystem>
#include <memory>
#include <stdexcept>
#include <fstream>

// #include <stdio.h>
// #include <unistd.h>

// #include <opencv2/calib3d.hpp>
// #include <opencv2/core.hpp>

// #include "math/m_vec3.h"
// #include "math/m_vec2.h"

// #include "util/u_time.h"

// #include "xrt/xrt_defines.h"
// #include "math/m_space.h"

// #include "os/os_time.h"
// #include "util/u_logging.h"
// #include "tracking/t_tracking.h"

// #include "tracking/t_calibration_opencv.hpp"

// #include <opencv2/opencv.hpp>
// #include "csv.hpp"
// #include "subprocess.hpp"

// #include "camera_calibration_rightinleft.hpp"
// #include "distort_image.hpp"
// #include "u_uniform_distribution.h"

namespace fs = std::experimental::filesystem;

struct create_dataset_info
{
	bool validation_dataset = false;
	
	fs::path wristpose;
	float wristpose_framerate;

	fs::path fingerpose;
	float fingerpose_framerate;

	int num_frames;
	float out_framerate;

	bool use_exr_background;
	bool render_alpha;

	// Do we want to point it to a camera calibration? I suspect no
	// Hmmm
	// Do we want to give it a list of camera calibrations to randomly interpolate between?

	fs::path out_dir;

	int hand_model_index;

	bool dont_render;
	bool dont_exit_blender_at_end;
};

int
create_dataset(create_dataset_info createInfo);
