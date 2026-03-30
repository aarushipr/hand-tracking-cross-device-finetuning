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


bool
hasEnding(std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

int
main()
{
	uniform_distribution *uniform_distribution = NULL;

	u_random_distribution_create(&uniform_distribution);

	std::vector<fs::path> fingerposes_csvs;
	std::vector<fs::path> wristposes_csvs;

	for (const fs::directory_entry &entry : fs::directory_iterator("/3/whatever2/finger_pose_train/")) {
		if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}
		fingerposes_csvs.push_back(entry.path());
	}

	for (const fs::directory_entry &entry : fs::directory_iterator("/3/whatever2/wrist_pose_train/")) {
				if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}
		wristposes_csvs.push_back(entry.path());
	}



	for (int i = 0; i < 1000; i++) {
		fs::path fingerchoice = fingerposes_csvs[u_random_distribution_get_sample_int(uniform_distribution, 0,
		                                                                              fingerposes_csvs.size())];
		fs::path wristchoice = wristposes_csvs[u_random_distribution_get_sample_int(uniform_distribution, 0,
		                                                                            wristposes_csvs.size())];

		// std::cout << "Making thing out of " << fingerchoice << "and " << wristchoice << std::endl;

		// csvpath_opencv = root_dir / poses_csvname_opencv;
		// csvpath_openxr = root_dir / poses_csvname_openxr;

		create_dataset_info info = {};
		info.wristpose = wristchoice;
		info.wristpose_framerate = 120;

		info.fingerpose = fingerchoice;
		info.fingerpose_framerate = 54;

		info.num_frames = 60;
		info.out_framerate = 30;

		info.out_dir = fs::path("/3/inshallah_throwthisaway") / fs::path("seq" + std::to_string(i));

		info.dont_exit_blender_at_end = true;
		info.dont_render = true;

		// This means we just cycle through them, one after another.
		// Since the trainer code randomly accesses them, this guarantees a uniform distribution.
		info.hand_model_index = i % 3;

		create_dataset(info);
	}
}
