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

#include "util/u_debug.h"

#include <opencv2/opencv.hpp>
#include "csv.hpp"
#include "subprocess.hpp"

#include "camera_calibration_rightinleft.hpp"
#include "distort_image.hpp"
#include "u_uniform_distribution.h"


namespace fs = std::experimental::filesystem;

#define num_threads 6

// we have mblab, 3dscanstore and uhhh right now.
#define num_hand_models 6

DEBUG_GET_ONCE_BOOL_OPTION(validation_dataset, "VALIDATION_DATASET", false);
DEBUG_GET_ONCE_BOOL_OPTION(wait_for_inspect, "GENERATOR_WAIT_FOR_INSPECT", false);
DEBUG_GET_ONCE_NUM_OPTION(generator_num_threads, "GENERATOR_NUM_THREADS", 1);

struct mtc_state
{
	std::mutex contend;
	int current_index = 0;
	uniform_distribution *da_uniform_distribution = NULL;
	std::vector<fs::path> fingerposes_csvs;
	std::vector<fs::path> wristposes_csvs;

	bool should_stop = false;
};


bool
hasEnding(std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

void
decide_background_and_alpha(struct mtc_state *st, bool &out_background, bool &out_alpha)
{
	float val = u_random_distribution_get_sample_float(st->da_uniform_distribution, 0, 1);

	// 10% chance of alpha and regular old lights
	if (val < 0.1) {
		out_background = false;
		out_alpha = true;
	} else if (val < 0.4) {
		out_background = true;
		out_alpha = true;
	} else {
		out_background = true;
		out_alpha = false;
	}

	// out_background = false;
	// out_alpha = true;
}

void
hog(struct mtc_state *st)
{
	while (true) {
		st->contend.lock();
		int our_idx = st->current_index;
		st->current_index += 1;
		st->contend.unlock();



		fs::path fingerchoice = st->fingerposes_csvs[u_random_distribution_get_sample_int(
		    st->da_uniform_distribution, 0, st->fingerposes_csvs.size())];
		fs::path wristchoice = st->wristposes_csvs[u_random_distribution_get_sample_int(
		    st->da_uniform_distribution, 0, st->wristposes_csvs.size())];

		// std::cout << "Making thing out of " << fingerchoice << "and " << wristchoice << std::endl;

		// csvpath_opencv = root_dir / poses_csvname_opencv;
		// csvpath_openxr = root_dir / poses_csvname_openxr;

		create_dataset_info info = {};
		info.validation_dataset = debug_get_bool_option_validation_dataset();
		info.wristpose = wristchoice;
		info.wristpose_framerate = 120;

		info.fingerpose = fingerchoice;
		info.fingerpose_framerate = 54;

		info.num_frames = 60;
		info.out_framerate = 30;

		info.dont_exit_blender_at_end = debug_get_bool_option_wait_for_inspect();
		info.dont_render = debug_get_bool_option_wait_for_inspect();

		fs::path superroot = fs::path("/3/inshallah8");

		if (debug_get_bool_option_validation_dataset()) {
			superroot = "/3/inshallah7_validation";
		}

		info.out_dir = superroot / fs::path("seq" + std::to_string(our_idx));


		// This means that (the final dataset) just cycles through them, one after another.
		// Since the trainer code randomly accesses them, this guarantees a uniform distribution.
		info.hand_model_index = our_idx % num_hand_models;

		decide_background_and_alpha(st, info.use_exr_background, info.render_alpha);

		// info.dont_exit_blender_at_end = false;
		// info.dont_render = false;

		create_dataset(info);
		if (st->should_stop) {
			return;
		}
	}
}


int
main()
{
	struct mtc_state st = {};
	u_random_distribution_create(&st.da_uniform_distribution);

	fs::path finger_pose_root;
	fs::path wrist_pose_root;

	if (debug_get_bool_option_validation_dataset()) {
		finger_pose_root = "/3/whatever2/finger_pose_val/";
		wrist_pose_root = "/3/whatever2/wrist_pose_val/";
	} else {
		finger_pose_root = "/3/whatever2/finger_pose_train/";
		wrist_pose_root = "/3/whatever2/wrist_pose_train/";
	}

	for (const fs::directory_entry &entry : fs::directory_iterator(finger_pose_root)) {
		std::cout << entry.path().string() << std::endl;
		if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}
		st.fingerposes_csvs.push_back(entry.path());
	}

	for (const fs::directory_entry &entry : fs::directory_iterator(wrist_pose_root)) {
		if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}

		st.wristposes_csvs.push_back(entry.path());
	}

	std::vector<std::thread> threads = {};

	for (int i = 0; i < debug_get_num_option_generator_num_threads(); i++) {
		threads.push_back(std::thread(hog, &st));
	}

	// This just waits for input on stdin. It's hacky and you should probably use something else
	char bleh[500];

	fgets(bleh, 500, stdin);

	U_LOG_E(
	    "##################################################################\n"
	    "##################################################################\n"
	    "##################################################################\n"
	    "\n"
	    "Stopping! (It'll take a while for all threads to complete, be patient!\n"
	    "\n"
	    "##################################################################\n"
	    "##################################################################\n"
	    "##################################################################\n");

	st.should_stop = true;
	for (std::thread &thread : threads) {
		thread.join();
	}

	return 0;
}