#include "kine_lm/lm_interface.hpp"
#include "math/m_api.h"
#include "randoviz.hpp"
#include "util/u_debug.h"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stereokit.h>
#include <random>
#include "make_hand_proportions.hpp"
#include "pose_csv.hpp"
#include "kine_lm/lm_defines.hpp"
#include "pose_diversity_hand_maker.h"

using namespace sk;


#define printf_pose(pose)                                                                                              \
	printf("%f %f %f  %f %f %f %f\n", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,       \
	       pose.orientation.y, pose.orientation.z, pose.orientation.w);

using namespace xrt::tracking::hand::mercury;

// DEBUG_GET_ONCE_FLOAT_OPTION(finger_pose_weight, "FP_WEIGHT", 0);

struct pgm_state
{

	bool going = false;
	struct fingerpose_creator *fingerpose_creator;
};

void
step(void *ptr)
{
	struct pgm_state &st = *(pgm_state *)ptr;

	if (sk::input_key(sk::key_f)) {
		st.going = true;
	}

	if (!st.going) {
		return;
	}

	TrajectorySample<26> out_sample;
	xrt_hand_joint_set joint_set;

	finger_pose_step(st.fingerpose_creator, &joint_set, &out_sample);

	sk::matrix fwd = matrix_t({0, 0, -0.3});

	sk::hierarchy_push(fwd);
	disp_xrt_hand(joint_set);
	// Whatever. This should be disabled sometimes but idk
	for (int i = 0; i < 26; i++) {
		draw_hand_axis_at_pose(sk_from_xrt(out_sample[i]), "a");
	}
	sk::hierarchy_pop();
}

void
shutdown(void *ptr)
{}

int
main(int argc, char *argv[])
{
	// u_trace_marker_init();



	struct pgm_state st;

#if 0
	TrajectoryReader<26> tr(
	    "/3/epics/artificial_data_4/artificial_data_generator/data/finger_pose/training/live0.csv");
	st.fingerpose_creator = create_finger_pose(&tr, black_male_3dscanstore_proportions, true, 1.0f / 60.0f);
#else
	st.fingerpose_creator = create_finger_pose(NULL, black_male_3dscanstore_proportions, true, 1.0f / 60.0f);
#endif


	sk_settings_t settings = {};
	settings.app_name = "u-uhhhh 🧐🧐";
	settings.assets_folder = "/2/XR/sk-gradient-descent/Assets";
	// settings.display_preference = display_mode_flatscreen;
	settings.display_preference = display_mode_mixedreality;
	settings.flatscreen_width = 1920;
	settings.flatscreen_height = 1080;
	settings.overlay_app = true;
	settings.overlay_priority = 1;
	if (!sk_init(settings))
		return 1;
	// sk::render_set_ortho_size(10.5f);
	// sk::render_set_projection(sk::projection_ortho);
	sk::render_enable_skytex(false);



	sk_run_data(step, (void *)&st, shutdown, (void *)&st);

	return 0;
}