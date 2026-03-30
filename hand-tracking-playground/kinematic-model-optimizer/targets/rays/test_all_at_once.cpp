// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "math/m_vec3.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_trace_marker.h"

#include "stereokit.h"
#include "stereokit_ui.h"
using namespace sk;

#include <Eigen/Core>
#include <Eigen/Geometry>

// #include "ceres/ceres.h"
// #include "ceres/rotation.h"
// #include "ceres/tiny_solver.h"
// #include "ceres/tiny_solver_autodiff_function.h"

#include "glog/logging.h"


#include <cmath>
#include <random>
#include "randoviz.hpp"
#include <iostream>

#include <fstream>
#include <iomanip>

#include "hand_interface.hpp"


struct file_read_one_frame
{
	xrt_vec2 left[21];
	float confidences_left[21];
	xrt_vec2 right[21];
	float confidences_right[21];
};



xrt_vec3
correct_direction(xrt_vec2 in)
{
	xrt_vec3 out = {in.x, -in.y, -1};
	return m_vec3_normalize(out);
}

struct pgm_state
{
	file_read_one_frame *captured_frames;
	one_frame_input *input_frames;
	int frame_idx = 0;
	LMKinematicHand *hand;
	hand_output last_hand_output;
	hand_output last_hand_output_2;

	std::ofstream timings_file;

	bool first_frame = true;

	void
	update_frame_idx(int diff)
	{
		this->frame_idx += diff;

		if (this->frame_idx < 0) {
			this->frame_idx = num_out_frames + this->frame_idx;
			return;
		}
		this->frame_idx = this->frame_idx % num_out_frames;
		return;
	}

	one_frame_input &
	current_frame()
	{
		return this->input_frames[this->frame_idx];
	}

	pgm_state()
	{
		this->captured_frames = U_TYPED_ARRAY_CALLOC(file_read_one_frame, num_out_frames);

		FILE *f = fopen("../binary_dump_jun3", "r");

		// fseek(f, 0L, SEEK_END);

		// std::cout << "eof?: " << feof(f) << std::endl;
		// std::cout << "error?: " << ferror(f) << std::endl;
		// std::cout << "ftell?: " << ftell(f) << std::endl;
		fread(this->captured_frames, 1, sizeof(file_read_one_frame) * num_out_frames, f);
		std::cout << "eof?: " << feof(f) << std::endl;
		std::cout << "error?: " << ferror(f) << std::endl;
		std::cout << "ftell?: " << ftell(f) << std::endl;
		fclose(f);

		this->input_frames = U_TYPED_ARRAY_CALLOC(one_frame_input, num_out_frames);
		for (int i = 0; i < num_out_frames; i++) {
			for (int j = 0; j < 21; j++) {
				this->input_frames[i].views[0].confidences[j] =
				    this->captured_frames[i].confidences_left[j];
				this->input_frames[i].views[1].confidences[j] =
				    this->captured_frames[i].confidences_right[j];

				this->input_frames[i].views[0].rays[j] =
				    correct_direction(this->captured_frames[i].left[j]);
				this->input_frames[i].views[1].rays[j] =
				    correct_direction(this->captured_frames[i].right[j]);
				float conf_left = this->captured_frames[i].left[j].x;
			}
		}
		create_kinematic_hand(dist_between_cameras, &this->hand);
		// give_identity_hand(this->hand, this->last_hand_output);

		this->timings_file.open("timings.csv");
		this->timings_file << std::fixed << std::setprecision(10) << std::showpos;
	}
};

bool
step(pgm_state &state)
{

	double opt_time = push_frame(state.hand, state.current_frame(), state.last_hand_output);
	std::cout << state.frame_idx << std::endl;

	state.timings_file << opt_time << std::endl;


	state.update_frame_idx(1);
	if (state.frame_idx == 0) {
		return false;
	}
	return true;
}

void
shutdown(void *ptr)
{}

int
main(int argc, char *argv[])
{
	u_trace_marker_init();

	struct pgm_state state;

	while (step(state)) {
		;
	}

	state.timings_file.flush();
	state.timings_file.close();

	return 0;
}