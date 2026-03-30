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
#include <iostream>
#include <random>
#include "randoviz.hpp"

// #define dist_between_cameras 0.139 // Good enough
// #define num_out_frames 2000

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
	}
};

static const char *kine_keys[5][5] = {

    {
        "UHHHHHNO",
        "THMB_MCP",
        "THMB_PXM",
        "THMB_DST",
        "THMB_TIP",
    },

    {
        "INDX_MCP",
        "INDX_PXM",
        "INDX_INT",
        "INDX_DST",
        "INDX_TIP",
    },

    {
        "MIDL_MCP",
        "MIDL_PXM",
        "MIDL_INT",
        "MIDL_DST",
        "MIDL_TIP",
    },

    {
        "RING_MCP",
        "RING_PXM",
        "RING_INT",
        "RING_DST",
        "RING_TIP",
    },

    {
        "LITL_MCP",
        "LITL_PXM",
        "LITL_INT",
        "LITL_DST",
        "LITL_TIP",
    },
};

void
disp_hand(hand_output &hand)
{
	// Draw axes
	draw_hand_axis_at_pose(hand.wrist, "WRIST");
	for (int finger_idx = 0; finger_idx < 5; finger_idx++) {

		// Draw axes
		for (int bone_idx = 0; bone_idx < 5; bone_idx++) {
			if (finger_idx == 0 && bone_idx == 0) {
				// Hidden extra finger joint
				continue;
			}
			draw_hand_axis_at_pose(hand.fingers[finger_idx][bone_idx], kine_keys[finger_idx][bone_idx]);
		}

		// Draw lines. Note we're going to only the 4th bone
		for (int bone_idx = 0; bone_idx < 4; bone_idx++) {
			if (finger_idx == 0 && bone_idx == 0) {
				// Hidden extra finger joint
				continue;
			}
			sk::vec3 joint0 = hand.fingers[finger_idx][bone_idx].position;
			sk::vec3 joint1 = hand.fingers[finger_idx][bone_idx + 1].position;

			float hue0 = ((finger_idx * 4) + bone_idx) / 24.0f;
			float hue1 = ((finger_idx * 4) + bone_idx + 1) / 24.0f;
			line_add(joint0, joint1, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
			         color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.0005);
		}
	}
}

void
disp_rays_one_camera(one_frame_one_view input, char *name)
{
	draw_camera_axis_at_pose({0, 0, 0}, sk::quat_identity, name);

	for (int i = 0; i < 21; i++) {
		float hue = (float)i / 21.0f;
		float conf = input.confidences[i];
		sk::color32 color = color_to_32(color_hsv(hue, 0.9, conf * conf, 0.4));
		sk::vec3 start = {0, 0, 0};
		sk::vec3 end = *(sk::vec3 *)&input.rays[i];
		sk::line_add(start, end, color, color, 0.0002);
	}
}

void
disp_rays_both_cameras(one_frame_input &frame)
{
	hierarchy_push(matrix_t({-dist_between_cameras / 2, 0, 0}));
	disp_rays_one_camera(frame.views[0], "Left camera");
	hierarchy_pop();


	hierarchy_push(matrix_t({dist_between_cameras / 2, 0, 0}));
	disp_rays_one_camera(frame.views[1], "Right camera");
	hierarchy_pop();
}

void
step(void *ptr)
{
	pgm_state &state(*(pgm_state *)ptr);


#if 0
	bool should_reinit = state.first_frame;
	if (sk::input_key(sk::key_right) & sk::button_state_just_active) {
		state.update_frame_idx(1);
		should_reinit = true;
	}

	if (sk::input_key(sk::key_left) & sk::button_state_just_active) {
		state.update_frame_idx(-1);
		should_reinit = true;
	}


#else
	bool should_reinit = true;
	state.update_frame_idx(1);
#endif

	if (should_reinit) {
		push_frame(state.hand, state.current_frame(), state.last_hand_output);
		// eval_to_viz_hand_2(state.hand, state.last_hand_output_2);
		std::cout << state.frame_idx << std::endl;
	}

	sk::matrix forward = sk::matrix_t({0, 0.1, -0.2});

	sk::hierarchy_push(forward);

	disp_rays_both_cameras(state.current_frame());

	// struct hand_output output;
	disp_hand(state.last_hand_output);

	// sk::matrix side = sk::matrix_t({0.15, 0, 0});

	// sk::hierarchy_push(side);
	// disp_hand(state.last_hand_output_2);

	state.first_frame = false;

	// sk::hierarchy_pop();
	sk::hierarchy_pop();
}

void
shutdown(void *ptr)
{}

int
main(int argc, char *argv[])
{
	u_trace_marker_init();
	sk_settings_t settings = {};
	settings.app_name = "u-uhhhh 🧐🧐";
	settings.assets_folder = "/2/XR/sk-gradient-descent/Assets";
	settings.display_preference = display_mode_flatscreen;
	// settings.display_preference = display_mode_mixedreality;
	settings.flatscreen_width = 1920;
	settings.flatscreen_height = 1080;
	settings.overlay_app = true;
	settings.overlay_priority = 1;
	if (!sk_init(settings))
		return 1;
	// sk::render_set_ortho_size(10.5f);
	// sk::render_set_projection(sk::projection_ortho);
	sk::render_enable_skytex(false);

	struct pgm_state state;



	sk_run_data(step, (void *)&state, shutdown, (void *)&state);


	return 0;
}