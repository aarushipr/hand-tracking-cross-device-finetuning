#include "hg_debug_instrumentation.h"
#include "hg_interface.h"
#include "randoviz.hpp"
#include "stereokit.h"
#include "stereokit_ui.h"
#include "xrt/xrt_defines.h"
using namespace sk;
#include "math/m_api.h"
#include "some_defs.hpp"
// #include "2d_viz.hpp"
// #include "../../monado/src/xrt/drivers/ht/templates/NaivePermutationSort.hpp"
#include "tracking/t_tracking.h"
#include "util/u_logging.h"
#include "euroc_player.hpp"
#include <fenv.h>

const char *view_keys[] = {"left", "right"};

text_style_t styles[3];

const char *hand_class_string[] = {
    "UNKNOWN", "EGO_LEFT", "EGO_RIGHT", "OTHER_LEFT", "OTHER_RIGHT",
};

state_t st = {};


void
assign_texture(struct xrt_frame *frame, struct flat_view &view)
{

	auto img_type = frame->format == XRT_FORMAT_L8 ? CV_8UC1 : CV_8UC3;
	auto cvt_type = frame->format == XRT_FORMAT_L8 ? cv::COLOR_GRAY2RGBA : cv::COLOR_RGB2RGBA;
	cv::Mat frame_mat(cv::Size(frame->width, frame->height), img_type, frame->data, frame->stride);

	cv::Mat tmp_mat;
	cv::cvtColor(frame_mat, tmp_mat, cvt_type);
	tex_set_colors(view.img_tex, frame->width, frame->height, tmp_mat.data);
}

void
update()
{
	// Order doesn't matter much here; it's hard to resize the window AND move the view at the same time

	bool advance_frame = sk::input_key(sk::key_right) & sk::button_state_just_inactive;
	advance_frame = true;


	if (advance_frame) {


		bool is_lefts[2] = {true, false};



		struct xrt_frame *l8_frames[2] = {NULL, NULL};

		for (int view = 0; view < 2; view++) {
			struct xrt_frame *tmp = NULL;

			euroc_player_load_next_frame(st.ep, is_lefts[view], tmp);

			assign_texture(tmp, st.flat_rgb_view[view]);



			cv::Mat tmp_mat(cv::Size(tmp->width, tmp->height), CV_8UC3, tmp->data, tmp->stride);

			cv::Mat mat_l8(cv::Size(960, 960), CV_8UC1);

			cv::cvtColor(tmp_mat, mat_l8, cv::COLOR_BGR2GRAY);

			xrt::auxiliary::tracking::FrameMat::Params params;
			params.timestamp_ns = tmp->timestamp;
			U_LOG_E("%zu", tmp->timestamp);
			params.stereo_format = XRT_STEREO_FORMAT_NONE;

			xrt::auxiliary::tracking::FrameMat::wrapL8(mat_l8, &l8_frames[view], params);

			// cv::imshow("h", tmp_mat);
			// printf("Oh boy\n");
			// cv::cvtColor(tmp_mat, st.flat_rgb_view[view].img_mat, cv::COLOR_BGR2RGBA);
			// st.flat_rgb_view[view].img_mat.convertTo(st.flat_rgb_view[view].img_mat, CV_32F, 1 / 255.0);

			// printf("Oh boy\n");
			// tex_set_colors(st.flat_rgb_view[view].img_tex, 960, 960, st.flat_rgb_view[view].img_mat.data);
			// sk::render_add_model(st.flat_rgb_view[view].img_model, transforms[view]);
		}



		uint64_t ts = {};
		fetestexcept(FE_ALL_EXCEPT);
		t_ht_sync_process(st.sync, l8_frames[0], l8_frames[1], &st.hands[0], &st.hands[1], &ts);

		U_LOG_E("Length %zu", st.the_hg_debug_info->timestamps.size());

		if (st.the_hg_debug_info->timestamps.size() != 6) {
			abort();
		}

		st.timing_writer->push(st.the_hg_debug_info->timestamps);

		for (int i = 0; i < 2; i++) {
			xrt_frame_reference(&l8_frames[i], NULL);
		}

		st.trajectory_writer[0]->push(ts, st.hands[0]);
		st.trajectory_writer[1]->push(ts, st.hands[1]);




		std::cout << st.hands[0].is_active << "	" << st.hands[1].is_active << std::endl;

		st.ep->img_seq++;

		assign_texture(st.the_hg_debug_info->scribbled_frame, st.flat_view_debug);
	}

	sk::hierarchy_push(matrix_t({0, 1, -0.5}));

	sk::matrix transforms[2] = {matrix_t({-.5, 0, 0}), matrix_t({.5, 0, 0})};
	for (int view = 0; view < 2; view++) {
		sk::render_add_model(st.flat_rgb_view[view].img_model, transforms[view]);
	}



	sk::matrix transform_down = matrix_trs({0, -1, 0}, sk::quat_identity, {2, 1, 1});

	sk::render_add_model(st.flat_view_debug.img_model, transform_down);


	sk::hierarchy_pop();

	for (int hand_idx = 0; hand_idx < 2; hand_idx++) {
		if (!st.hands[hand_idx].is_active) {
			continue;
		}
		xrt_hand_joint_set &hand = st.hands[hand_idx];
		struct xrt_hand_joint_value *values = hand.values.hand_joint_set_default;
		for (int joint_idx = 0; joint_idx < 26; joint_idx++) {
			sk::pose_t pose;
			// sk::vec3 position;
			pose.position.x = values[joint_idx].relation.pose.position.x;
			pose.position.y = values[joint_idx].relation.pose.position.y;
			pose.position.z = values[joint_idx].relation.pose.position.z;

			pose.orientation.x = values[joint_idx].relation.pose.orientation.x;
			pose.orientation.y = values[joint_idx].relation.pose.orientation.y;
			pose.orientation.z = values[joint_idx].relation.pose.orientation.z;
			pose.orientation.w = values[joint_idx].relation.pose.orientation.w;
			draw_axis(pose, 0.015f, 0.13);
		}
	}
}

void
create_texture_model(sk::mesh_t &m, struct flat_view &out)
{
	out.img_tex = tex_create(tex_type_image, sk::tex_format_rgba32);
	tex_set_sample(out.img_tex, sk::tex_sample_point);

	out.img_material = material_copy_id("default/material");

	material_set_float(out.img_material, "tex_scale", 1);
	material_set_cull(out.img_material, sk::cull_none);

	material_set_texture(out.img_material, "diffuse", out.img_tex);

	out.img_model = model_create_mesh(m, out.img_material);
}



int
main()
{
	sk_settings_t settings = {};
	settings.app_name = "StereoKit webcam demo!";
	settings.display_preference = sk::display_mode_flatscreen;
	// settings.disable_flatscreen_mr_sim = true;
	if (!sk_init(settings))
		return 1;


	struct t_image_boundary_info info;
	info.views[0].type = HT_IMAGE_BOUNDARY_CIRCLE;
	info.views[1].type = HT_IMAGE_BOUNDARY_CIRCLE;


	//!@todo This changes by like 50ish pixels from device to device. For now, the solution is simple: just
	//! make the circle a bit bigger than we'd like.
	// Maybe later we can do vignette calibration? Write a tiny optimizer that tries to fit Index's
	// gradient? Unsure.
	info.views[0].boundary.circle.normalized_center.x = 0.5f;
	info.views[0].boundary.circle.normalized_center.y = 0.5f;

	info.views[1].boundary.circle.normalized_center.x = 0.5f;
	info.views[1].boundary.circle.normalized_center.y = 0.5f;

	info.views[0].boundary.circle.normalized_radius = 0.55;
	info.views[1].boundary.circle.normalized_radius = 0.55;

	static const char *dataset_path = "/3/epics/dataset_playback/euroc_recording_20220626004733/";
	static const char *calibration_path = "/3/epics/dataset_playback/euroc_recording_20220626004733/calibration.json";

	struct t_stereo_camera_calibration *calibration_ref = NULL;

	t_stereo_camera_calibration_load(calibration_path, &calibration_ref);


	st.sync = t_hand_tracking_sync_mercury_create(calibration_ref, HT_OUTPUT_SPACE_LEFT_CAMERA, info);

	st.the_hg_debug_info = mercury_get_debug_info_pointer(st.sync);

	st.the_hg_debug_info->tuneable_values->scribble_keypoint_model_outputs = true;

	U_LOG_E("%p", st.sync);
	// abort();



	struct euroc_player_config config = {};
	strcpy(config.dataset.path, dataset_path);


	config.dataset.is_stereo = true;
	config.dataset.is_colored = true;
	config.dataset.has_gt = false;
	// config.dataset.gt_device_name  = "What?";
	config.dataset.width = 960;
	config.dataset.height = 960;

	config.playback.stereo = true;
	config.playback.color = true;
	config.playback.skip_perc = false;
	config.playback.skip_first = 0;
	config.playback.scale = 1.0;
	config.playback.max_speed = true;
	config.playback.speed = 1.0;
	// config.send_all

	config.log_level = U_LOGGING_INFO;
	st.ep = euroc_player_create_weird(NULL, &config);

	euroc_player_preload(st.ep);


	// calib.

	sk::mesh_t m = sk::mesh_gen_plane({1, 1}, {0, 0, 1}, {0, 1, 0});

	for (int i = 0; i < 2; i++) {
		create_texture_model(m, st.flat_rgb_view[i]);
	}

	create_texture_model(m, st.flat_view_debug);



	styles[0] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {1, 0, 1, .7});
	styles[1] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {0, 1, 1, .7});
	styles[2] = text_make_style(sk::font_find(default_id_font), 20 * mm2m, {1, 0, 0, .7});

	st.timing_columns = {"function_call_start", "allocate_debug_frame", "acquire_hand_rois", "run_keypoint_estimator_models", "run_optimizer", "anything_else"};


	st.timing_writer = new TimingWriter{"/3/whatever/", "timing.csv", true, st.timing_columns};

	st.trajectory_writer[0] = new HandTrajectoryWriter{"/3/whatever/", "left_smoothed.csv", true};
	st.trajectory_writer[1] = new HandTrajectoryWriter{"/3/whatever/", "right_smoothed.csv", true};



	while (sk_step(update)) {
		// Do nothing! update did everything already!
	};

	sk_shutdown();
	return 0;
}
