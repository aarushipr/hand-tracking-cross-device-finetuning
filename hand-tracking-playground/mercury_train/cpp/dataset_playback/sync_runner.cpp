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
#include "load_basalt_calibration.hpp"

#if 0
#define CALIBRATION_JSON "/4/epics/Datasets2/devices/moshi_reverb_g2/calibration_result/calibration.json"
#define DATASET_PATH  "/4/epics/Datasets2/devices/moshi_reverb_g2/hand_recording_lights_on"
#define CAMERA_INFO get_reverb_info
#else
#define CALIBRATION_JSON "/4/epics/Datasets2/devices/moshi_valve_index/calibration_result/calibration.json"
#define DATASET_PATH  "/4/epics/Datasets2/devices/moshi_valve_index/euroc_recording_20230214182035/"
#define CAMERA_INFO get_index_info

#endif

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
get_calibration(sync_runner &run)
{
	std::string t20_config = read_file("/4/clones/mercury_train/attic/T20_config.json");

	struct vive_config c = {};

	const char *e = t20_config.c_str();

	char *j = (char *)malloc(sizeof(char) * 500000);

	strcpy(j, e);

	vive_config_parse(&c, j, U_LOGGING_ERROR);

	xrt_pose head_in_left; // unused

	vive_get_stereo_camera_calibration(&c, &run.calib, &head_in_left);
}

struct t_camera_extra_info
get_index_info()
{

	// zero-initialized out of paranoia
	struct t_camera_extra_info info = {};

	info.views[0].camera_orientation = CAMERA_ORIENTATION_0;
	info.views[1].camera_orientation = CAMERA_ORIENTATION_0;

	info.views[0].boundary_type = HT_IMAGE_BOUNDARY_CIRCLE;
	info.views[1].boundary_type = HT_IMAGE_BOUNDARY_CIRCLE;

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
	return info;
}

struct t_camera_extra_info
get_reverb_info()
{
	// zero-initialized out of paranoia
	struct t_camera_extra_info info = {};


	info.views[0].camera_orientation = CAMERA_ORIENTATION_270;
	info.views[1].camera_orientation = CAMERA_ORIENTATION_270;

	info.views[0].boundary_type = HT_IMAGE_BOUNDARY_NONE;
	info.views[1].boundary_type = HT_IMAGE_BOUNDARY_NONE;

	return info;
}

void
get_ht_algo(sync_runner &run)
{


	struct t_camera_extra_info info = CAMERA_INFO();

	char path[1024] = {};

	u_file_get_hand_tracking_models_dir(path, ARRAY_SIZE(path));


	run.ht_algo = t_hand_tracking_sync_mercury_create(run.calib, info, path);
}

void
get_euroc_player(sync_runner &run)
{
	euroc_player_config config;

	const char *the_path = DATASET_PATH;

	euroc_player_fill_default_config_for(&config, the_path);
	// config.playback.max_speed = true;
	config.playback.use_source_ts = true;
#if 0
	config.playback.play_from_start = true;
  config.playback.skip_first = 2;
#endif
	config.playback.print_progress = true;
	run.euroc_player = euroc_player_create(&run.context, the_path, &config);
}

static void
receive_left(struct xrt_frame_sink *sink, struct xrt_frame *frame)
{
	struct sync_runner *run_ptr = (sync_runner *)(container_of(sink, struct sync_runner, left));
	struct sync_runner &run = *run_ptr;
	assert(run.frames[0] == NULL);
	xrt_frame_reference(&run.frames[0], frame);
}

static void
receive_right(struct xrt_frame_sink *sink, struct xrt_frame *frame)
{
	struct sync_runner *run_ptr = (sync_runner *)(container_of(sink, struct sync_runner, right));
	struct sync_runner &run = *run_ptr;

	assert(run.frames[0] != NULL);
	assert(run.frames[1] == NULL);

	xrt_frame_reference(&run.frames[1], frame);

	xrt_hand_joint_set hands[2];
	uint64_t out_timestamp_unused;

	// Block till they're processed.
	t_ht_sync_process(run.ht_algo, run.frames[0], run.frames[1], &hands[0], &hands[1], &out_timestamp_unused);

	xrt_frame_reference(&run.frames[0], NULL);
	xrt_frame_reference(&run.frames[1], NULL);
}


int
main()
{

	sync_runner run = {};



	load_basalt_calibration(CALIBRATION_JSON,
	                        &run.calib);
	t_stereo_camera_calibration_dump(run.calib);



	// This definitely needs to be first
	u_debug_gui_create(&run.debug_gui);

	run.left.push_frame = receive_left;
	run.right.push_frame = receive_right;

	run.sinks.cam_count = 2;
	run.sinks.cams[0] = &run.left;
	run.sinks.cams[1] = &run.right;

	// get_calibration(run);
	get_ht_algo(run);
	u_debug_gui_start(run.debug_gui, NULL, NULL);

	get_euroc_player(run);

	xrt_fs_slam_stream_start(run.euroc_player, &run.sinks);



	os_nanosleep(U_TIME_1S_IN_NS * 1000ull);


	u_debug_gui_stop(&run.debug_gui);


	return 0;
}