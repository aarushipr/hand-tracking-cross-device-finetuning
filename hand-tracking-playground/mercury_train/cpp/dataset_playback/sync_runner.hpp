
#include "euroc/euroc_interface.h"
#include "hg_debug_instrumentation.hpp"
#include "xrt/xrt_defines.h"
#include "math/m_api.h"
#include "tracking/t_tracking.h"
#include "util/u_logging.h"
#include "util/u_debug_gui.h"
#include <iostream>

struct sync_runner
{
	// struct xrt_frame_node node;
	xrt_frame_context context; // do we need?
	struct xrt_frame *frames[2];

	struct xrt_frame_sink left;
	struct xrt_frame_sink right;
	struct xrt_slam_sinks sinks;

	struct xrt_fs * euroc_player;
	struct t_hand_tracking_sync * ht_algo;

	t_stereo_camera_calibration *calib;
	struct u_debug_gui *debug_gui;
};


