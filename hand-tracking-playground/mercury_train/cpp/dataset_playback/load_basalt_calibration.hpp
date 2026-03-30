#pragma once
#include "xrt/xrt_defines.h"
#include "tracking/t_tracking.h"



void
load_basalt_calibration(const char* path, t_stereo_camera_calibration **out_calib);
