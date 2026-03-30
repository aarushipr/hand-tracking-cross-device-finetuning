#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>

#include <string>
#include <unistd.h>
#include "math/m_vec3.h"

#include "util/u_time.h"

#include "xrt/xrt_defines.h"
#include "math/m_space.h"
#include <filesystem>
#include <fstream>
#include "os/os_time.h"
#include "util/u_logging.h"
#include "tracking/t_tracking.h"

#include "tracking/t_calibration_opencv.hpp"

#include <opencv2/opencv.hpp>

struct image_distort_data;

namespace cubemap_view_idx {
enum cubemap_view_idx
{
	FORWARD,
	LEFT,
	RIGHT,
	TOP,
	BOTTOM
};
}

struct cubemap_view
{
	cv::Mat views[5];
};


void
init_distort_image(struct image_distort_data **out_data, t_stereo_camera_calibration *calib);

cv::Mat
distort_image(struct image_distort_data *data, cubemap_view &in, int camera_idx);