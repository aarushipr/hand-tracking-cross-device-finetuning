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
#include "csv.h"


int
main()
{
	cv::Mat blarg(cv::Size(1920, 960), CV_8UC3);
	cv::imshow("h", blarg);
	cv::waitKey(0);

	for (int idx = 0; idx < 100; idx++) {
		cv::Mat mat = cv::imread("/3/inshallah2/" + std::to_string(idx) + ".jpg");
		cv::imshow("h", mat);
		cv::waitKey(0);
	}
}
