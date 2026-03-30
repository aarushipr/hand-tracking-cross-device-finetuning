#include "kine_lm/lm_interface.hpp"
#include "math/m_api.h"
#include "util/u_logging.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <cjson/cJSON.h>
#include <util/u_json.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include "kine_lm/lm_interface.hpp"


using namespace xrt::tracking::hand::mercury;

extern const char *black_male_3dscanstore_proportions;

void
make_hand_proportions(const cJSON *proportions_json,
                      lm::hand_proportions &out_hand_proportions,
                      lm::HandLimit &out_hand_limit);
