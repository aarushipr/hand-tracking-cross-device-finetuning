#include "kine_lm/lm_interface.hpp"
#include "math/m_api.h"
#include "randoviz.hpp"
#include "util/u_logging.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <cjson/cJSON.h>
#include <util/u_json.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stereokit.h>
#include <random>

#include "kine_lm/lm_interface.hpp"
#include "kine_lm/lm_defines.hpp"

using namespace sk;
using namespace xrt::tracking::hand::mercury;

const char *black_male_3dscanstore_proportions = R"_({
    "metacarpal_roots": [
        [
            0.2796430289745331,
            -0.14705124497413635,
            -0.20856977999210358
        ],
        [
            0.14629897475242615,
            -0.019592294469475746,
            -0.34980085492134094
        ],
        [
            0.016987890005111694,
            -0.025486260652542114,
            -0.3802037537097931
        ],
        [
            -0.09922372549772263,
            -0.03485030680894852,
            -0.35672733187675476
        ],
        [
            -0.2146603912115097,
            -0.06979238986968994,
            -0.3075757622718811
        ]
    ],
    "metacarpal_plus_x": [
        [
            -0.0404481403529644,
            -0.934524655342102,
            0.35359275341033936
        ],
        [
            0.9754888415336609,
            0.012478617951273918,
            0.21969486773014069
        ],
        [
            0.9995777010917664,
            0.0108641954138875,
            -0.02695089764893055
        ],
        [
            0.9670203924179077,
            0.015156984329223633,
            -0.2542474567890167
        ],
        [
            0.9387156367301941,
            0.01915733516216278,
            -0.34415996074676514
        ]
    ],
    "metacarpal_plus_z": [
        [
            -0.7921674847602844,
            0.24567027390003204,
            0.5586745738983154
        ],
        [
            -0.21806412935256958,
            -0.07897480577230453,
            0.9727337956428528
        ],
        [
            0.027375806123018265,
            -0.041070275008678436,
            0.9987812042236328
        ],
        [
            0.2533673346042633,
            0.044691283255815506,
            0.9663372039794922
        ],
        [
            0.34127551317214966,
            0.08859006315469742,
            0.9357793927192688
        ]
    ],
    "metacarpal_length": [
        0.41646560422405604,
        0.6531925621020456,
        0.6205493709295176,
        0.5389691416404421,
        0.47925196047565183
    ],
    "metacarpal_min_max_x": [
        [
            -1.6441004276275635,
            1.1187562942504883
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ]
    ],
    "metacarpal_min_max_y": [
        [
            -1.734857439994812,
            1.7715094089508057
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ]
    ],
    "metacarpal_min_max_z": [
        [
            -0.8726646304130554,
            0.8726646304130554
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ],
        [
            -0.03490658476948738,
            0.03490658476948738
        ]
    ],
    "nonx_length": [
        [
            0.43605442367252284,
            0.32842375159010473
        ],
        [
            0.48424422319444277,
            0.30936484779036366,
            0.21491366930647218
        ],
        [
            0.49241040463142793,
            0.3796110153584143,
            0.24347751281961136
        ],
        [
            0.5085686179191667,
            0.3501198477036776,
            0.2588817860214048
        ],
        [
            0.41607733650489775,
            0.2589697141509887,
            0.20144822687655065
        ]
    ],
    "hand_size": 0.09565402567386627
}

)_";


void
make_hand_limit(struct cJSON *hi, lm::HandLimit out_hand_limit)
{
	// This is intentionally incomplete. Time pressure
	lm::HandLimit lim = {};

	cJSON *minmax_x = cJSON_GetObjectItem(hi, "metacarpal_min_max_x");
	for (int f = 0; f < 5; f++) {
		cJSON *root = cJSON_GetArrayItem(minmax_x, f);

		cJSON *min = cJSON_GetArrayItem(root, 0);
		cJSON *max = cJSON_GetArrayItem(root, 1);

		if (f == 0) {
			lim.thumb_mcp_swing_x.min = min->valuedouble;
			lim.thumb_mcp_swing_x.max = max->valuedouble;
		} else {
			lim.fingers->mcp_swing_x.min = min->valuedouble;
			lim.fingers->mcp_swing_x.max = max->valuedouble;
		}
	}

	cJSON *minmax_y = cJSON_GetObjectItem(hi, "metacarpal_min_max_y");
	for (int f = 0; f < 5; f++) {
		cJSON *root = cJSON_GetArrayItem(minmax_y, f);

		cJSON *min = cJSON_GetArrayItem(root, 0);
		cJSON *max = cJSON_GetArrayItem(root, 1);

		if (f == 0) {
			lim.thumb_mcp_swing_y.min = min->valuedouble;
			lim.thumb_mcp_swing_y.max = max->valuedouble;
		} else {
			lim.fingers->mcp_swing_y.min = min->valuedouble;
			lim.fingers->mcp_swing_y.max = max->valuedouble;
		}
	}

	// Unsure of what Blender actually uses here, so this might be vaguely wrong but hopefully not too wrong
	cJSON *minmax_z = cJSON_GetObjectItem(hi, "metacarpal_min_max_z");
	for (int f = 0; f < 5; f++) {
		cJSON *root = cJSON_GetArrayItem(minmax_z, f);

		cJSON *min = cJSON_GetArrayItem(root, 0);
		cJSON *max = cJSON_GetArrayItem(root, 1);

		if (f == 0) {
			lim.thumb_mcp_twist.min = min->valuedouble;
			lim.thumb_mcp_twist.max = max->valuedouble;
		} else {
			lim.fingers->mcp_twist.min = min->valuedouble;
			lim.fingers->mcp_twist.max = max->valuedouble;
		}
	}
}

void
make_hand_proportions(const char *proportions_json,
                      lm::hand_proportions &out_hand_proportions,
                      lm::HandLimit &out_hand_limit)
{
	struct cJSON *hi = cJSON_Parse(proportions_json);
	struct lm::hand_proportions props = lm::default_hand_proportions();

	cJSON *size = cJSON_GetObjectItem(hi, "hand_size");
	props.hand_size = size->valuedouble;

	cJSON *roots = cJSON_GetObjectItem(hi, "metacarpal_roots");
	for (int f = 0; f < 5; f++) {
		cJSON *root = cJSON_GetArrayItem(roots, f);

		cJSON *x = cJSON_GetArrayItem(root, 0);
		cJSON *y = cJSON_GetArrayItem(root, 1);
		cJSON *z = cJSON_GetArrayItem(root, 2);
		props.rel_translations.t[f][0].x = x->valuedouble;
		props.rel_translations.t[f][0].y = y->valuedouble;
		props.rel_translations.t[f][0].z = z->valuedouble;
	}

	cJSON *metacarpal_length = cJSON_GetObjectItem(hi, "metacarpal_length");

	for (int f = 0; f < 5; f++) {
		cJSON *len = cJSON_GetArrayItem(metacarpal_length, f);
		if (f == 0) {
			props.rel_translations.t[f][2].z = -len->valuedouble;

		} else {
			props.rel_translations.t[f][1].z = -len->valuedouble;
		}
	}

	cJSON *metacarpal_plus_x = cJSON_GetObjectItem(hi, "metacarpal_plus_x");
	cJSON *metacarpal_plus_z = cJSON_GetObjectItem(hi, "metacarpal_plus_z");
	for (int f = 0; f < 5; f++) {
		xrt_vec3 xrt_x;
		{
			cJSON *root = cJSON_GetArrayItem(metacarpal_plus_x, f);

			cJSON *x = cJSON_GetArrayItem(root, 0);
			cJSON *y = cJSON_GetArrayItem(root, 1);
			cJSON *z = cJSON_GetArrayItem(root, 2);
			xrt_x.x = x->valuedouble;
			xrt_x.y = y->valuedouble;
			xrt_x.z = z->valuedouble;
		}

		xrt_vec3 xrt_z;
		{
			cJSON *root = cJSON_GetArrayItem(metacarpal_plus_z, f);

			cJSON *x = cJSON_GetArrayItem(root, 0);
			cJSON *y = cJSON_GetArrayItem(root, 1);
			cJSON *z = cJSON_GetArrayItem(root, 2);
			xrt_z.x = x->valuedouble;
			xrt_z.y = y->valuedouble;
			xrt_z.z = z->valuedouble;
		}

		xrt_quat q = {};
		math_quat_from_plus_x_z(&xrt_x, &xrt_z, &q);

		props.metacarpal_root_orientations[f] = lm::Quat<HandScalar>(q);
	}

	cJSON *nonx_lengths = cJSON_GetObjectItem(hi, "nonx_length");

	for (int f = 0; f < 5; f++) {
		cJSON *this_finger_nonx_lengths = cJSON_GetArrayItem(nonx_lengths, f);

		if (f == 0) {
			cJSON *f1 = cJSON_GetArrayItem(this_finger_nonx_lengths, 0);
			cJSON *f2 = cJSON_GetArrayItem(this_finger_nonx_lengths, 1);
			props.rel_translations.t[f][3].z = -f1->valuedouble;
			props.rel_translations.t[f][4].z = -f2->valuedouble;
		} else {
			cJSON *f1 = cJSON_GetArrayItem(this_finger_nonx_lengths, 0);
			cJSON *f2 = cJSON_GetArrayItem(this_finger_nonx_lengths, 1);
			cJSON *f3 = cJSON_GetArrayItem(this_finger_nonx_lengths, 2);
			props.rel_translations.t[f][2].z = -f1->valuedouble;
			props.rel_translations.t[f][3].z = -f2->valuedouble;
			props.rel_translations.t[f][4].z = -f3->valuedouble;
		}
	}

	make_hand_limit(hi, out_hand_limit);
	out_hand_proportions = props;
}