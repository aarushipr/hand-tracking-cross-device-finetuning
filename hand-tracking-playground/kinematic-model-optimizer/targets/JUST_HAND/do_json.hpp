#pragma once
#include "cjson/cJSON.h"
#include "util/u_json.h"
#include "stdio.h"
#include "stereokit.h"
#include "util/u_logging.h"
#include <cassert>
#include <unistd.h>


typedef struct one_frame_t
{
	xrt_vec3 jts[21];
	int num_hands;
} one_frame_t;



struct ui_state
{
	bool disp_gt = true;
	sk::material_t sp_material;
	sk::model_t sphere;
	int64_t length_ns;
	int64_t num_frames;
	one_frame_t *frames;
};


cJSON *
load_file()
{

	bool json_allocated = false;
	const char *config_path = "../moses.json";

	FILE *config_file = fopen(config_path, "r");
	cJSON *asfd;

	if (config_file == NULL) {
		return NULL;
	}

	fseek(config_file, 0, SEEK_END);    // Go to end of file
	int file_size = ftell(config_file); // See offset we're at. This should be the file size in bytes.
	rewind(config_file);                // Go back to the beginning of the file

	if (file_size == 0) {
		U_LOG_E("0");
		return NULL;
	}
	// else if (file_size > 3 * pow(1024, 200)) { // 3 MiB
	// 	U_LOG_E("30");

	// 	return NULL;
	// }

	char *json = (char *)calloc(file_size + 1, 1);
	json_allocated = true;

	fread(json, 1, file_size, config_file);
	fclose(config_file);
	json[file_size] = '\0';


	asfd = cJSON_Parse(json);
	if (asfd == NULL) {
		const char *error_ptr = cJSON_GetErrorPtr();
		if (error_ptr != NULL) {
		}
		return NULL;
	}

	free(asfd);

	return asfd;
	// Explodes if there's an error. Too bad.
	// parse_error:
	// 	if (json_allocated) {
	// 		free(json);
	// 	}
	// 	U_LOG_E("BAD.");
	// 	return NULL;
}

static const char *keys[21] = {
    "WRIST",

    "THMB_MCP", "THMB_PXM", "THMB_DST", "THMB_TIP",

    "INDX_PXM", "INDX_INT", "INDX_DST", "INDX_TIP",

    "MIDL_PXM", "MIDL_INT", "MIDL_DST", "MIDL_TIP",

    "RING_PXM", "RING_INT", "RING_DST", "RING_TIP",

    "LITL_PXM", "LITL_INT", "LITL_DST", "LITL_TIP",
};

static void
jsonGetHandsData(const cJSON *j_frame, one_frame_t *out_frame)
{

	const cJSON *j_detected_hands = u_json_get(j_frame, "detected_hands");

	const cJSON *j_hand;

	cJSON_ArrayForEach(j_hand, j_detected_hands)
	{
        // Just the first hand.
		for (int idx_joint = 0; idx_joint < 21; idx_joint++) {
			u_json_get_float_array(u_json_get(j_hand, keys[idx_joint]), (float *)&out_frame->jts[idx_joint], 3);
		}
		out_frame->num_hands = 1;
		break;
	}
	// out_frame->num_hands = idx_hand;
}

void
do_json_thing(ui_state *state)
{
	const cJSON *j_root = load_file();


	const cJSON *j_length_ns = u_json_get(j_root, "length_ns");
	double d_length_ns = 0.0f;
	u_json_get_double(j_length_ns, &d_length_ns);
	if (d_length_ns < 1.0f) {
		U_LOG_E("Bad length!");
		abort();
	}
	state->length_ns = d_length_ns;


	const cJSON *j_num_frames = u_json_get(j_root, "num_frames");
	double d_num_frames = 0.0f;
	u_json_get_double(j_num_frames, &d_num_frames);
	if (d_num_frames < 1.0f) {
		abort();
	}
	state->num_frames = d_num_frames;

	state->frames = (one_frame_t *)calloc(state->num_frames, sizeof(one_frame_t));

	const cJSON *j_hand_array = u_json_get(j_root, "hand_array");
	const cJSON *j_hand_array_element = NULL;
	// U_LOG_E("%p %p %p", j_root, j_hand_array, j_hand_array_element);
	int i = 0;
	cJSON_ArrayForEach(j_hand_array_element, j_hand_array)
	{
		// U_LOG_E("ASDF %lu %d", state->num_frames, i);
		// const cJSON *j_ts = u_json_get(j_hand_array_element, "ts");
		// double d_ts;
		// u_json_get_double(j_ts, &d_ts);
		// state->frames[i].ts = d_ts;

		jsonGetHandsData(j_hand_array_element, &state->frames[i]);
		i++;
	}
}