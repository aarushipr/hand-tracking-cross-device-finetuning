#pragma once
#include "cjson/cJSON.h"
#include "stereokit.h"
#include "stereokit_ui.h"
#include "randoviz.hpp"

#include "util/u_json.h"
#include "util/u_file.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <unistd.h>

#include <filesystem>


using namespace sk;
namespace fs = std::filesystem;
// extern const char *view_keys[];

// extern const char *hand_class_string[];

extern text_style_t styles[3];


static const char *view_keys[] = {"left", "right"};


static const char *hand_class_string[] = {"UNKNOWN", "EGO_LEFT", "EGO_RIGHT", "OTHER_LEFT", "OTHER_RIGHT"};


enum hand_class
{
	UNKNOWN = -1,
	EGO_LEFT = 0,
	EGO_RIGHT = 1,
	OTHER_LEFT = 2,
	OTHER_RIGHT = 3,
	NUM_VALID_HAND_CLASSES = 5,
};

// int num_hand_classes = 


struct hand_bbox_t
{
	float cx, cy, w, h;
	enum hand_class type;
};

struct view_t
{
	std::vector<hand_bbox_t> hands;
	char *filename;
};


struct one_frame_t
{
	struct view_t views[2];
	bool handedness_keyframe = false;
	bool positions_confirmed = false;
	uint64_t timestamp;
};

struct state_t
{
	struct
	{
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-jan26-garage"); // 1260*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-livingroom"); //1210*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-piano"); //101*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-momsroom-cat"); // 1218*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures-old/moses-feb7-hallway"); // 1218*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/seth-skatepark-feb13-1"); // 1218*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/seth-skatepark-feb13-2"); // 1218*2

		fs::path root = {};
		fs::path machine_annotated = fs::path("machine_annotated.json");
		fs::path human_annotated = fs::path("human_annotated_last.json");
	} paths;

	struct
	{
		fs::path img_filename;
		cv::Mat img_mat;
		sk::material_t img_material;
		sk::tex_t img_tex;
		sk::model_t img_model;

		sk::vec2 mouse_location;

	} view[2];

	struct
	{
		bool active = false;
		sk::vec3 start_point;
	} drawing_new_frame;

	struct
	{
		bool active = false;
		bool after_one_frame = false;
		int start_frame_idx;
		int start_frame_bbox_idx;

		int end_frame_idx;
		int end_frame_bbox_idx;
	} linking_two_boxes;

	int32_t old_display_width;
	int32_t old_display_height;
	sk::vec2 global_mouse_position;

	int num_confirmed = 0;

	int num_frames = 0;
	int curr_frame_idx = 0;

	int focus = 0;
	struct cJSON *json_root;
	struct cJSON *json_frames;

	std::vector<one_frame_t> frames;

	one_frame_t *this_frame;


};