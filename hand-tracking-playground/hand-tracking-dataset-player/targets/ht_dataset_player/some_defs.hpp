#pragma once
// #include "cjson/cJSON.h"
#include "stereokit.h"
#include "stereokit_ui.h"
#include "randoviz.hpp"

#include "tracking/t_hand_tracking.h"
#include "hg_interface.h"
#include "hg_debug_instrumentation.h"

#include "util/u_json.h"
#include "util/u_file.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <unistd.h>

#include <filesystem>
#include <fstream>


using namespace sk;
namespace fs = std::filesystem;
extern const char *view_keys[];

extern const char *hand_class_string[];

extern text_style_t styles[3];

#define CSV_EOL "\r\n"
#define CSV_PRECISION 10


//! Writes poses and their timestamps to a CSV file
class HandTrajectoryWriter
{
public:
	bool enabled; // Modified through UI

private:
	std::string directory;
	std::string filename;
	std::ofstream file;
	bool created = false;

	void
	create()
	{
		std::filesystem::create_directories(directory);
		file = std::ofstream{directory + "/" + filename};
		file << "#timestamp [ns]";
		for (int i = 0; i < 26; i++) {
			file << ",p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
			        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []";
		}
		file << CSV_EOL;

		file << std::fixed << std::setprecision(CSV_PRECISION);
	}


public:
	HandTrajectoryWriter(const std::string &dir, const std::string &fn, bool e) : enabled(e), directory(dir), filename(fn) {}

	void
	push(timepoint_ns ts, const struct xrt_hand_joint_set &hand_pose)
	{
		if (!hand_pose.is_active) {
			return;
		}

		if (!enabled) {
			return;
		}

		if (!created) {
			created = true;
			create();
		}


		file << ts;
		for (int i = 0; i < 26; i++) {
			xrt_vec3 p = hand_pose.values.hand_joint_set_default[i].relation.pose.position;
			xrt_quat r = hand_pose.values.hand_joint_set_default[i].relation.pose.orientation;

			file << "," << p.x << "," << p.y << "," << p.z << ",";
			file << r.w << "," << r.x << "," << r.y << "," << r.z;
		}
		file << CSV_EOL;
	}
};



class TimingWriter
{
public:
	bool enabled; // Modified through UI

private:
	std::string directory;
	std::string filename;
	std::vector<std::string> column_names;
	std::ofstream file;
	bool created = false;

	void
	create()
	{
		std::filesystem::create_directories(directory);
		file = std::ofstream{directory + "/" + filename};
		file << "#";
		for (const std::string &col : column_names) {
			std::string delimiter = &col != &column_names.back() ? "," : CSV_EOL;
			file << col << delimiter;
		}
	}

public:
	TimingWriter(const std::string &dir, const std::string &fn, bool e, const std::vector<std::string> &cn) : enabled(e), directory(dir), filename(fn), column_names(cn) {}

	void
	push(const std::vector<timepoint_ns> &timestamps)
	{
		if (!enabled) {
			return;
		}

		if (!created) {
			created = true;
			create();
		}

		for (const timepoint_ns &ts : timestamps) {
			std::string delimiter = &ts != &timestamps.back() ? "," : CSV_EOL;
			file << ts << delimiter;
		}
	}
};


enum hand_class
{
	UNKNOWN = -1,
	EGO_LEFT = 0,
	EGO_RIGHT = 1,
	OTHER_LEFT = 2,
	OTHER_RIGHT = 3,
};


struct hand_bbox_t
{
	float cx, cy, w, h;
	enum hand_class type;
};

struct view_t
{
	std::vector<hand_bbox_t> hands;
	char *filename;
	uint64_t ts;
};


struct one_frame_t
{
	struct view_t views[2];
	bool handedness_keyframe = false;
	bool positions_confirmed = false;
};

struct flat_view
{
	sk::material_t img_material;
	sk::tex_t img_tex;
	sk::model_t img_model;
};

struct state_t
{
	struct euroc_player *ep;
	struct t_hand_tracking_sync *sync;
	hg_debug_info *the_hg_debug_info;
	TimingWriter *timing_writer;
	HandTrajectoryWriter *trajectory_writer[2];
	std::vector<std::string> timing_columns;

	struct
	{
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-jan26-garage"); // 1260*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-livingroom"); //1210*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-piano"); //101*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/moses-feb6-momsroom-cat"); // 1218*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures-old/moses-feb7-hallway"); // 1218*2
		// fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/seth-skatepark-feb13-1"); // 1218*2
		fs::path root = fs::path("/3/epics/hand_bbox_T32969/bbox-captures/seth-skatepark-feb13-2"); // 1218*2
		fs::path ann_dir = fs::path("annotations");
		fs::path machine_annotated = fs::path("machine_annotated.json");
		fs::path human_annotated = fs::path("human_annotated_last.json");
	} paths;

	struct xrt_hand_joint_set hands[2] = {};
	struct flat_view flat_rgb_view[2];
	struct flat_view flat_view_debug;

	// struct
	// {

	// 	fs::path img_filename;
	// 	struct xrt_frame *img_frame_rgb;
	// 	struct xrt_frame *img_frame_l8;
	// 	cv::Mat img_mat = {};
	// 	sk::material_t img_material;
	// 	sk::tex_t img_tex;
	// 	sk::model_t img_model;

	// 	sk::vec2 mouse_location;

	// } view[2];

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