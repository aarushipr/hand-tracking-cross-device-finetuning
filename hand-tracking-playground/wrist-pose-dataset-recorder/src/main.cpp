#include "stereokit.h"
#include "stereokit_ui.h"
using namespace sk;

#include <stdio.h>

#include <unistd.h>
#include "util/u_time.h"

#include "xrt/xrt_defines.h"
#include "math/m_space.h"
#include <filesystem>
#include <fstream>
#include "os/os_time.h"
#include "util/u_logging.h"

namespace fs = std::filesystem;
#define CSV_EOL "\r\n"
#define CSV_PRECISION 10

//! Writes poses and their timestamps to a CSV file
class TrajectoryWriter
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
		file << "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
		        "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []" CSV_EOL;
		file << std::fixed << std::setprecision(CSV_PRECISION);
	}


public:
	TrajectoryWriter(const std::string &dir, const std::string &fn, bool e)
	    : enabled(e), directory(dir), filename(fn)
	{}

	void
	push(timepoint_ns ts, const xrt_pose &head_pose, const xrt_pose &wrist_pose)
	{
		if (!enabled) {
			return;
		}

		if (!created) {
			created = true;
			create();
		}

		xrt_space_relation out = {};
		struct xrt_relation_chain xrc = {};

		m_relation_chain_push_relation(&xrc, &wrist_pose);
		m_relation_chain_push_inverted_relation(&xrc, &head_pose);
		m_relation_chain_resolve(&xrc, &out);
		xrt_vec3 p = out.pose.position;
		xrt_quat r = out.pose.orientation;


		file << "," << p.x << "," << p.y << "," << p.z << ",";
		file << r.w << "," << r.x << "," << r.y << "," << r.z;
	}
	void
	flush()
	{
		file.flush();
	}
};



struct state
{
	TrajectoryWriter *tw;
	bool started = false;
	uint64_t start_time;
};
struct state the_state;

void
hand_window(sk::handed_ hand, const char *hi)
{
	const hand_t *hande = input_hand(hand);
	if (hande->tracked_state == button_state_inactive)
		return;
	vec3 left_position = hande->palm.position;

	vec3 head_position = input_head()->position;
	pose_t pose;
	pose.position = (left_position * .9) + (head_position * .1);
	pose.orientation = quat_lookat(left_position, head_position);
	// quat_mul(head_potition, quat_from_angles(0, 180, 0)); // head_potition;
	if (input_hand(hand)->tracked_state == button_state_active) {
		hierarchy_push(pose_matrix(pose));
		text_add_at(hi, matrix_identity);
		hierarchy_pop();
		// ui_window_begin(hi, pose, vec2{0, 0});
		// ui_window_end();
	}
}

struct xrt_pose
xrtpose_from_skpose(sk::pose_t in_pose)
{
	xrt_pose pose;
	pose.position.x = in_pose.position.x;
	pose.position.y = in_pose.position.y;
	pose.position.z = in_pose.position.z;

	pose.orientation.w = in_pose.orientation.w;
	pose.orientation.x = in_pose.orientation.x;
	pose.orientation.y = in_pose.orientation.y;
	pose.orientation.z = in_pose.orientation.z;
	return pose;
}



void
push_the_thing(state &st)
{
	const sk::pose_t &head = *sk::input_head();
	const sk::pose_t &wrist = sk::input_hand(sk::handed_left)->wrist;

	xrt_pose head_xrt = xrtpose_from_skpose(head);
	xrt_pose wrist_xrt = xrtpose_from_skpose(wrist);



	xrt_space_relation out = {};
	xrt_relation_chain xrc = {};
	m_relation_chain_push_pose(&xrc, &wrist_xrt);
	m_relation_chain_push_inverted_pose_if_not_identity(&xrc, &head_xrt);
	// XXX: this is NOT how you get the time!!!! It should be close enough but no no no no nono!!!!
	timepoint_ns time = os_monotonic_get_ns();
	// st.tw->push(time, pose);
	m_relation_chain_resolve(&xrc, &out);
	st.tw->push(time, out.pose);
}

void
update()
{
	state &st = the_state;


	const sk::controller_t *cont = sk::input_controller(sk::handed_right);

	sk::line_add_axis(sk::input_hand(sk::handed_left)->wrist, 0.1);

	if (cont->x1 & sk::button_state_just_inactive) {
		if (st.started) {
			st.tw->flush();
			sk::sk_shutdown();
		} else {
			st.started = true;
			st.start_time = os_monotonic_get_ns();
		}
		hand_window(sk::handed_left, "Press B on right controller to start!");
	}



	U_LOG_E("%f %d %d", cont->trigger, cont->x1, cont->x2);

	// if (!st.started) {
	// 	if
	// }


	if (st.started) {
		// XXX: this is NOT how you get the time!!!! It should be close enough but no no no no nono!!!!
		// timepoint_ns time = os_monotonic_get_ns();

		// const sk::hand_t *hand = sk::input_hand(sk::handed_left);

		// sk::pose_t skpose = hand->wrist;
		// xrt_pose pose;
		// pose.position.x = skpose.position.x;
		// pose.position.y = skpose.position.y;
		// pose.position.z = skpose.position.z;

		// pose.orientation.w = skpose.orientation.w;
		// pose.orientation.x = skpose.orientation.x;
		// pose.orientation.y = skpose.orientation.y;
		// pose.orientation.z = skpose.orientation.z;


		// st.tw->push(time, pose);
		push_the_thing(st);

		uint64_t time_now = os_monotonic_get_ns();
		// time_now better be a bigger number than st.start_time unless you're a time traveler
		uint64_t diff = time_now - st.start_time;

		double diff_d = diff;
		diff_d *= 1.0 / (double)U_TIME_1S_IN_NS;

		char bad[128];

		sprintf(bad, "%0.5f", diff_d);

		hand_window(sk::handed_left, bad);

		// If it's been 60 seconds since we started, quit. We want our datasets to be 60 seconds long.
		if (diff_d > 60) {
			st.tw->flush();
			sk::sk_shutdown();
		}
	}
}

int
main()
{
	sk_settings_t settings = {};
	settings.app_name = "demo!";
	settings.display_preference = display_mode_mixedreality;
	if (!sk_init(settings))
		return 1;

	state &st = the_state;

	const char *base_path = getenv("WRECORDER_BASEPATH");
	const char *filename = getenv("WRECORDER_FILENAME");

	// st.tw = new TrajectoryWriter("/3/whatever", "three.csv", true);
	st.tw = new TrajectoryWriter(base_path, filename, true);


	while (sk_step(update)) {
		// Do nothing! update did everything already!
	};

	sk_shutdown();
	return 0;
}
