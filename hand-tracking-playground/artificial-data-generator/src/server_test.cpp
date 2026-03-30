#include "artificialdata.pb.h"
#include "math/m_api.h"
#include "math/m_space.h"
#include "randoviz.hpp"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <filesystem>
#include <grpcpp/security/server_credentials.h>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <random>
#include "make_hand_proportions.hpp"

#include <string>

#include <grpcpp/grpcpp.h>
#include "artificialdata.grpc.pb.h"
#include "subprocess.hpp"
#include "csv.hpp"
#include "pose_csv.hpp"
#include "pose_diversity_hand_maker.h"

#include "u_uniform_distribution.h"

#include "util/u_debug.h"

// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;
namespace fs = std::filesystem;


DEBUG_GET_ONCE_BOOL_OPTION(use_first_frame, "GEN_USE_FIRST_FRAME", false)
DEBUG_GET_ONCE_BOOL_OPTION(dont_render, "GEN_DONT_RENDER", false)
DEBUG_GET_ONCE_BOOL_OPTION(dont_exit_immediately, "GEN_DONT_EXIT_IMMEDIATELY", false)
DEBUG_GET_ONCE_TRISTATE_OPTION(use_finger_mocap, "GEN_USE_FINGER_MOCAP")
DEBUG_GET_ONCE_OPTION(finger_mocap_file_override, "GEN_FINGER_MOCAP_FILE", NULL)
DEBUG_GET_ONCE_NUM_OPTION(num_blender_instances, "GEN_NUM_BLENDER_INSTANCES", 9)

fs::path superroot = "/3/inshallah10";

using namespace sk;

#define CSV_EOL "\r\n"
#define CSV_PRECISION 10

#define WRITE_WRIST_REL

#define NUM_MODELS 7
std::string server_address = "0.0.0.0:50052";

#define printf_pose(pose)                                                                                              \
	printf("%f %f %f  %f %f %f %f\n", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,       \
	       pose.orientation.y, pose.orientation.z, pose.orientation.w);

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using namespace xrt::tracking::hand::mercury;

struct single_model_state
{
	std::string name;
	int num_times_generated;
};

struct blender_instance_slot
{
	subprocess::Popen *popen = nullptr;
	int current_model_idx = 0;
};

struct clientrequest_state
{
	fingerpose_creator *creator;

	bool use_finger_mocap = false;

	uniform_distribution *dist;



	TrajectoryWriter<26, std::ostringstream> *finger_trajectory_writer;


	int framerate = 30;
	int num_frames = 200;
	float frametime = 1.0 / framerate;
};


template <typename... Args>
std::string
string_format(const std::string &format, Args... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
	if (size_s <= 0) {
		throw std::runtime_error("Error during formatting.");
	}
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}


void
start_blender(std::vector<blender_instance_slot> &slots, int slot_idx)
{
	blender_instance_slot &slot = slots[slot_idx];

	std::vector<std::string> args = {
	    "blender", "-P", "/3/epics/artificial_data_4/artificial_data_generator/py_generator/blender_main.py"};


	subprocess::environment env = //
	    subprocess::environment{{
	        {"SERVER_ADDRESS", server_address},                                                            //
	        {"SLOT_IDX", string_format("%d", slot_idx)},                                                   //
	        {"MODEL_IDX", string_format("%d", slot.current_model_idx)},                                    //
	        {"DONT_EXIT_IMMEDIATELY", string_format("%d", debug_get_bool_option_dont_exit_immediately())}, //
	    }};

	// Blegh!
	slot.popen = new subprocess::Popen( //
	    std::vector<std::string>(args), //
	    subprocess::environment(env)    //
	);
	// p.wait();
}

void
step(struct clientrequest_state &st)
{

	xrt_hand_joint_set set = {};
	set.is_active = true; // Needed because

	finger_pose_step(st.creator, &set, NULL);

	TrajectorySample<26> hand_pose = {};

	wrist_rel_joint_poses(set, hand_pose);

	st.finger_trajectory_writer->push(0, hand_pose);
}


bool
hasEnding(std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

fs::path
get_csv_within_directory(uniform_distribution *dist, fs::path directory)
{
	std::vector<fs::path> csvs = {};

	for (const fs::directory_entry &entry : fs::directory_iterator(directory)) {
		if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}

		csvs.push_back(entry.path());
	}

	fs::path place = csvs[u_random_distribution_get_sample_int64_t(dist, 0, csvs.size())];
	return place;
}

template <int trnum>
int64_t
get_start_ts_in_tr(clientrequest_state &st, TrajectoryReader<trnum> &tr)
{
	int64_t last_ts = tr.values.back().timestamp;

	last_ts -= 100; // I am so lazy



	int64_t sequence_length = (float)st.num_frames * st.frametime * U_TIME_1S_IN_NS; // One second long

	int64_t last_start_ts = last_ts - sequence_length;

	// std::cout << last_start_ts << std::endl;
	U_LOG_E("lasdf a %ld seqlen %ld last_ts %ld", last_start_ts, sequence_length, last_ts);

	return u_random_distribution_get_sample_int64_t(st.dist, 0, last_start_ts);
}

std::string
handle_csv(clientrequest_state &st)
{
	// uniform_distribution *dist = NULL;
	// u_random_distribution_create(&dist);



#if 1
	fs::path place = get_csv_within_directory(
	    st.dist, "/3/epics/artificial_data_4/artificial_data_generator/data/wrist_elbow_pose/training/");

#else
	fs::path place = "/3/epics/artificial_data_4/artificial_data_generator/data/wrist_elbow_pose/debug/chill.csv";
#endif


	// fs::path place = "/3/whatever2/wrist_pose_train/2.csv";
	TrajectoryReader<2> tr(place);

	int64_t start_ts = 0;

	if (!debug_get_bool_option_use_first_frame()) {
		start_ts = get_start_ts_in_tr(st, tr);
	}


	std::stringstream str;

	TrajectoryWriter<2, std::stringstream> tw(str);


	double t = 0.0f;

	for (int i = 0; i < st.num_frames; i++) {
		int64_t t_ns = (t * U_TIME_1S_IN_NS) + start_ts;
		TrajectorySample<2> hi = {};
		bool retval = tr.get_value(t_ns, hi);
		if (!retval) {
			U_LOG_E("no");
		}

		tw.push(t_ns, hi);


		t += st.frametime;
	}

	return str.str();
}

class ArtificialDataImplementation final : public artificialdata::ArtificialDataNexus::Service
{
	std::mutex self_access_mutex = {};
	int sequence_num = 0;

	int num_sequences_per_model[NUM_MODELS] = {};

	// This is only a vector because we want to control the num of instances with env vars/without recompiling.
	// Populated at startup then elements are no longer added or removed.
	//
public:
	std::vector<blender_instance_slot> blender_instances;

	Status
	goodbye(grpc::ServerContext *context,
	        const artificialdata::sayonara *say,
	        artificialdata::Empty *reply) override
	{
		U_LOG_E("Called");
		this->self_access_mutex.lock();

		int slot_idx = say->slot_idx();
		U_LOG_E("Waiting for Blender instance in %d to exit", slot_idx);
		U_LOG_E("a");
		// Ugh, we need a thread
		this->blender_instances[slot_idx].popen->kill();
		U_LOG_E("OK, Blender instance in %d has exited", slot_idx);

		delete this->blender_instances[slot_idx].popen;

		// Figure out which model idx to use
		int lowest_num_sequences = 100000000;
		int model_with_fewest_sequences = 0;

		for (int i = 0; i < NUM_MODELS; i++) {
			if (this->num_sequences_per_model[i] < lowest_num_sequences) {
				model_with_fewest_sequences = i;
				lowest_num_sequences = this->num_sequences_per_model[i];
			}
		}

		U_LOG_E("OK, going with model %d - %d", model_with_fewest_sequences,
		        this->num_sequences_per_model[model_with_fewest_sequences]);

		this->blender_instances[slot_idx].current_model_idx = model_with_fewest_sequences;
		start_blender(this->blender_instances, slot_idx);


		this->self_access_mutex.unlock();
		return Status::OK;
	}

	void
	decide_background_and_alpha(clientrequest_state &st, bool &out_background, bool &out_alpha)
	{
		float val = u_random_distribution_get_sample_float(st.dist, 0, 1);

		// 10% chance of alpha and regular old lights
		if (val < 0.1) {
			out_background = false;
			out_alpha = true;
		} else if (val < 0.4) {
			// 30% chance of alpha and HDR background lights
			out_background = true;
			out_alpha = true;
		} else {
			// 10% of alpha and randomized lighting
			out_background = true;
			out_alpha = false;
		}

		// out_background = false;
		// out_alpha = true;
	}

	Status
	askForSequence(grpc::ServerContext *context,
	               const artificialdata::sequenceRequest *request,
	               artificialdata::sequenceReply *reply) override
	{

		this->self_access_mutex.lock();

		fs::path seqname = string_format("seq%d", this->sequence_num);
		fs::path seqfolder = superroot / seqname;
		fs::path alpha = seqfolder / "imgs_alpha";
		fs::path color = seqfolder / "imgs_color";



		// std::string model_name = request->model_name();
		int slot_idx = request->slot_idx();
		int model_idx = blender_instances[slot_idx].current_model_idx;
		num_sequences_per_model[model_idx]++;

		this->sequence_num++;



		this->self_access_mutex.unlock();


		fs::remove_all(seqfolder);


		fs::create_directories(alpha);
		fs::create_directories(color);

		reply->set_output_alpha_images_folder(alpha);
		reply->set_output_color_images_folder(color);

		reply->set_hand_poses_csv(seqfolder / "hand_poses.csv");
		reply->set_camera_info_csv(seqfolder / "camera_info.csv");
		reply->set_valid_samples_csv(seqfolder / "valid_samples.csv");



		std::string proportions_json = {};

		proportions_json = request->proportions_json();



		struct clientrequest_state st;

		u_random_distribution_create(&st.dist);

		bool render_alpha;
		bool use_exr_background;

		decide_background_and_alpha(st, use_exr_background, render_alpha);

		reply->set_render_alpha(render_alpha);
		reply->set_use_exr_background(use_exr_background);

		enum debug_tristate_option dto = debug_get_tristate_option_use_finger_mocap();

		if (dto == DEBUG_TRISTATE_OFF) {
			st.use_finger_mocap = false;
		} else if (dto == DEBUG_TRISTATE_ON) {
			st.use_finger_mocap = true;
		} else {
			st.use_finger_mocap = (u_random_distribution_get_sample_float(st.dist, 0, 1) < 0.35);
		}

		TrajectoryReader<26> *tr = NULL;
		int64_t start_ts = 0;

		// 35% chance of finger mocap
		if (st.use_finger_mocap) {
			// THIS LEAKS
			fs::path filep;
			if (debug_get_option_finger_mocap_file_override()) {
				filep = debug_get_option_finger_mocap_file_override();
			} else {
				filep = get_csv_within_directory(
				    st.dist,
				    "/3/epics/artificial_data_4/artificial_data_generator/data/finger_pose/training/");
			}
			tr = new TrajectoryReader<26>(filep);
			if (!debug_get_bool_option_use_first_frame()) {
				start_ts = get_start_ts_in_tr(st, *tr);
			}
			st.creator = create_finger_pose(tr, proportions_json.c_str(), start_ts, st.frametime);
		} else {
			st.creator = create_finger_pose(NULL, proportions_json.c_str(), start_ts, st.frametime);
		}



		U_LOG_E("Got a request!");



		reply->set_num_frames(st.num_frames);

		reply->set_dont_render(debug_get_bool_option_dont_render());
		// reply->set_stop_after_one(debug_get_bool_option_stop_after_one());


		std::ostringstream fingerstream = {};

		TrajectoryWriter<26, std::ostringstream> fingerpose_writer(fingerstream);

		st.finger_trajectory_writer = &fingerpose_writer;

		for (int i = 0; i < st.num_frames; i++) {
			step(st);
		}


		reply->set_fingerpose_csv(fingerstream.str());

		std::string bl = handle_csv(st);
		reply->set_wristpose_csv(bl);

		if (tr != NULL) {
			delete tr;
		}

		return Status::OK;
	}
};



int
main(int argc, char *argv[])
{

	// fs::remove_all(superroot);
	fs::create_directories(superroot);
	for (auto f : fs::directory_iterator(superroot)) {
		fs::remove_all(f);
	}

	std::vector<std::string> args = {
	    "python3", "/3/epics/artificial_data_4/artificial_data_generator/py_generator/loadwatcher.py"};
	subprocess::Popen p = subprocess::Popen(args);


	// std::string server_address(server_address);
	ArtificialDataImplementation service;

	grpc::ServerBuilder builder;



	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);

	std::unique_ptr<Server> server(builder.BuildAndStart());

	U_LOG_E("Starting!");

	int num_instances = debug_get_num_option_num_blender_instances();

	for (int i = 0; i < num_instances; i++) {
		blender_instance_slot slot = {};
		slot.current_model_idx = i % NUM_MODELS;
		service.blender_instances.push_back(slot);
	}

	// The Rube Goldberg machine begins to spin into action...
	for (int i = 0; i < num_instances; i++) {
		start_blender(service.blender_instances, i);
	}
	// This just waits for input on stdin. It's hacky and you should probably use something else
	char bleh[500];

	fgets(bleh, 500, stdin);

	U_LOG_E(
	    "##################################################################\n"
	    "##################################################################\n"
	    "##################################################################\n"
	    "\n"
	    "Stopping! (It'll take a while for all threads to complete, be patient!\n"
	    "\n"
	    "##################################################################\n"
	    "##################################################################\n"
	    "##################################################################\n");

	// Probably fine
	p.kill();


	server->Shutdown();

	server->Wait();

	return 0;
}