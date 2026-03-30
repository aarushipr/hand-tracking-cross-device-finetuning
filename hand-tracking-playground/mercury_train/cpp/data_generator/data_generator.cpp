#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "math/m_api.h"
#include "math/m_space.h"
#include "math/m_vec3.h"
#include "util/u_file.h"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <filesystem>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <random>
#include "make_hand_proportions.hpp"

#include <string>

#include "shls/subprocess.hpp"
#include "shls/csv.hpp"
#include "aux/pose_csv.hpp"
#include "pose_diversity_hand_maker.h"

#include "aux/u_random_distribution.h"
#include "util/u_json.hpp"

#include "util/u_debug.h"

#include "aux/config_dirs.h"

namespace fs = std::filesystem;


DEBUG_GET_ONCE_BOOL_OPTION(use_first_frame, "GEN_USE_FIRST_FRAME", false)
DEBUG_GET_ONCE_BOOL_OPTION(dont_render, "GEN_DONT_RENDER", false)
DEBUG_GET_ONCE_BOOL_OPTION(dont_exit_immediately, "GEN_DONT_EXIT_IMMEDIATELY", false)
DEBUG_GET_ONCE_TRISTATE_OPTION(use_finger_mocap, "GEN_USE_FINGER_MOCAP")
DEBUG_GET_ONCE_OPTION(finger_mocap_file_override, "GEN_FINGER_MOCAP_FILE", NULL)
DEBUG_GET_ONCE_NUM_OPTION(num_blender_instances, "GEN_NUM_BLENDER_INSTANCES", 9)
DEBUG_GET_ONCE_NUM_OPTION(first_sequence_num, "GEN_FIRST_SEQUENCE_NUM", 0)
DEBUG_GET_ONCE_OPTION(gen_superroot, "GEN_SUPERROOT", nullptr)

DEBUG_GET_ONCE_OPTION(manifest, "GEN_MODELS_MANIFEST", "/4/clones/free_hand_meshes/model_manifest.json")

using xrt::auxiliary::util::json::JSONBuilder;
using xrt::auxiliary::util::json::JSONNode;

#define CSV_EOL "\r\n"
#define CSV_PRECISION 10

#define WRITE_WRIST_REL

#define printf_pose(pose)                                                                                              \
	printf("%f %f %f  %f %f %f %f\n", pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,       \
	       pose.orientation.y, pose.orientation.z, pose.orientation.w);

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

	TrajectoryWriter<26, std::ostringstream> *finger_trajectory_writer;


	int framerate = 30;
	int num_frames = 200;
	float frametime = 1.0 / framerate;
};


struct ArtificialDataImplementation
{
	std::mutex self_access_mutex = {};
	bool should_stop = false;
	int sequence_num = debug_get_num_option_first_sequence_num();

	int last_used_model_idx = {};
	uint32_t port;
	fs::path superroot;


public:
	std::vector<blender_instance_slot> blender_instances;
	int num_hands;
	std::vector<fs::path> blender_files = {};
	std::vector<size_t> num_sequences_per_model = {};
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
get_csv_within_directory(fs::path directory)
{
	std::vector<fs::path> csvs = {};

	for (const fs::directory_entry &entry : fs::directory_iterator(directory)) {
		if (!hasEnding(entry.path().string(), "csv")) {
			continue;
		}

		csvs.push_back(entry.path());
	}

	fs::path place = csvs[u_random_distribution_get_sample_int64_t(0, csvs.size())];
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

	return u_random_distribution_get_sample_int64_t(0, last_start_ts);
}

std::string
handle_csv(clientrequest_state &st)
{



#if 1
	fs::path place =
	    get_csv_within_directory(MERCURY_TRAIN_ROOT_DIR "/data/wrist_elbow_pose/training/");

#else
	fs::path place = MERCURY_TRAIN_ROOT_DIR "/data/wrist_elbow_pose/debug/chill.csv";
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


void
start_blender(ArtificialDataImplementation &impl, int slot_idx)
{
	blender_instance_slot &slot = impl.blender_instances[slot_idx];

	std::vector<std::string> args = {"blender", "-P",
	                                 MERCURY_TRAIN_ROOT_DIR "/py/data_generator/blender_main.py"};

	std::string file = impl.blender_files[slot.current_model_idx].string();


	subprocess::environment env = //
	    subprocess::environment{{
	        {"SERVER_PORT", string_format("%u", impl.port)},                                               //
	        {"SLOT_IDX", string_format("%d", slot_idx)},                                                   //
	        {"MODEL_FILE", file},                                                                          //
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
goodbye(ArtificialDataImplementation &impl, int slot_idx)
{
	U_LOG_E("Called");
	impl.self_access_mutex.lock();

	U_LOG_E("Waiting for Blender instance in %d to exit", slot_idx);
	U_LOG_E("a");
	// Ugh, we need a thread
	impl.blender_instances[slot_idx].popen->kill();
	U_LOG_E("OK, Blender instance in %d has exited", slot_idx);

	delete impl.blender_instances[slot_idx].popen;

	// Figure out which model idx to use
	size_t lowest_num_sequences = 100000000;
	size_t model_with_fewest_sequences = 0;

	for (int i = 0; i < impl.num_hands; i++) {
		// Very crappy damped control theory:
		// If we just assigned one Blender instance to work on a certain model, we don't want the _next_
		// one to work on that model just 'cause it still has the lowest number of sequences.
		// This isn't a perfect solution either but should stop oscillations.
		if (i == impl.last_used_model_idx) {
			continue;
		}
		if (impl.num_sequences_per_model[i] < lowest_num_sequences) {
			model_with_fewest_sequences = i;
			lowest_num_sequences = impl.num_sequences_per_model[i];
		}
	}

	U_LOG_E("OK, going with model %lu - %lu", model_with_fewest_sequences,
	        impl.num_sequences_per_model[model_with_fewest_sequences]);

	impl.blender_instances[slot_idx].current_model_idx = model_with_fewest_sequences;
	start_blender(impl, slot_idx);
	impl.last_used_model_idx = model_with_fewest_sequences;


	impl.self_access_mutex.unlock();
	return;
}

void
decide_background_and_alpha(clientrequest_state &st, bool &out_background, bool &out_alpha)
{
	float val = u_random_distribution_get_sample_float(0, 1);

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

void
askForSequence(struct ArtificialDataImplementation &impl, JSONNode &request, JSONBuilder &reply)
{
	U_LOG_E("Called!");
	impl.self_access_mutex.lock();

	fs::path seqname = string_format("seq%d", impl.sequence_num);
	fs::path seqfolder = impl.superroot / seqname;
	fs::path alpha = seqfolder / "imgs_alpha";
	fs::path color = seqfolder / "imgs_color";


	// std::string model_name = request->model_name();
	int slot_idx = request["slot_idx"].asInt();
	int model_idx = impl.blender_instances[slot_idx].current_model_idx;
	impl.num_sequences_per_model[model_idx]++;

	impl.sequence_num++;



	impl.self_access_mutex.unlock();


	fs::remove_all(seqfolder);


	fs::create_directories(alpha);
	fs::create_directories(color);


	reply << "{";

	reply << "output_alpha_images_folder" << alpha;
	reply << "output_color_images_folder" << color;

	reply << "hand_poses_csv" << std::string(seqfolder / "hand_poses.csv");

	reply << "camera_info_csv" << std::string(seqfolder / "camera_info.csv");
	reply << "valid_samples_csv" << std::string(seqfolder / "valid_samples.csv");

	JSONNode proportions_json = request["proportions"];


	struct clientrequest_state st;


	bool render_alpha;
	bool use_exr_background;

	decide_background_and_alpha(st, use_exr_background, render_alpha);

	reply << "render_alpha" << render_alpha;
	reply << "use_exr_background" << use_exr_background;

	enum debug_tristate_option dto = debug_get_tristate_option_use_finger_mocap();

	if (dto == DEBUG_TRISTATE_OFF) {
		st.use_finger_mocap = false;
	} else if (dto == DEBUG_TRISTATE_ON) {
		st.use_finger_mocap = true;
	} else {
		st.use_finger_mocap = (u_random_distribution_get_sample_float(0, 1) < 0.35);
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
			filep =
			    get_csv_within_directory(MERCURY_TRAIN_ROOT_DIR "/data/finger_pose/training/");
		}
		tr = new TrajectoryReader<26>(filep);
		if (!debug_get_bool_option_use_first_frame()) {
			start_ts = get_start_ts_in_tr(st, *tr);
		}
		st.creator = create_finger_pose(tr, proportions_json.getCJSON(), start_ts, st.frametime);
	} else {
		st.creator = create_finger_pose(NULL, proportions_json.getCJSON(), start_ts, st.frametime);
	}



	U_LOG_E("Got a request!");



	reply << "num_frames" << st.num_frames;

	reply << "dont_render" << debug_get_bool_option_dont_render();


	std::ostringstream fingerstream = {};

	TrajectoryWriter<26, std::ostringstream> fingerpose_writer(fingerstream);

	st.finger_trajectory_writer = &fingerpose_writer;

	for (int i = 0; i < st.num_frames; i++) {
		step(st);
	}


	// note! it's stupid that we're embedding CSV files into json! it works fine because Python's json library
	// transparently parses the \\r\\n's out, but that doesn't make it a good choice! Doing it this way made way
	// more sense when we were using gRPC/protobuf and not bare sockets/json.
	//
	// If stuff like this trips you up in the future, please move to *bare sockets/protobuf*! protobuf is pretty
	// good, it's just gRPC that isn't.

	reply << "fingerpose_csv" << fingerstream.str();

	std::string bl = handle_csv(st);
	reply << "wristpose_csv" << bl;

	reply << "}";

	if (tr != NULL) {
		delete tr;
	}

	return;
}

void
parse_manifest(ArtificialDataImplementation &service)
{
	const char *m = debug_get_option_manifest();
	fs::path manifest_path(m);
	fs::path dir = manifest_path.parent_path();



	const char *file_content = u_file_read_content_from_path(m);
	cJSON *config_json = cJSON_Parse(file_content);

	printf("%s\n", file_content);

	const cJSON *hs = u_json_get(config_json, "hands");
	int num_hands = cJSON_GetArraySize(hs);
	service.num_hands = num_hands;

	for (int i = 0; i < num_hands; i++) {
		const cJSON *h = cJSON_GetArrayItem(hs, i);
		const cJSON *blend_file_j = cJSON_GetObjectItem(h, "blender_file");

		const char *blend_file = blend_file_j->valuestring;

		// This one segfaults
		fs::path blend_file_rel(blend_file);
		fs::path full = dir / blend_file_rel;
		service.blender_files.push_back(full);
	}
}

void
error(const char *msg)
{
	perror(msg);
	exit(1);
}


void
wait_for_exit_thread(ArtificialDataImplementation *impl)
{
	// This just waits for input on stdin. It's hacky and you should probably use something else
	char bleh[500];

	char *ret = fgets(bleh, 500, stdin);

	// We don't care about what the user wrote, we're just leaving.
	(void)ret;

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
	impl->should_stop = true;
	return;
}

#define BUFSIZE 1024 * 1024 * 64
#define MAX_CONNECTIONS 8
int sockfd, newsockfd;


int
main() // int argc, char *argv[])
{
	U_LOG_E("I am alive! Code root dir is %s", MERCURY_TRAIN_ROOT_DIR);
	// fs::remove_all(impl.superroot);

		ArtificialDataImplementation service;
	parse_manifest(service);


	const char *sr = debug_get_option_gen_superroot();
	if (sr == NULL) {
		U_LOG_E("Need a root dir to output secquences");
		abort();
	}

	service.superroot = sr;
	fs::create_directories(service.superroot);
	for (auto f : fs::directory_iterator(service.superroot)) {
		fs::remove_all(f);
	}

	// std::vector<std::string> args = {"python3",
	// MERCURY_TRAIN_ROOT_DIR "/py/data_generator/loadwatcher.py"}; subprocess::Popen p =
	// subprocess::Popen(args);




	socklen_t clilen;
	char *buffer = (char *)malloc(BUFSIZE);

	struct sockaddr_in serv_addr, cli_addr;
	int n;

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	int flag = 1;
	// SO_REUSEADDR makes the OS reap this socket right after we quit.
	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
	if (sockfd < 0)
		error("ERROR opening socket");

	socklen_t addrlen = sizeof(serv_addr);


	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	serv_addr.sin_port = 0; // Set port to 0 to let the OS choose a port


	if (bind(sockfd, (struct sockaddr *)&serv_addr, addrlen) < 0)
		error("ERROR on binding");

	// Get the port the OS chose for us
	if (getsockname(sockfd, (struct sockaddr *)&serv_addr, &addrlen) < 0) {
		perror("getsockname");
		exit(EXIT_FAILURE);
	}

	service.port = ntohs(serv_addr.sin_port);
	int num_instances = debug_get_num_option_num_blender_instances();

	listen(sockfd, num_instances+100);
	clilen = sizeof(cli_addr);


	U_LOG_E("Starting!");


	for (int i = 0; i < num_instances; i++) {
		blender_instance_slot slot = {};
		slot.current_model_idx = i % service.num_hands;
		service.blender_instances.push_back(slot);
		service.num_sequences_per_model.push_back(0);
	}

	// The Rube Goldberg machine begins to spin into action...
	for (int i = 0; i < num_instances; i++) {
		start_blender(service, i);
	}

	std::thread stop_thread = std::thread(wait_for_exit_thread, &service);


	while (!service.should_stop) {
		newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
		if (newsockfd < 0)
			error("ERROR on accept");
		bzero(buffer, BUFSIZE);
		n = read(newsockfd, buffer, BUFSIZE - 1);
		if (n < 0)
			error("ERROR reading from socket");
		printf("C++: GOT %s\n", buffer);
		{
			cJSON *c = cJSON_Parse(buffer);
			JSONNode request(c);
			JSONBuilder reply{};
			if (request["message_type"].asString() == "normal") {
				askForSequence(service, request, reply);
			} else {
				goodbye(service, request["slot_idx"].asInt());
			}
			std::shared_ptr<JSONNode> node = reply.getBuiltNode();
			char *blah = cJSON_Print(node->getCJSON());

			U_LOG_E("Going to try to send %s", blah);


			n = write(newsockfd, blah, strlen(blah));
			free(blah);
			if (n < 0)
				error("ERROR writing to socket");
			close(newsockfd);
		}
	}


	stop_thread.join();


	return 0;
}