
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "math/m_eigen_interop.hpp"

#include "artificialdata_loader.grpc.pb.h"
#include "artificialdata_loader.pb.h"
#include "math/m_api.h"
#include "math/m_space.h"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "util/u_trace_marker.h"
#include "xrt/xrt_defines.h"
#include <filesystem>
#include <grpcpp/security/server_credentials.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <random>

#include <string>

#include <grpcpp/grpcpp.h>
#include "artificialdata.grpc.pb.h"
#include "../subprocess.hpp"
#include "../csv.hpp"
#include "../pose_csv.hpp"

#include "../u_uniform_distribution.h"

#include "util/u_debug.h"
#include "math/m_vec3.h"
#include "math/m_vec2.h"
#include "dataloader_common.hpp"



#define BE_SERVER

fs::path superroot = "/3/inshallah10/";

DEBUG_GET_ONCE_BOOL_OPTION(loadfast, "AD4_LOADFAST", false)

struct AugmentationResult
{
	float homothety_scale;
	float move_overall_variance;
	float move_per_joint_variance;
	float sg_expand_val;
	float stereographic_radius;
	int prediction_type;
};

struct ElbowResult
{
	xrt_vec2 norm_finite_diff_direction_px_coord;
	float dotprod_elbowdir_wristloc;
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

std::string
leftpadfour(int i)
{
	std::string str = string_format("%04d", i);
	return str;
}


void
load_sequence(sequence &out_sequence, int sequence_idx)
{

	std::string seqname = string_format("seq%d", sequence_idx);
	fs::path root = superroot / seqname;

	out_sequence.path = root;


	csv::CSVReader reader_caminfo((root / "camera_info.csv").string());
	csv::CSVReader reader_hand_poses((root / "hand_poses.csv").string());


	for (int frame_idx = 0; frame_idx < 200; frame_idx++) {
		single_frame frame = {};

		csv::CSVRow caminf;
		csv::CSVRow handpose;
		reader_caminfo.read_row(caminf);
		reader_hand_poses.read_row(handpose);

		xrt_pose campose;
		campose.position.x = caminf[0].get<float>();
		campose.position.y = caminf[1].get<float>();
		campose.position.z = caminf[2].get<float>();

		campose.orientation.x = caminf[3].get<float>();
		campose.orientation.y = caminf[4].get<float>();
		campose.orientation.z = caminf[5].get<float>();
		campose.orientation.w = caminf[6].get<float>();

		frame.camera_pose = campose;



		frame.camera_info.fx = caminf[7].get<float>();
		frame.camera_info.fy = caminf[8].get<float>();
		frame.camera_info.cx = caminf[9].get<float>();
		frame.camera_info.cy = caminf[10].get<float>();


		for (int i = 0; i < 26; i++) {
			int root = i * 7;
			frame.hand_landmarks[i].position.x = handpose[root + 0].get<float>();
			frame.hand_landmarks[i].position.y = handpose[root + 1].get<float>();
			frame.hand_landmarks[i].position.z = handpose[root + 2].get<float>();

			// hoooo boy can i trust this? who knowwssss
			frame.hand_landmarks[i].orientation.w = handpose[root + 3].get<float>();
			frame.hand_landmarks[i].orientation.x = handpose[root + 4].get<float>();
			frame.hand_landmarks[i].orientation.y = handpose[root + 5].get<float>();
			frame.hand_landmarks[i].orientation.z = handpose[root + 6].get<float>();

			// Yes, these values are non-zero.
			// U_LOG_E("%f %f %f %f",                         //
			//         frame.hand_landmarks[i].orientation.w, //
			//         frame.hand_landmarks[i].orientation.x, //
			//         frame.hand_landmarks[i].orientation.y, //
			//         frame.hand_landmarks[i].orientation.z);

			// // HACK! THIS SHOULD BE FIXED IN DATA GENERATOR!
			// // We're applying a post-rotation - Blender bones' "forward" is +y.
			// // We're still in Blender world coords, just changing the basis vectors basically.
			// xrt_quat rot_blender_to_xr;
			// xrt_vec3 plusx = XRT_VEC3_UNIT_X;
			// math_quat_from_angle_vector(M_PI / 2, &plusx, &rot_blender_to_xr);

			// math_quat_rotate(&frame.hand_landmarks[i].orientation, &rot_blender_to_xr,
			//                  &frame.hand_landmarks[i].orientation);
		}
		out_sequence.frames.push_back(frame);
	}
}


xrt_vec3
center_of_bounding_cuboid(hand26 &joints)
{
	xrt_vec3 min = joints[0].position;
	xrt_vec3 max = joints[0].position;

	for (int i = 0; i < 25; i++) {
		xrt_vec3 &pt = joints[i].position;
		min.x = fmin(pt.x, min.x);
		min.y = fmin(pt.y, min.y);
		min.z = fmin(pt.z, min.z);

		max.x = fmax(pt.x, max.x);
		max.y = fmax(pt.y, max.y);
		max.z = fmax(pt.z, max.z);
	}

	return m_vec3_mul_scalar((min + max), 0.5);
}

void
predict_points(hand26 &two_frames_ago, hand26 &one_frame_ago, hand26 &out)
{
	for (int i = 0; i < 25; i++) {
		xrt_vec3 from_to = one_frame_ago[i].position - two_frames_ago[i].position;
		out[i].position = one_frame_ago[i].position + from_to;
	}
}

void
predict_points_scaled(hand26 &many_frames_ago, hand26 &one_frame_ago, int num_frames_between_one_and_many, hand26 &out)
{
	for (int i = 0; i < 25; i++) {
		xrt_vec3 from_to = one_frame_ago[i].position - many_frames_ago[i].position;
		from_to = from_to * (1.0f / num_frames_between_one_and_many);
		out[i].position = one_frame_ago[i].position + from_to;
	}
}


void
rotate_points_into_local(xrt_pose camera_pose, hand26 &global, hand26 &out_local)
{
	xrt_quat camposinv;
	math_quat_invert(&camera_pose.orientation, &camposinv);
	for (int i = 0; i < 26; i++) {
		xrt_vec3 fingerpose_global = global[i].position;

		xrt_vec3 fingerpose_local = fingerpose_global - camera_pose.position;



		math_quat_rotate_vec3(&camposinv, &fingerpose_local, &out_local[i].position);
		math_quat_rotate(&camposinv, &global[i].orientation, &out_local[i].orientation);
	}
}

void
homothety(hand26 &global, float scale)
{
	Eigen::Vector3f center = xrt::auxiliary::math::map_vec3(global[11].position);

	Eigen::Affine3f scale_mat = Eigen::Affine3f::Identity();
	scale_mat.linear() *= scale;

	Eigen::Affine3f translation = Eigen::Affine3f::Identity();

	translation.translation() = center * (1 - scale);

	Eigen::Affine3f transform = scale_mat * translation;

	for (int i = 0; i < 25; i++) {
		xrt::auxiliary::math::map_vec3(global[i].position) =
		    transform * xrt::auxiliary::math::map_vec3(global[i].position);
	}
}

void
add_rel_depth(const hand26 &hand, hand25_2d &out_joints_in_img)
{
	float hand_size = hand_length(hand);
	float midpxm_depth = m_vec3_len(hand[11].position);
	for (int i = 0; i < 25; i++) {
		float jd = m_vec3_len(hand[i].position);
		out_joints_in_img[i].depth_relative_to_midpxm = (jd - midpxm_depth) / hand_size;
	}
}

float
normalf(std::mt19937 &mt, float center, float stddev)
{
	std::normal_distribution<float> rd(center, stddev);
	return rd(mt);
}

void
randommove(hand26 &hand, float variance_overall = 0.05, float variance_per_joint = 0.01)
{
	std::random_device dev;
	std::mt19937 mt = std::mt19937(dev());


	xrt_vec3 overall;

	overall.x = normalf(mt, 0, variance_overall);
	overall.y = normalf(mt, 0, variance_overall);
	overall.z = normalf(mt, 0, variance_overall);

	for (int i = 0; i < 25; i++) {
		hand[i].position.x += overall.x + normalf(mt, 0, variance_per_joint);
		hand[i].position.y += overall.y + normalf(mt, 0, variance_per_joint);
		hand[i].position.z += overall.z + normalf(mt, 0, variance_per_joint);
	}
}

float
pront(hand26 &hand_gt_in_camera)
{
	xrt_vec3 wrist = hand_gt_in_camera[0].position;
	xrt_vec3 elbow = hand_gt_in_camera[25].position;

	xrt_vec3 dir_to_wrist = wrist; // heeehee
	xrt_vec3 dir_wrist_to_elbow = elbow - wrist;

	// These could be replaced iwth probably a well-placed len_sqrd
	// But I don't care.
	math_vec3_normalize(&dir_to_wrist);
	math_vec3_normalize(&dir_wrist_to_elbow);

	return m_vec3_dot(dir_to_wrist, dir_wrist_to_elbow);
}

void
get_sample(sequence &ours,
           int frame_idx,
           cv::Mat &out_128x128,
           cv::Mat &out_alpha,
           bool &out_alpha_valid,
           hand25_2d &out_px_coord_gt,
           hand25_2d &out_px_coord_pred,
           AugmentationResult &result,
           ElbowResult &elbow_result)
{

	uniform_distribution *distribution = NULL;
	u_random_distribution_create(&distribution);



	single_frame &f = ours.frames[frame_idx];



	float fx = f.camera_info.fx;
	float fy = f.camera_info.fy;
	float cx = f.camera_info.cx;
	float cy = f.camera_info.cy;



	bool must_not_predict = frame_idx == 0;
	bool can_predict_1 = frame_idx >= 1;
	bool can_predict_2 = frame_idx >= 2;

	bool can_predict_10 = frame_idx >= 10;

	hand26 ground_truth_local;



	rotate_points_into_local(f.camera_pose, f.hand_landmarks, ground_truth_local);

	hand26 predicted_global;


	result.move_overall_variance = 0.015f;
	result.move_per_joint_variance = 0.01f;

	float homothety_min =
	    0.88f; // Intentionally smaller than the other side, to account for per_joint_variance making us bigger
	float homothety_max = 1.1f;
	result.homothety_scale = u_random_distribution_get_sample_float(distribution, homothety_min, homothety_max);


	bool use_predict = u_random_distribution_get_sample_float(distribution, 0.0f, 1.0f) < 0.95;
	use_predict = use_predict && !(must_not_predict);


	if (use_predict) {
		if (can_predict_10 && (u_random_distribution_get_sample_float(distribution, 0.0f, 1.0f) < 0.02)) {
			predict_points_scaled(ours.frames[frame_idx - 10].hand_landmarks,
			                      ours.frames[frame_idx - 1].hand_landmarks, 9, predicted_global);
			result.prediction_type = 3;
		} else if (can_predict_2) {
			predict_points(ours.frames[frame_idx - 2].hand_landmarks,
			               ours.frames[frame_idx - 1].hand_landmarks, predicted_global);
			result.prediction_type = 2;
		} else if (can_predict_1) {
			predicted_global = ours.frames[frame_idx - 1].hand_landmarks;
			result.prediction_type = 1;
		}
	} else {
		result.prediction_type = 0;
		predicted_global = ours.frames[frame_idx].hand_landmarks;
		result.move_overall_variance = 0.04f;
		result.move_per_joint_variance = 0.03f;
		homothety_max = 1.3f;
	}

	hand26 predicted_local;
	rotate_points_into_local(f.camera_pose, predicted_global, predicted_local);

	result.homothety_scale = u_random_distribution_get_sample_float(distribution, homothety_min, homothety_max);
	homothety(predicted_local, result.homothety_scale);



	randommove(predicted_local, result.move_overall_variance, result.move_per_joint_variance);


	xrt_vec3 direction = center_of_bounding_cuboid(predicted_local);


	float twist = u_random_distribution_get_sample_float(distribution, -M_PI, M_PI);

	float expand_val_c = 1.25;
	float expand_val_r = 0.1;
	result.sg_expand_val = u_random_distribution_get_sample_float(distribution, expand_val_c - expand_val_r,
	                                                              expand_val_c + expand_val_r);

	elbow_result.dotprod_elbowdir_wristloc = pront(ground_truth_local);


	std::vector<cv::Mat> color_alpha_in = {};
	std::vector<cv::Mat> color_alpha_out = {};

	color_alpha_out.push_back(cv::Mat());
	color_alpha_out.push_back(cv::Mat());

	////
	fs::path color = ours.path / "imgs_color";
	std::string hey = "Image" + leftpadfour(frame_idx) + ".jpg";
	fs::path full = color / hey;
	cv::Mat mat = cv::imread(full.string());
	color_alpha_in.push_back(mat);
	///
	fs::path alpha = ours.path / "imgs_alpha";
	std::string alpha_filename = "Image" + leftpadfour(frame_idx) + ".jpg";
	fs::path full_alpha = alpha / alpha_filename;
	bool alphaexists = fs::exists(full_alpha);
	cv::Mat maybe_alpha = {}; // nervous about scoping stuff so aaa
	if (alphaexists) {
		maybe_alpha = cv::imread(full_alpha.string());
		color_alpha_in.push_back(maybe_alpha);
		out_alpha_valid = true;
	}


	///
	distort_image(color_alpha_in,                                   //
	              direction,                                        //
	              cv::Scalar{0, 255, 0},                            //
	              f.camera_info,                                    //
	              twist,                                            //
	              result.sg_expand_val,                             //
	              ground_truth_local,                               //
	              predicted_local,                                  //
	              out_px_coord_gt,                                  //
	              out_px_coord_pred,                                //
	              elbow_result.norm_finite_diff_direction_px_coord, //
	              result.stereographic_radius,                      //
	              color_alpha_out);

	add_rel_depth(ground_truth_local, out_px_coord_gt);
	add_rel_depth(predicted_local, out_px_coord_pred);

	std::array<float, 5> curls = {};

	hand_curls(ground_truth_local, curls);

	out_128x128 = color_alpha_out[0];
	if (alphaexists) {
		out_alpha = color_alpha_out[1];
	}


	u_random_distribution_destroy(&distribution);
}
#ifdef BE_SERVER
class ArtificialDataLoaderImplementation final : public artificialdata_loader::ArtificialDataLoader::Service
{

public:
	std::vector<sequence> sequences = {};

	grpc::Status
	askForNumSequences(grpc::ServerContext *context,
	                   const artificialdata_loader::numSequencesRequest *request,
	                   artificialdata_loader::numSequencesReply *reply) override
	{
		reply->set_num(this->sequences.size());
		return grpc::Status::OK;
	}

	grpc::Status
	askForSample(grpc::ServerContext *context,
	             const artificialdata_loader::sampleRequest *request,
	             artificialdata_loader::sampleReply *reply) override
	{
		XRT_TRACE_MARKER();
		U_LOG_D("got a request!");



		size_t seq_idx = request->sequence_idx();
		size_t frame_idx = request->frame_idx();

		U_LOG_D("%zu %zu", seq_idx, frame_idx);

		if (seq_idx >= this->sequences.size()) {
			return grpc::Status::CANCELLED;
		}

		sequence &ours = this->sequences[seq_idx];

		if (frame_idx >= ours.frames.size()) {
			return grpc::Status::CANCELLED;
		}

		// cv::Mat orig;
		cv::Mat dist;
		cv::Mat maybe_dist_alpha;
		bool alpha_valid = false;
		hand25_2d px_coord_gt;
		hand25_2d px_coord_pred;

		AugmentationResult augresult = {};
		ElbowResult elbresult = {};

		get_sample(ours, frame_idx, dist, maybe_dist_alpha, alpha_valid, px_coord_gt, px_coord_pred, augresult,
		           elbresult);

		U_LOG_D("%f %f %f", elbresult.norm_finite_diff_direction_px_coord.x,
		        elbresult.norm_finite_diff_direction_px_coord.y, elbresult.dotprod_elbowdir_wristloc);

		float tglenxy =
		    sqrtf(1.0f - (elbresult.dotprod_elbowdir_wristloc * elbresult.dotprod_elbowdir_wristloc));

		float actlenxy = m_vec2_len(elbresult.norm_finite_diff_direction_px_coord);


		elbresult.norm_finite_diff_direction_px_coord =
		    m_vec2_mul_scalar(elbresult.norm_finite_diff_direction_px_coord, tglenxy / actlenxy);

		U_LOG_D("%f %f %f", elbresult.norm_finite_diff_direction_px_coord.x,
		        elbresult.norm_finite_diff_direction_px_coord.y, elbresult.dotprod_elbowdir_wristloc);
		float len = elbresult.norm_finite_diff_direction_px_coord.x *
		                elbresult.norm_finite_diff_direction_px_coord.x + //
		            elbresult.norm_finite_diff_direction_px_coord.y *
		                elbresult.norm_finite_diff_direction_px_coord.y + //
		            elbresult.dotprod_elbowdir_wristloc * elbresult.dotprod_elbowdir_wristloc;
		U_LOG_D("%f", len);

#if 0
		for (int i = 0; i < 30; i++) {
			for (int j = 0; j < 30; j++) {
		dist.at<char>(32+i, 32+j) = 0;

			}
		}
#endif

		reply->set_image_data(dist.data, 128 * 128);
		if (alpha_valid) {
			U_LOG_D("Alpha valid!");
			reply->set_alpha_data(maybe_dist_alpha.data, 128 * 128);
			reply->set_alpha_data_valid(alpha_valid);
		}
		reply->set_image_width_height(128);
		// reply->
		for (int i = 0; i < 25; i++) {
			reply->add_keypoints_gt_px_x(px_coord_gt[i].x);
			reply->add_keypoints_gt_px_y(px_coord_gt[i].y);
			reply->add_keypoints_gt_depth_rel_midpxm(px_coord_gt[i].depth_relative_to_midpxm);

			reply->add_keypoints_pose_predicted_px_x(px_coord_pred[i].x);
			reply->add_keypoints_pose_predicted_px_y(px_coord_pred[i].y);
			reply->add_keypoints_pose_predicted_depth_rel_midpxm(px_coord_pred[i].depth_relative_to_midpxm);

		}

		reply->set_homothety_scale(augresult.homothety_scale);
		// result.set_mov
		reply->set_sg_expand_val(augresult.sg_expand_val);
		reply->set_stereographic_radius(augresult.stereographic_radius);

		reply->set_prediction_type(augresult.prediction_type);
		reply->set_move_overall_variance(augresult.move_overall_variance);
		reply->set_move_per_joint_variance(augresult.move_per_joint_variance);


		reply->set_elbow_direction_px_x_normalized(elbresult.norm_finite_diff_direction_px_coord.x);
		reply->set_elbow_direction_px_y_normalized(elbresult.norm_finite_diff_direction_px_coord.y);
		reply->set_dotprod_elbowdir_wristloc(elbresult.dotprod_elbowdir_wristloc);



		return grpc::Status::OK;
	}
};
#endif

int
main(int argc, char *argv[])
{

	u_trace_marker_init();



#ifdef BE_SERVER


	ArtificialDataLoaderImplementation impl;

	int num_sequences = 0;

	for (const auto &entry : fs::directory_iterator(superroot)) {
		if (fs::is_directory(entry)) {
			num_sequences++;
		}
	}

	if (debug_get_bool_option_loadfast()) {
		num_sequences = 20;
	}

	for (int i = 0; i < num_sequences; i++) {
		// U_LOG_E("loading sequcence ")
		printf("loading sequence %d", i);
		printf("\r");
		fflush(stdout);
		sequence ours;
		load_sequence(ours, i);
		impl.sequences.push_back(ours);
	}


// AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
	grpc::ServerBuilder builder;

	for (int i = 0; i < 32; i++) {
		int root_port = 50050;
		int this_port = root_port+i;
		std::string server_address = string_format("0.0.0.0:%d", this_port);
		builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

	}

	// std::string server_address("0.0.0.0:50050");
	// std::string server_address_2("0.0.0.0:50051");



	// // handle_csv();
	// // abort();



	// builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	// builder.AddListeningPort(server_address_2, grpc::InsecureServerCredentials());
	builder.RegisterService(&impl);

	grpc::ResourceQuota q("quota");
	q.SetMaxThreads(24);
	q.Resize(10000000000000000000);
	// q.
	builder.SetResourceQuota(q);

	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

	U_LOG_E("Starting!");

	server->Wait();


#else

	sequence ours;

	load_sequence(ours);



	for (int frame_idx = 0; frame_idx < 200; frame_idx++) {

		cv::Mat orig;
		cv::Mat dist;
		hand25_2d px_coord_gt;
		hand25_2d px_coord_pred;

		bool valid;

		get_sample(ours, frame_idx, orig, dist, px_coord_gt, px_coord_pred, valid);

		cv::Mat colored;
		cv::cvtColor(dist, colored, cv::COLOR_GRAY2BGR);

		for (int i = 0; i < 25; i++) {
			cv::circle(colored, {(int)px_coord_pred[i].x, (int)px_coord_pred[i].y}, 2, {0, 255, 0});
			cv::circle(colored, {(int)px_coord_gt[i].x, (int)px_coord_gt[i].y}, 2, {255, 0, 255});
		}

		cv::imshow("aaa", orig);
		cv::imshow("aafa", colored);
		cv::waitKey(0);
	}
#endif
	return 0;
}