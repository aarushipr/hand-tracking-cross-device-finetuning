#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "math/m_space.h"
#include "util/u_logging.h"
#include "util/u_time.h"
#include "xrt/xrt_defines.h"
#include <filesystem>
#include "csv.hpp"

// This is a lot like m_relation_history.cpp

namespace fs = std::filesystem;

#define CSV_EOL "\r\n"
#define CSV_PRECISION 10

#define WRITE_WRIST_REL

template <int num_poses> using TrajectorySample = std::array<xrt_pose, num_poses>;


template <int num_poses> class TrajectoryReader
{
public:
	struct retval
	{
		int64_t timestamp;
		TrajectorySample<num_poses> poses;
	};

	std::vector<retval> values = {};

	// void
	// lower_bound_func(TrajectorySample<num_poses> samp, int64_t desired_timestamp);

	TrajectoryReader(fs::path filepath)
	{
		this->values.clear();
		csv::CSVReader reader(filepath.string());

		int64_t first_timestamp = 0;
		bool first = true;

		for (csv::CSVRow &row : reader) {
			retval ret = {};
			int acc_idx = 0;
			if (first) {
				ret.timestamp = 0;
				first_timestamp = row[acc_idx++].get<int64_t>();
			} else {
				ret.timestamp = row[acc_idx++].get<int64_t>() - first_timestamp;
			}
			first = false;
			for (int i = 0; i < num_poses; i++) {
				ret.poses[i].position.x = row[acc_idx++].get<float>();
				ret.poses[i].position.y = row[acc_idx++].get<float>();
				ret.poses[i].position.z = row[acc_idx++].get<float>();

				ret.poses[i].orientation.w = row[acc_idx++].get<float>();
				ret.poses[i].orientation.x = row[acc_idx++].get<float>();
				ret.poses[i].orientation.y = row[acc_idx++].get<float>();
				ret.poses[i].orientation.z = row[acc_idx++].get<float>();
			}

			this->values.push_back(ret);
		}
	}


	bool
	get_value(int64_t at_timestamp_ns, TrajectorySample<num_poses> &out_poses)
	{

		const auto b = this->values.begin();
		const auto e = this->values.end();

		// find the first element *not less than* our value. the lambda we pass is the comparison
		// function, to compare against timestamps.
		const auto sample = std::lower_bound(b, e, at_timestamp_ns, [](const retval &rhe, int64_t timestamp) {
			return rhe.timestamp < timestamp;
		});


		if (at_timestamp_ns == sample->timestamp) {
			// exact match
			U_LOG_T("Exact match in the buffer!");
			// Grrrrr
			retval &p = *sample;
			out_poses = p.poses;
			return true;
		}
		if (sample == e || sample == b) {
			// Pose-predicting doesn't make sense here.
			return false;
		}
		U_LOG_T("Interpolating within buffer!");

		// We precede *it and follow *(it - 1) (which we know exists because we already handled
		// the it = begin() case)
		const auto &predecessor = *(sample - 1);
		const auto &successor = *sample;

		// Do the thing.
		int64_t diff_before = static_cast<int64_t>(at_timestamp_ns) - predecessor.timestamp;
		int64_t diff_after = static_cast<int64_t>(successor.timestamp) - at_timestamp_ns;

		float amount_to_lerp = (float)diff_before / (float)(diff_before + diff_after);

		for (int i = 0; i < num_poses; i++) {
			xrt_pose result{};
			result.position =
			    m_vec3_lerp(predecessor.poses[i].position, successor.poses[i].position, amount_to_lerp);

			math_quat_slerp(&predecessor.poses[i].orientation, &successor.poses[i].orientation,
			                amount_to_lerp, &result.orientation);
			out_poses[i] = result;
		}



		return true;
	}
};



//! Writes poses and their timestamps to an output stream - probablye either a ostringstream or ofstream.
template <int num_poses, typename output_type> class TrajectoryWriter
{
public:
	// Apparently we can't own this. Dragons et al.
	output_type &output;

private:
	bool created = false;

	void
	create()
	{
		output << "#timestamp [ns]";
		for (int i = 0; i < num_poses; i++) {
			output << ",p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], "
			          "q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []";
		}
		output << CSV_EOL;

		output << std::fixed << std::setprecision(CSV_PRECISION);
	}


public:
	TrajectoryWriter(output_type &stream) : output(stream) {}
	// {
	// 	this->output = stream;
	// }

	void
	push(int64_t ts, const TrajectorySample<num_poses> &sample)
	{


		if (!created) {
			created = true;
			create();
		}

		this->output << ts;
		for (int i = 0; i < num_poses; i++) {
			xrt_vec3 p = sample[i].position;
			xrt_quat r = sample[i].orientation;

			this->output << "," << p.x << "," << p.y << "," << p.z << ",";
			this->output << r.w << "," << r.x << "," << r.y << "," << r.z;
		}
		this->output << CSV_EOL;
	}
};


static bool
wrist_rel_joint_poses(xrt_hand_joint_set &hand_pose, TrajectorySample<26> &out_hand_pose)
{

	const xrt_space_relation &wrist = hand_pose.values.hand_joint_set_default[XRT_HAND_JOINT_WRIST].relation;

	for (int i = 0; i < 26; i++) {

		xrt_space_relation out = {};
		struct xrt_relation_chain xrc = {};

		m_relation_chain_push_pose(&xrc, &hand_pose.values.hand_joint_set_default[i].relation.pose);
		m_relation_chain_push_inverted_pose_if_not_identity(&xrc, &wrist.pose);
		m_relation_chain_resolve(&xrc, &out);

		out_hand_pose[i] = out.pose;
	}
	return true;
}
