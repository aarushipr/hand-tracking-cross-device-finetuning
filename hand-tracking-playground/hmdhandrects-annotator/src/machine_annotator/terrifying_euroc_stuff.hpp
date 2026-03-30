#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include <filesystem>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdio.h>
#include <fstream>
#include <future>
#include <thread>
#include "assert.h"

namespace fs = std::filesystem;



typedef int64_t timepoint_ns;

using std::find_if;
using std::ifstream;
using std::is_same_v;
using std::pair;
using std::stof;
using std::string;

// using img_sample = pair<timepoint_ns, string>;

typedef struct img_sample {
    timepoint_ns ts;
    fs::path full_path;
    fs::path within_euroc_path;

} img_sample;

using img_samples = std::vector<img_sample>;




//! Parse and load image names and timestamps into `samples`
//! If read_n > 0, read at most that amount of samples
//! Returns whether the appropriate data.csv file could be opened
static bool
euroc_player_preload_img_data(const string &dataset_path, img_samples &samples, bool is_left, int64_t read_n = -1)
{
	// Parse image data, assumes data.csv is well formed
	string cam_name = is_left ? "cam0" : "cam1";
	string imgs_path = dataset_path + "/mav0/" + cam_name + "/data";
	string csv_filename = dataset_path + "/mav0/" + cam_name + "/data.csv";
	ifstream fin{csv_filename};
	if (!fin.is_open()) {
		return false;
	}

	string line;
	getline(fin, line); // Skip header line
	while (getline(fin, line) && read_n-- != 0) {
		size_t i = line.find(',');
		timepoint_ns timestamp = stoll(line.substr(0, i));
		string img_name_tail = line.substr(i + 1);

		// Standard euroc datasets use CRLF line endings, so let's remove the extra '\r'
		if (img_name_tail.back() == '\r') {
			img_name_tail.pop_back();
		}

		string img_name = imgs_path + "/" + img_name_tail;
		img_sample sample{timestamp, img_name, fs::path{"mav0"} / fs::path{cam_name} / fs::path{"data"} / img_name_tail};
		samples.push_back(sample);
	}
	return true;
}

//! Trims left and right sequences so that they start and end at the same sample
//! Note that this function does not guarantee that the dataset is free of framedrops.
static void
euroc_player_match_stereo_seqs(img_samples &ls, img_samples &rs)
{

	// Assumes dataset is properly formatted with monotonically increasing timestamps
	timepoint_ns first_ts = MAX(ls.at(0).ts, rs.at(0).ts);
	timepoint_ns last_ts = MIN(ls.back().ts, rs.back().ts);

	auto is_first = [first_ts](const img_sample &s) { return s.ts == first_ts; };
	auto is_last = [last_ts](const img_sample &s) { return s.ts == last_ts; };

	img_samples::iterator lfirst = find_if(ls.begin(), ls.end(), is_first);
	img_samples::iterator llast = find_if(ls.begin(), ls.end(), is_last);
	assert(lfirst != ls.end() && llast != ls.end());

	img_samples::iterator rfirst = find_if(rs.begin(), rs.end(), is_first);
	img_samples::iterator rlast = find_if(rs.begin(), rs.end(), is_last);
	assert(rfirst != rs.end() && rlast != rs.end());

	ls.assign(lfirst, llast + 1);
	rs.assign(rfirst, rlast + 1);
}

void
euroc_player_preload(fs::path path, img_samples &ls, img_samples &rs)
{

	ls.clear();
	euroc_player_preload_img_data(path, ls, true);

	rs.clear();
	euroc_player_preload_img_data(path, rs, false);

	euroc_player_match_stereo_seqs(ls, rs);
}
