#include <cjson/cJSON.h>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "nms.hpp"
#include "../CLI11.hpp"
#include <filesystem>


#include "terrifying_euroc_stuff.hpp"

namespace fs = std::filesystem;

static const char *camera_names[2] = {"left", "right"};

Ort::Env &
getEnv(void)
{
	static Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "mosesNamespace");

	return env;
}

void
drawBB(float cX, float cY, float w, float h, cv::Mat &img)
{
	// cX*=640.0;
	// cY*=640.0;
	// w*=640.0;
	// h*=640.0;
	cv::circle(img, {(int)cX, (int)cY}, (int)(w / 10), {0, 0, 255});

	cv::Rect rect((int)cX - w / 2, (int)cY - h / 2, (int)w, (int)h);
	cv::rectangle(img, rect, {0, 0, 255});
}

struct hand_detector_t
{
	Ort::Session *session = nullptr;
	Ort::MemoryInfo *memoryInfo = nullptr;
	std::vector<int64_t> inputDims = {};
	const char *inputName = nullptr;

	std::vector<const char *> outputNames = {};
};

struct hand_detector_t
init_hand_detector()
{
	GraphOptimizationLevel onnx_optim = GraphOptimizationLevel::ORT_ENABLE_ALL;
	Ort::SessionOptions sessionOptions = {};
	sessionOptions.SetGraphOptimizationLevel(onnx_optim);
	const char *modelLocation = "../data/best.onnx";
	Ort::Session *session = new Ort::Session(getEnv(), modelLocation, sessionOptions);
	size_t numInputNodes = session->GetInputCount();
	// std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;

	Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
	// Ort::TensorTypeAndShapeInfo inputTASI =
	// inputTypeInfo.GetTensorTypeAndShapeInfo(); int64_t cool[10] = {0};
	// inputTASI.GetDimensions(cool, 10);
	// for (size_t i = 0; i < 10; i++) {
	// 	printf("dim %d\n", cool[i]);
	// }

	std::vector<int64_t> inputDims = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
	size_t num_dims = inputDims.size();
	assert(num_dims == 4); // That's what the hand tracking model wants right now;
	                       // if it's not four we have the wrong guy
	// std::cout << "number of dimensions is " << inputDims.size() << "\n";
	// for (size_t i = 0; i < num_dims; i++) {
	//   std::cout << "dimension " << i << " is " << inputDims[i] << "\n";
	// }

	std::vector<const char *> outputNames;
	Ort::AllocatorWithDefaultOptions allocator;
	char *inputName = session->GetInputName(0, allocator);
	// printf("Input name: %s\n", inputName);

	for (size_t i = 0; i < session->GetOutputCount(); ++i) {
		auto output_name = session->GetOutputName(i, allocator);
		// std::cout << "output " << i << " is " << output_name << std::endl;
		// outputNames.push_back (output_name);
		auto type_info = session->GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> outputDims = tensor_info.GetShape();
		// for (size_t i = 0; i < outputDims.size(); i++) {
		//   std::cout << "   dimension " << i << " is " << outputDims[i] << "\n";
		// }
		// std::cout << "shape is " << tensor_info.GetShape();
	}
	outputNames.push_back("915");
	outputNames.push_back("751");

	// Ort::MemoryInfo temp =  Ort::MemoryInfo::CreateCpu(
	// 		OrtAllocatorType::OrtArenaAllocator,
	// OrtMemType::OrtMemTypeDefault);

	Ort::MemoryInfo *memoryInfo = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));
	struct hand_detector_t hd; // = {session};
	hd.session = session;
	hd.memoryInfo = memoryInfo;
	hd.inputDims = inputDims;
	hd.inputName = inputName;
	hd.outputNames = outputNames;
	return hd;
}

int
main(int argc, char **argv)
{

	CLI::App app{"Automatic bounding box annotator!!"};

	std::string euroc_path_str;
	app.add_option("--euroc_path", euroc_path_str, "Path to EuRoC dataset")->required(true);

	CLI11_PARSE(app, argc, argv);

	std::cout << euroc_path_str << std::endl;


	struct hand_detector_t hd = init_hand_detector();


	fs::path euroc_path_fs = euroc_path_str;

	img_samples ls, rs;

	euroc_player_preload(euroc_path_fs, ls, rs);

	cJSON *j = cJSON_CreateObject();
	cJSON *j_frames = cJSON_AddArrayToObject(j, "frames");

	for (size_t i = 0; i < ls.size(); i++) {
	// for (size_t i = 2400; i < 2410; i++) {
		img_sample s[] = {ls[i], rs[i]};
		cJSON *this_frame = cJSON_CreateObject();
		cJSON_AddNumberToObject(this_frame, "timestamp", s[0].ts);
		cJSON_AddBoolToObject(this_frame, "handedness_keyframe", false);
		cJSON_AddBoolToObject(this_frame, "position_keyframe", false);

		for (int camera_idx = 0; camera_idx < 2; camera_idx++) {
			// printf("%s %s\n", s[0].full_path.c_str(), s[0].within_euroc_path.c_str());

			cJSON *this_frame_camera = cJSON_AddObjectToObject(this_frame, camera_names[camera_idx]);

			cJSON_AddStringToObject(this_frame_camera, "filename", s[camera_idx].within_euroc_path.c_str());


			const size_t inputTensorSize = 640 * 640 * 3; // mostly from Netron and other testing
			std::vector<cv::Mat> planes;


			const char *image_file = s[camera_idx].full_path.c_str();



			cv::Mat raw_input = cv::imread(image_file);

			float scale_factor = 640.0 / raw_input.cols;

			int height = raw_input.rows * scale_factor;

			cv::resize(raw_input, raw_input, {640, height});


			// std::cout << "rawcontinuous? " << raw_input.isContinuous() << "\n";

			cv::Mat img = cv::Mat::zeros(640, 640, CV_8UC3);



			// img.at<uint8_t>(640,0,0)
			// memcpy(img.data + (640 * 80 * 3), raw_input.data, 640 * 480 * 3);
			memcpy(img.data, raw_input.data, 640 * height * 3); // Very very very bad
			                                                    // cv::imshow("coolo",img);
			                                                    // cv::waitKey(0);



			// std::cout << "continuous? " << img.isContinuous() << "\n";
			assert(img.isContinuous());

			// Make image planar instead of interleaved
			cv::split(img, planes);
			cv::Mat red = planes[2];
			cv::Mat green = planes[1];
			cv::Mat blue = planes[0];
			uint8_t combined_planes[640 * 640 * 3] = {0};
			memcpy(combined_planes, red.data, 640 * 640);
			memcpy(combined_planes + (640 * 640), green.data, 640 * 640);
			memcpy(combined_planes + (640 * 640 * 2), blue.data, 640 * 640);
			float real_thing[640 * 640 * 3] = {0};
			for (size_t i = 0; i < 640 * 640 * 3; i++) {
				real_thing[i] = (float)combined_planes[i] / 255.0;
			}
			// Hope it was worth it...

			std::vector<const char *> inputNames{hd.inputName};

			std::vector<Ort::Value> inputTensors;

			inputTensors.push_back(Ort::Value::CreateTensor<float>(*hd.memoryInfo, real_thing, inputTensorSize, hd.inputDims.data(), hd.inputDims.size()));

			std::vector<Ort::Value> out = hd.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, hd.outputNames.data(), hd.outputNames.size());
			float *classes = out[0].GetTensorMutableData<float>();
			// float *boxes = out[1].GetTensorMutableData<float>();
			int stride = 6;
			std::vector<detection> detections;
			int count = 0;
			for (size_t i = 0; i < 25200; i++) {
				int rt = i * stride;
				// printf("rt is %d\n", rt);
				float x = classes[rt];
				float y = classes[rt + 1];
				float w = classes[rt + 2];
				float h = classes[rt + 3];
				float confidence = classes[rt + 4];
				// float unknown = classes[rt + 5];
				// if (left < 200.0 && right > 200.0) {
				// printf("left %f right %f cX %f cY %f %f %f\n", left, right, cX, cY, w,h);

				// }

				if (confidence > 0.6) {
					detection det;
					det.bbox.w = w;
					det.bbox.h = h;
					det.bbox.x = x;
					det.bbox.y = y;
					det.conf = confidence;
					det.class_id = 1;
					det.prob = confidence;

					detections.push_back(det);
					count++;
				}
			}
			FilterBoxesNMS(detections, count, 0.7);

			cJSON *hands_array = cJSON_AddArrayToObject(this_frame_camera, "hands");



			for (detection &det : detections) {
				if (det.prob > 0.001) {
					cJSON *hand_entry = cJSON_CreateArray();

					float cx = det.bbox.x / scale_factor;
					float cy = det.bbox.y / scale_factor;

					float w = det.bbox.w / scale_factor;
					float h = det.bbox.h / scale_factor;

					cJSON_AddItemToArray(hand_entry, cJSON_CreateNumber(cx));
					cJSON_AddItemToArray(hand_entry, cJSON_CreateNumber(cy));
					cJSON_AddItemToArray(hand_entry, cJSON_CreateNumber(w));
					cJSON_AddItemToArray(hand_entry, cJSON_CreateNumber(h));
					cJSON_AddItemToArray(hand_entry, cJSON_CreateNumber(-1));

					cJSON_AddItemToArray(hands_array, hand_entry);

					drawBB(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h, img);
					printf("%f %f %f %f ", det.bbox.x / scale_factor, det.bbox.y / scale_factor, det.bbox.w / scale_factor, det.bbox.h / scale_factor);
				}
			}
			cv::imshow("h", img);
			cv::waitKey(1);
		}
		printf("\n");


		cJSON_AddItemToArray(j_frames, this_frame);
	}

	const char *bleh = cJSON_Print(j);
	printf("%s\n", bleh);

	std::ofstream out(euroc_path_fs / "machine_annotated.json");
	out << bleh;

	return 0;



	// Load image

	// cv::Mat img = cv::imread("righthand.png");



	// return 0;
	// cv::imshow("coolo", img);
	// cv::waitKey(0);
}
