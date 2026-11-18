// Adapted from the upstream C++ demo; retain UFLDv2 crop, anchors and decoding.
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

vector<int64_t> argmax_1(const float* v, const vector<int64_t>& dims)
{
	vector<int64_t> ret;
	ret.resize(dims[0] * dims[2] * dims[3]);

	for (int64_t i = 0; i < dims[2]; i++) {
		for (int64_t j = 0; j < dims[3]; j++) {
			int64_t offset = dims[3] * i + j;
			float max_val = -std::numeric_limits<float>::infinity();
			int64_t max_index = 0;
			for (int64_t k = 0; k < dims[1]; k++) {
				size_t index = k * dims[2] * dims[3] + offset;
				if (v[index] > max_val) {
					max_val = v[index];
					max_index = k;
				}
			}
			ret[offset] = max_index;
		}
	}
	return ret;
}

int64_t sum_valid(const vector<int64_t>& v, int64_t num, int64_t interval, int64_t offset)
{
	int64_t sum = 0;
	for (int64_t i = 0; i < num; i++) {
		sum += v[i * interval + offset];
	}
	return sum;
}

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct InferenceConfig {
    string input_color = "bgr";
    string enhance = "none";
    double row_min_ratio = 0.5;
    double col_min_ratio = 0.25;
};

static Json inference_settings(const InferenceConfig& config) {
    return {{"input_color", config.input_color}, {"enhance", config.enhance},
            {"row_min_ratio", config.row_min_ratio}, {"col_min_ratio", config.col_min_ratio},
            {"crop_ratio", 0.6}};
}

static bool passes_lane_gate(int64_t valid, int64_t total, double ratio) {
    return valid > total * ratio;
}

static Mat enhance_pixels(const Mat& bgr, const InferenceConfig& config) {
    if (config.enhance == "none") return bgr;
    Mat ycrcb, result;
    cvtColor(bgr, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);
    createCLAHE(2.0, Size(8, 8))->apply(channels[0], channels[0]);
    merge(channels, ycrcb);
    cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
    return result;
}

static vector<float> normalize_pixels(const Mat& img, int height, int width,
                                      const InferenceConfig& config) {
    if (img.type() != CV_8UC3 || img.cols != width || img.rows < height)
        throw runtime_error("Expected resized CV_8UC3 image with enough rows for bottom crop");
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};
    vector<float> tensor(size_t(height) * width * 3);
    size_t index = 0;
    for (int c = 0; c < 3; ++c) {
        const int source_channel = config.input_color == "rgb" ? 2 - c : c;
        for (int y = img.rows - height; y < img.rows; ++y)
            for (int x = 0; x < width; ++x) {
                float pixel = img.ptr<uchar>(y)[x * 3 + source_channel];
                tensor[index++] = (pixel / 255.0 - mean[c]) / stddev[c];
            }
    }
    return tensor;
}

class Ultra_Fast_Lane_Detection_v2
{
public:
	Ultra_Fast_Lane_Detection_v2(const string& model_path, const string& device, int gpu_id, const fs::path& profile_dir,
                                const InferenceConfig& config = {});
	Mat detect(Mat& cv_image);
	void verify_profile(bool require_cuda);
    void benchmark(int warmup, int iterations);
    vector<vector<Point>> lanes;
    Json diagnostics = Json::array();
    double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0;

private:
    InferenceConfig config_;
	void normalize_(Mat img);
	float SoftMaxFast(const float* src, float* dst, int64_t length);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	string dataset;
	int num_row;
	int num_col;
	vector<float> row_anchor;
	vector<float> col_anchor;
	float crop_ratio;
	void GenerateAnchor();

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Ultra-Fast-Lane-Detection-v2");
	std::unique_ptr<Ort::Session> ort_session;
	SessionOptions sessionOptions = SessionOptions();
	vector<string> input_names;
	vector<string> output_names;
    vector<const char*> input_ptrs, output_ptrs;
    vector<Value> run();
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};

Ultra_Fast_Lane_Detection_v2::Ultra_Fast_Lane_Detection_v2(
    const string& model_path, const string& device, int gpu_id, const fs::path& profile_dir,
    const InferenceConfig& config) : config_(config)
{
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    // Keep CPU resource usage bounded and reproducible on laptop systems.
    sessionOptions.SetIntraOpNumThreads(4);
    fs::create_directories(profile_dir);
    auto prefix = profile_dir / ("ort_" + device + "_" +
        to_string(Clock::now().time_since_epoch().count()));
    sessionOptions.EnableProfiling(prefix.c_str());
    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = gpu_id;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    // fs::path is narrow on Linux and wide on Windows, as required by ORT.
    ort_session = std::make_unique<Session>(env, fs::path(model_path).c_str(), sessionOptions);
    AllocatorWithDefaultOptions allocator;
    if (ort_session->GetInputCount() != 1 || ort_session->GetOutputCount() != 4)
        throw runtime_error("Expected one input and four UFLDv2 outputs");
    for (size_t i = 0; i < ort_session->GetInputCount(); ++i) {
        input_names.emplace_back(ort_session->GetInputNameAllocated(i, allocator).get());
        auto type = ort_session->GetInputTypeInfo(i);
        auto info = type.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw runtime_error("Input must be float32");
        input_node_dims.push_back(info.GetShape());
    }
    for (size_t i = 0; i < ort_session->GetOutputCount(); ++i) {
        output_names.emplace_back(ort_session->GetOutputNameAllocated(i, allocator).get());
        auto type = ort_session->GetOutputTypeInfo(i);
        auto info = type.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw runtime_error("Outputs must be float32");
        output_node_dims.push_back(info.GetShape());
    }
    // Bind the four heads by their unique validated CULane shapes. Exporters
    // may rename/reorder graph outputs; do not depend on basename or node names.
    const vector<int64_t> expected_input{1, 3, 320, 1600};
    const vector<vector<int64_t>> expected_outputs{
        {1, 200, 72, 4}, {1, 100, 81, 4}, {1, 2, 72, 4}, {1, 2, 81, 4}};
    if (input_node_dims[0] != expected_input)
        throw runtime_error("Unsupported/dynamic input shape: require float32 CULane [1,3,320,1600]");
    vector<string> ordered_names;
    for (const auto& shape : expected_outputs) {
        auto it = find(output_node_dims.begin(), output_node_dims.end(), shape);
        if (it == output_node_dims.end())
            throw runtime_error("Unsupported/dynamic output shapes: require CULane "
                "[1,200,72,4], [1,100,81,4], [1,2,72,4], [1,2,81,4]");
        ordered_names.push_back(output_names[it - output_node_dims.begin()]);
    }
    output_names = ordered_names;
    output_node_dims = expected_outputs;
    for (const auto& name : input_names) input_ptrs.push_back(name.c_str());
    for (const auto& name : output_names) output_ptrs.push_back(name.c_str());
    inpHeight = 320;
    inpWidth = 1600;
    dataset = "culane";
    num_row = 72;
    num_col = 81;
    crop_ratio = 0.6f;
    GenerateAnchor();
    cout << "ORT " << OrtGetApiBase()->GetVersionString()
         << " device_requested=" << device << " gpu_id=" << gpu_id
         << " input_shape=[1,3,320,1600] CPU_threads=4\n";
    cout << "Inference settings: " << inference_settings(config_).dump() << '\n';
}

vector<Value> Ultra_Fast_Lane_Detection_v2::run() {
    array<int64_t, 4> shape{1, 3, inpHeight, inpWidth};
    auto memory = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Value::CreateTensor<float>(memory, input_image_.data(),
        input_image_.size(), shape.data(), shape.size());
    // Default Run synchronizes GPU work before returning CPU output tensors.
    auto outputs = ort_session->Run(RunOptions{nullptr}, input_ptrs.data(), &tensor, 1,
                                   output_ptrs.data(), output_ptrs.size());
    for (size_t i = 0; i < outputs.size(); ++i)
        if (outputs[i].GetTensorTypeAndShapeInfo().GetShape() != output_node_dims[i])
            throw runtime_error("Runtime output shape differs from validated model metadata");
    return outputs;
}

void Ultra_Fast_Lane_Detection_v2::verify_profile(bool require_cuda) {
    AllocatorWithDefaultOptions allocator;
    auto filename = ort_session->EndProfilingAllocated(allocator);
    ifstream file(filename.get());
    Json events;
    file >> events;
    size_t cuda_nodes = 0, cpu_nodes = 0, cuda_compute = 0;
    double cuda_us = 0, cpu_us = 0;
    for (const auto& event : events) {
        if (!event.contains("args")) continue;
        const auto& args = event["args"];
        string provider = args.value("provider", "");
        string op = args.value("op_name", "");
        double duration = event.value("dur", 0.0);
        if (provider == "CUDAExecutionProvider") {
            ++cuda_nodes; cuda_us += duration;
            if (op.find("Conv") != string::npos || op.find("Gemm") != string::npos ||
                op.find("MatMul") != string::npos) ++cuda_compute;
        } else if (provider == "CPUExecutionProvider") {
            ++cpu_nodes; cpu_us += duration;
        }
    }
    cout << "Profiling: " << filename.get() << " CUDA_kernel_events=" << cuda_nodes
         << " CUDA_Conv_Gemm_MatMul_events=" << cuda_compute
         << " CPU_kernel_events=" << cpu_nodes << " CUDA_node_us=" << cuda_us
         << " CPU_node_us=" << cpu_us << "\n";
    if (require_cuda && (cuda_nodes == 0 || cuda_compute == 0))
        throw runtime_error("CUDA verification failed: no CUDA major-compute events; CPU fallback rejected");
    if (!require_cuda && cpu_nodes == 0)
        throw runtime_error("CPU verification failed: no CPU kernel events in profiling");
    cout << "Verified EP: " << (require_cuda ? "CUDAExecutionProvider (CPU nodes reported above)"
                                          : "CPUExecutionProvider") << "\n";
}

void Ultra_Fast_Lane_Detection_v2::benchmark(int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) run();
    vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        auto outputs = run();
        samples.push_back(elapsed_ms(start));
    }
    double sum = 0;
    for (double value : samples) sum += value;
    sort(samples.begin(), samples.end());
    auto percentile = [&](double p) {return samples[size_t(ceil(p * samples.size())) - 1];};
    cout << "Benchmark Session::Run (profiling disabled): warmup=" << warmup
         << " iterations=" << iterations << " mean_ms=" << sum / samples.size()
         << " p50_ms=" << percentile(0.50) << " p95_ms=" << percentile(0.95) << "\n";
}

void Ultra_Fast_Lane_Detection_v2::GenerateAnchor()
{
	for (int i = 0; i < num_row; i++)
	{
		if (dataset == "culane")
		{
			row_anchor.push_back(0.42 + i * (1.0 - 0.42) / (num_row - 1));
		}
		else
		{
			row_anchor.push_back((160 + i * (710 - 160) / (num_row - 1)) / 720.0);
		}
	}
	for (int i = 0; i < num_col; i++)
	{
		col_anchor.push_back(0.0 + i * (1.0 - 0.0) / (num_col - 1));
	}
}

void Ultra_Fast_Lane_Detection_v2::normalize_(Mat img)
{
    input_image_ = normalize_pixels(img, inpHeight, inpWidth, config_);
}

float Ultra_Fast_Lane_Detection_v2::SoftMaxFast(const float* src, float* dst, int64_t length)
{
	const float alpha = *std::max_element(src, src + length);
	float denominator{ 0 };

	for (int64_t i = 0; i < length; ++i) {
		dst[i] = std::exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int64_t i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

Mat Ultra_Fast_Lane_Detection_v2::detect(Mat& srcimg)
{
	auto start = Clock::now();
    if (srcimg.empty()) throw runtime_error("Empty input image");
	const int img_h = srcimg.rows;
	const int img_w = srcimg.cols;
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, int(float(this->inpHeight) / this->crop_ratio)), 0, 0, INTER_LINEAR);
    dstimg = enhance_pixels(dstimg, config_);
	this->normalize_(dstimg);
    preprocess_ms = elapsed_ms(start);
    start = Clock::now();
    vector<Value> ort_outputs = run();
    inference_ms = elapsed_ms(start);
    start = Clock::now();

	////pred2coords
	const float* loc_row = ort_outputs[0].GetTensorMutableData<float>();
	const float* loc_col = ort_outputs[1].GetTensorMutableData<float>();
	const float* exist_row = ort_outputs[2].GetTensorMutableData<float>();
	const float* exist_col = ort_outputs[3].GetTensorMutableData<float>();

	vector<int64_t> loc_row_dims = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64_t num_grid_row = loc_row_dims[1];
	int64_t num_cls_row = loc_row_dims[2];
	int64_t num_lane_row = loc_row_dims[3];
	vector<int64_t> loc_col_dims = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	int64_t num_grid_col = loc_col_dims[1];
	int64_t num_cls_col = loc_col_dims[2];
	int64_t num_lane_col = loc_col_dims[3];

	vector<int64_t> exist_row_dims = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	vector<int64_t> exist_col_dims = ort_outputs[3].GetTensorTypeAndShapeInfo().GetShape();

	vector<int64_t> max_indices_row = argmax_1(loc_row, loc_row_dims);
	vector<int64_t> valid_row = argmax_1(exist_row, exist_row_dims);
	vector<int64_t> max_indices_col = argmax_1(loc_col, loc_col_dims);
	vector<int64_t> valid_col = argmax_1(exist_col, exist_col_dims);

    diagnostics = Json::array();
    for (int slot = 0; slot < 4; ++slot) diagnostics.push_back(Json::object());
	vector<vector<Point>> line_list(4);
	for (int64_t i : { 1, 2 })
	{
        const auto count = sum_valid(valid_row, num_cls_row, num_lane_row, i);
        const bool passed = passes_lane_gate(count, num_cls_row, config_.row_min_ratio);
        diagnostics[i] = {{"slot", i}, {"branch", "row"}, {"valid_anchors", count},
            {"total_anchors", num_cls_row}, {"minimum_valid_anchors", int(floor(num_cls_row * config_.row_min_ratio)) + 1},
            {"passed_gate", passed}};
		if (passed)
		{
			for (int64_t k = 0; k < num_cls_row; k++)
			{
				int64_t index = k * num_lane_row + i;
				if (valid_row[index] != 0)
				{
					vector<float> pred_all_list;
					vector<int64_t> all_ind_list;
					for (int64_t all_ind = max(0, int(max_indices_row[index] - 1)); all_ind <= min(num_grid_row - 1, max_indices_row[index] + 1); all_ind++)
					{
						pred_all_list.push_back(loc_row[all_ind * num_cls_row * num_lane_row + index]);
						all_ind_list.push_back(all_ind);
					}
					vector<float> pred_all_list_softmax(pred_all_list.size());
					this->SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
					float out_temp = 0;
					for (size_t l = 0; l < pred_all_list.size(); l++)
					{
						out_temp += pred_all_list_softmax[l] * all_ind_list[l];
					}
					float x = (out_temp + 0.5) / (num_grid_row - 1.0);
					float y = row_anchor[k];
					// End anchors map exactly to width/height; clamp only boundary pixels.
                    if (std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 && x <= 1 && y <= 1)
                        line_list[i].push_back(Point(min(img_w-1, int(x*img_w)), min(img_h-1, int(y*img_h))));
				}
			}
		}
	}

	for (int64_t i : {0, 3})
	{
        const auto count = sum_valid(valid_col, num_cls_col, num_lane_col, i);
        const bool passed = passes_lane_gate(count, num_cls_col, config_.col_min_ratio);
        diagnostics[i] = {{"slot", i}, {"branch", "col"}, {"valid_anchors", count},
            {"total_anchors", num_cls_col}, {"minimum_valid_anchors", int(floor(num_cls_col * config_.col_min_ratio)) + 1},
            {"passed_gate", passed}};
		if (passed)
		{
			for (int64_t k = 0; k < num_cls_col; k++)
			{
				int64_t index = k * num_lane_col + i;
				if (valid_col[index] != 0)
				{
					vector<float> pred_all_list;
					vector<int64_t> all_ind_list;
					for (int64_t all_ind = max(0, int(max_indices_col[index] - 1)); all_ind <= min(num_grid_col - 1, max_indices_col[index] + 1); all_ind++)
					{
						pred_all_list.push_back(loc_col[all_ind * num_cls_col * num_lane_col + index]);
						all_ind_list.push_back(all_ind);
					}
					vector<float> pred_all_list_softmax(pred_all_list.size());
					this->SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
					float out_temp = 0;
					for (size_t l = 0; l < pred_all_list.size(); l++)
					{
						out_temp += pred_all_list_softmax[l] * all_ind_list[l];
					}
					float y = (out_temp + 0.5) / (num_grid_col - 1.0);
					float x = col_anchor[k];
					// End anchors map exactly to width/height; clamp only boundary pixels.
                    if (std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 && x <= 1 && y <= 1)
                        line_list[i].push_back(Point(min(img_w-1, int(x*img_w)), min(img_h-1, int(y*img_h))));
				}
			}
		}
	}
	Mat drawimg = srcimg.clone();
	for (auto& line : line_list)
	{
		for (auto& p : line)
		{
			circle(drawimg, p, 3, Scalar(0, 255, 0), -1);
		}

	}

    lanes.clear();
    for (size_t slot = 0; slot < line_list.size(); ++slot) {
        diagnostics[slot]["exported_points"] = line_list[slot].size();
        if (!line_list[slot].empty()) lanes.push_back(line_list[slot]);
    }
    postprocess_ms = elapsed_ms(start);
	return drawimg;
}


struct Options {
    InferenceConfig inference;
    string model = "weights/ufldv2_culane_res18_320x1600.onnx";
    string image, image_dir, output, output_dir = "outputs", device = "cuda";
    int gpu_id = 0, warmup = 10, iterations = 100;
    bool save_points = false, benchmark = false, skip_existing = false;
};

static void help() {
    cout << "UFLDv2 CULane C++ / ONNX Runtime GPU demo\n"
         << "Usage: ufldv2_gpu --model FILE (--image FILE | --image-dir DIR) [options]\n"
         << "  --output FILE          Single-image output (default outputs/<stem>_result.jpg)\n"
         << "  --output-dir DIR       Batch/default output directory (default outputs)\n"
         << "  --device cuda|cpu      Default cuda; CUDA failures are fatal\n"
         << "  --gpu-id N             Default 0\n"
         << "  --save-points          Save original-image pixel coordinates as JSON\n"
         << "  --skip-existing        Resume completed outputs (requires --save-points)\n"
         << "  --input-color bgr|rgb  Tensor channel order (default bgr, original behavior)\n"
         << "  --enhance none|clahe   Optional luminance CLAHE (default none)\n"
         << "  --row-min-ratio R      Require valid row anchors > R*72 (default 0.5)\n"
         << "  --col-min-ratio R      Require valid col anchors > R*81 (default 0.25)\n"
         << "  --benchmark            Benchmark synchronous Run after profile verification\n"
         << "  --warmup N             Default 10 (non-negative)\n"
         << "  --iterations N         Default 100 (positive)\n"
         << "  --help                 Show this help\n"
         << "Profiling: <output-dir>/profiles; only static CULane float32 exports supported.\n";
}

static int integer(const string& value, const string& option) {
    size_t end = 0;
    int parsed;
    try { parsed = stoi(value, &end); }
    catch (const exception&) { throw runtime_error("Invalid integer for " + option + ": " + value); }
    if (end != value.size() || parsed < 0) throw runtime_error("Invalid integer for " + option + ": " + value);
    return parsed;
}

static double ratio_value(const string& value, const string& option) {
    size_t end = 0;
    double parsed;
    try { parsed = stod(value, &end); }
    catch (const exception&) { throw runtime_error("Invalid ratio for " + option + ": " + value); }
    if (end != value.size() || !std::isfinite(parsed) || parsed < 0 || parsed > 1)
        throw runtime_error("Ratio must be finite and in [0,1] for " + option);
    return parsed;
}

static Options parse(int argc, char** argv) {
    Options o;
    set<string> seen;
    for (int i = 1; i < argc; ++i) {
        string key = argv[i];
        if (!seen.insert(key).second) throw runtime_error("Duplicate option: " + key);
        if (key == "--save-points") { o.save_points = true; continue; }
        if (key == "--skip-existing") { o.skip_existing = true; continue; }
        if (key == "--benchmark") { o.benchmark = true; continue; }
        const set<string> valued{"--model", "--image", "--image-dir", "--output", "--output-dir",
                                "--device", "--gpu-id", "--warmup", "--iterations",
                                "--input-color", "--enhance", "--row-min-ratio", "--col-min-ratio"};
        if (!valued.count(key)) throw runtime_error("Unknown option: " + key);
        if (++i == argc || string(argv[i]).rfind("--", 0) == 0 || string(argv[i]).empty())
            throw runtime_error("Missing value for " + key);
        string value = argv[i];
        if (key == "--model") o.model = value;
        else if (key == "--image") o.image = value;
        else if (key == "--image-dir") o.image_dir = value;
        else if (key == "--output") o.output = value;
        else if (key == "--output-dir") o.output_dir = value;
        else if (key == "--device") o.device = value;
        else if (key == "--gpu-id") o.gpu_id = integer(value, key);
        else if (key == "--warmup") o.warmup = integer(value, key);
        else if (key == "--iterations") o.iterations = integer(value, key);
        else if (key == "--input-color") o.inference.input_color = value;
        else if (key == "--enhance") o.inference.enhance = value;
        else if (key == "--row-min-ratio") o.inference.row_min_ratio = ratio_value(value, key);
        else if (key == "--col-min-ratio") o.inference.col_min_ratio = ratio_value(value, key);
    }
    if (o.image.empty() == o.image_dir.empty()) throw runtime_error("Specify exactly one of --image and --image-dir");
    if (!o.image_dir.empty() && !o.output.empty()) throw runtime_error("--output is single-image only; use --output-dir");
    if (o.device != "cuda" && o.device != "cpu") throw runtime_error("--device must be cuda or cpu");
    if (o.inference.input_color != "bgr" && o.inference.input_color != "rgb")
        throw runtime_error("--input-color must be bgr or rgb");
    if (o.inference.enhance != "none" && o.inference.enhance != "clahe")
        throw runtime_error("--enhance must be none or clahe");
    if (o.iterations == 0) throw runtime_error("--iterations must be positive");
    if (o.skip_existing && !o.save_points) throw runtime_error("--skip-existing requires --save-points to validate existing results");
    return o;
}

static void save_json(const fs::path& destination, const fs::path& source,
                      const Mat& image, const Options& o, const vector<vector<Point>>& lanes,
                      const Json& diagnostics = Json::array()) {
    Json points = Json::array();
    for (const auto& lane : lanes) {
        Json sequence = Json::array();
        for (const auto& point : lane) sequence.push_back({point.x, point.y});
        points.push_back(sequence);
    }
    Json data{{"source_image", source.string()}, {"image_width", image.cols},
              {"image_height", image.rows}, {"model_path", o.model},
              {"device_requested", o.device}, {"lanes", points},
              {"inference_settings", inference_settings(o.inference)}, {"lane_diagnostics", diagnostics}};
    ofstream file(destination);
    file << data.dump(2) << '\n';
    file.close();
    if (!file) throw runtime_error("Cannot write JSON: " + destination.string());
}

int main(int argc, char** argv) {
    Options o;
    try {
        for (int i = 1; i < argc; ++i) if (string(argv[i]) == "--help") { help(); return 0; }
        if (argc == 1) { help(); return 2; }
        o = parse(argc, argv);
        if (!fs::is_regular_file(o.model))
            throw runtime_error("Model file not found: " + o.model +
                "; finish downloading the project UFLDv2 ONNX weights (partial downloads are not models)");
        if (fs::path(o.model).extension() != ".onnx")
            throw runtime_error("Model must be a completed .onnx file, not a partial download");
        vector<fs::path> images;
        if (!o.image.empty()) {
            if (!fs::is_regular_file(o.image)) throw runtime_error("Image file not found: " + o.image);
            images.emplace_back(o.image);
        } else {
            if (!fs::is_directory(o.image_dir)) throw runtime_error("Image directory not found: " + o.image_dir);
            for (const auto& entry : fs::directory_iterator(o.image_dir)) {
                if (!entry.is_regular_file()) continue;
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(tolower(c)); });
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") images.push_back(entry.path());
            }
            sort(images.begin(), images.end());
            if (images.empty()) throw runtime_error("No JPG/JPEG/PNG images in " + o.image_dir);
        }
        // Reserve all output names in advance; retain the source extension in batch stems
        // (road.jpg -> road.jpg_result.jpg), avoiding road.jpg/road.png collisions.
        vector<fs::path> destinations;
        vector<fs::path> pending_images;
        size_t skipped = 0;
        set<fs::path> occupied;
        // Canonicalize inputs once; rescanning every image for each output makes
        // large folders spend quadratic time in filesystem checks before Run.
        set<fs::path> input_paths;
        for (const auto& input : images) input_paths.insert(fs::weakly_canonical(input));
        const auto model_path = fs::weakly_canonical(o.model);
        for (const auto& source : images) {
            fs::path destination = !o.output.empty() ? fs::path(o.output) :
                fs::path(o.output_dir) / ((o.image.empty() ? source.filename().string() : source.stem().string()) + "_result.jpg");
            auto json_path = destination; json_path.replace_extension(".json");
            if (o.skip_existing && fs::is_regular_file(destination) && fs::is_regular_file(json_path)) {
                ifstream saved(json_path);
                const auto metadata = Json::parse(saved);
                if (fs::weakly_canonical(metadata.at("source_image").get<string>()) != fs::weakly_canonical(source) ||
                    fs::weakly_canonical(metadata.at("model_path").get<string>()) != model_path ||
                    metadata.at("device_requested").get<string>() != o.device ||
                    !metadata.at("lanes").is_array() || fs::file_size(destination) == 0)
                    throw runtime_error("Existing output metadata does not match requested input/model/device: " + destination.string());
                const Json saved_settings = metadata.value("inference_settings", inference_settings(InferenceConfig{}));
                if (saved_settings != inference_settings(o.inference))
                    throw runtime_error("Existing output has different preprocessing/postprocessing settings: " + destination.string());
                ++skipped;
                continue;
            }
            auto reserve = [&](const fs::path& p) {
                auto absolute = fs::weakly_canonical(p);
                if (fs::exists(p) || !occupied.insert(absolute).second)
                    throw runtime_error("Refusing to overwrite existing/colliding output: " + p.string());
                if (input_paths.count(absolute)) throw runtime_error("Output collides with input image");
                if (absolute == model_path) throw runtime_error("Output collides with model");
            };
            reserve(destination);
            if (o.save_points) reserve(json_path);
            destinations.push_back(destination);
            pending_images.push_back(source);
        }
        images = std::move(pending_images);
        cout << "Batch: pending=" << images.size() << " skipped=" << skipped << '\n';
        if (images.empty()) {
            cout << "Summary: success=0 failed=0 skipped=" << skipped << '\n';
            return 0;
        }
        Ultra_Fast_Lane_Detection_v2 net(o.model, o.device, o.gpu_id, fs::path(o.output_dir) / "profiles", o.inference);
        size_t succeeded = 0, failed = 0;
        bool verified = false;
        for (size_t i = 0; i < images.size(); ++i) {
            auto total_start = Clock::now();
            bool auditing = false;
            try {
                Mat image = imread(images[i].string(), IMREAD_COLOR);
                if (image.empty()) throw runtime_error("Cannot decode image: " + images[i].string());
                Mat result = net.detect(image);
                // Audit first successful Run before producing any claimed CUDA results.
                auto audit_start = Clock::now();
                if (!verified) {
                    auditing = true;
                    net.verify_profile(o.device == "cuda");
                    auditing = false;
                    verified = true;
                }
                if (o.benchmark) net.benchmark(o.warmup, o.iterations);
                const double audit_benchmark_ms = elapsed_ms(audit_start);
                auto destination = destinations[i];
                if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
                if (!imwrite(destination.string(), result)) throw runtime_error("imwrite failed: " + destination.string());
                if (o.save_points) {
                    auto json_path = destination; json_path.replace_extension(".json");
                    save_json(json_path, images[i], image, o, net.lanes, net.diagnostics);
                }
                cout << "image=" << images[i] << " size=" << image.cols << 'x' << image.rows
                     << " lanes=" << net.lanes.size() << " output=" << destination
                     << " preprocess_ms=" << net.preprocess_ms << " Run_ms=" << net.inference_ms
                     << " postprocess_draw_ms=" << net.postprocess_ms
                     << " total_ms=" << elapsed_ms(total_start) - audit_benchmark_ms << "\n";
                ++succeeded;
            } catch (const Ort::Exception&) {
                // A failed GPU execution is session-fatal; never continue as CPU.
                throw;
            } catch (const exception& e) {
                cerr << "Image failed: " << images[i] << ": " << e.what() << '\n';
                ++failed;
                if (auditing) throw;
            }
        }
        cout << "Summary: success=" << succeeded << " failed=" << failed << " skipped=" << skipped << '\n';
        return failed || !verified ? 1 : 0;
    } catch (const exception& e) {
        cerr << "ERROR model=" << o.model << " device=" << o.device << " gpu_id=" << o.gpu_id
             << ": " << e.what() << '\n';
        return 1;
    }
}
