#define main ufldv2_program_main
#include "../onnxruntime/main.cpp"
#undef main
#include <cassert>

int main() {
    InferenceConfig baseline;
    cv::Mat image(2, 2, CV_8UC3);
    image.row(0).setTo(cv::Scalar(200, 210, 220));
    image.at<cv::Vec3b>(1, 0) = {10, 20, 30};
    image.at<cv::Vec3b>(1, 1) = {40, 50, 60};
    const auto before = image.clone();
    const auto bgr = normalize_pixels(image, 1, 2, baseline);
    auto rgb_config = baseline;
    rgb_config.input_color = "rgb";
    const auto rgb = normalize_pixels(image, 1, 2, rgb_config);
    assert(bgr.size() == 6 && rgb.size() == 6);
    const float means[] = {0.485f, 0.456f, 0.406f};
    const float deviations[] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        for (int x = 0; x < 2; ++x) {
            const auto pixel = image.at<cv::Vec3b>(1, x);
            assert(std::abs(bgr[c * 2 + x] - float((pixel[c] / 255.0 - means[c]) / deviations[c])) < 1e-6);
            assert(std::abs(rgb[c * 2 + x] - float((pixel[2 - c] / 255.0 - means[c]) / deviations[c])) < 1e-6);
        }
    }
    assert(cv::norm(image, before, cv::NORM_INF) == 0);
    cv::Mat gray(16, 16, CV_8UC3, cv::Scalar(180, 180, 180));
    assert(normalize_pixels(gray, 8, 16, baseline) == normalize_pixels(gray, 8, 16, rgb_config));
    auto clahe_config = baseline;
    clahe_config.enhance = "clahe";
    auto enhanced = enhance_pixels(gray, clahe_config);
    assert(enhanced.type() == CV_8UC3 && enhanced.size() == gray.size());
    assert(cv::norm(gray, cv::Mat(16, 16, CV_8UC3, cv::Scalar(180, 180, 180)), cv::NORM_INF) == 0);
    for (int count = 0; count <= 72; ++count)
        assert(passes_lane_gate(count, 72, 0.5) == (count >= 37));
    for (int count = 0; count <= 81; ++count)
        assert(passes_lane_gate(count, 81, 0.25) == (count > 81 / 4));
    assert(!passes_lane_gate(28, 72, 0.4) && passes_lane_gate(29, 72, 0.4));
    assert(!passes_lane_gate(21, 72, 0.3) && passes_lane_gate(22, 72, 0.3));
    assert(!passes_lane_gate(0, 72, 0) && passes_lane_gate(1, 72, 0));
    for (auto value : {"nan", "inf", "-0.1", "1.1", "0.3junk"}) {
        bool rejected = false;
        try { ratio_value(value, "--row-min-ratio"); }
        catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    assert(ratio_value("0.4", "--row-min-ratio") == 0.4);
    assert(inference_settings(baseline) != inference_settings(rgb_config));
    std::cout << "PASS: NCHW/bottom-crop, BGR/RGB, grayscale invariance, immutable sources, CLAHE shape, "
                 "all default gate boundaries, relaxed gates, invalid ratios, settings identity\n";
}
