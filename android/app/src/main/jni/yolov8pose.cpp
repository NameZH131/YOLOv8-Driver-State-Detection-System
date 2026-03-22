#include "yolov8pose.h"
#include "net.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <android/log.h>

#define LOG_TAG "YOLOv8Pose"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Debug logging macros - only compiled in debug builds
#ifdef DEBUG
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define LOG_DEBUG(...) ((void)0)
#endif

namespace yolo {

YOLOv8Pose::YOLOv8Pose() : net_(nullptr), useGPU_(false) {}

YOLOv8Pose::~YOLOv8Pose() {
    if (net_) {
        delete static_cast<ncnn::Net*>(net_);
        net_ = nullptr;
    }
}

bool YOLOv8Pose::init(const std::string& paramPath, const std::string& binPath, bool useGPU) {
    useGPU_ = useGPU;
    
    auto* net = new ncnn::Net();
    net->opt.use_vulkan_compute = useGPU_;
    net->opt.use_fp16_packed = true;
    net->opt.use_fp16_storage = true;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_int8_storage = false;
    net->opt.use_packing_layout = true;
    
    // Load model
    if (net->load_param(paramPath.c_str()) != 0) {
        delete net;
        return false;
    }
    if (net->load_model(binPath.c_str()) != 0) {
        delete net;
        return false;
    }
    
    net_ = net;
    return true;
}

cv::Mat YOLOv8Pose::preprocess(const cv::Mat& image, float& scale, float& padX, float& padY) {
    int imgW = image.cols;
    int imgH = image.rows;
    
    // Calculate scale to fit target size
    scale = std::min(static_cast<float>(targetSize_) / imgW, 
                     static_cast<float>(targetSize_) / imgH);
    
    int newW = static_cast<int>(imgW * scale);
    int newH = static_cast<int>(imgH * scale);
    
    // Calculate padding
    padX = (targetSize_ - newW) / 2.0f;
    padY = (targetSize_ - newH) / 2.0f;
    
    // Resize and pad
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));
    
    cv::Mat padded(targetSize_, targetSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));
    
    return padded;
}

DetectionResult YOLOv8Pose::detect(const cv::Mat& image, float confThreshold, float iouThreshold) {
    DetectionResult result;
    
    if (!net_) return result;
    
    float scale, padX, padY;
    cv::Mat input = preprocess(image, scale, padX, padY);
    
    // Convert to ncnn Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_RGB2BGR, 
                                          targetSize_, targetSize_);
    
    // Normalize
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean, norm);
    
    // Inference
    auto* net = static_cast<ncnn::Net*>(net_);
    ncnn::Extractor ex = net->create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    
    ncnn::Mat out;
    ex.extract("out0", out);
    
    // Postprocess
    postprocess(&out, out.w, scale, padX, padY, result, confThreshold, iouThreshold);
    
    return result;
}

void YOLOv8Pose::postprocess(const void* output, int outputSize, float scale, float padX, float padY,
                              DetectionResult& result, float confThreshold, float iouThreshold) {
    const ncnn::Mat& out = *static_cast<const ncnn::Mat*>(output);
    
    // NCNN Mat output format: [num_features, num_boxes]
    // out.h = num_features (56), out.w = num_boxes (8400)
    // Data layout: row-major, so data[feature_idx * num_boxes + box_idx]
    // 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3)
    
    int numFeatures = out.h;  // 56
    int numBoxes = out.w;     // 8400
    const float* data = (const float*)out.data;
    
    // Debug: print output shape
    LOG_DEBUG("Output shape: h=%d w=%d c=%d", out.h, out.w, out.c);
    
    // Debug: print feature ranges to understand output format (only in debug builds)
#ifdef DEBUG
    float minVals[10], maxVals[10];
    for (int f = 0; f < 10 && f < numFeatures; f++) {
        minVals[f] = 1e9f;
        maxVals[f] = -1e9f;
        for (int i = 0; i < numBoxes; i++) {
            float v = data[f * numBoxes + i];
            if (v < minVals[f]) minVals[f] = v;
            if (v > maxVals[f]) maxVals[f] = v;
        }
    }
    LOG_DEBUG("Feature ranges (0-9):");
    LOG_DEBUG("  feat0: [%.1f, %.1f]", minVals[0], maxVals[0]);
    LOG_DEBUG("  feat1: [%.1f, %.1f]", minVals[1], maxVals[1]);
    LOG_DEBUG("  feat2: [%.1f, %.1f]", minVals[2], maxVals[2]);
    LOG_DEBUG("  feat3: [%.1f, %.1f]", minVals[3], maxVals[3]);
    LOG_DEBUG("  feat4 (conf): [%.3f, %.3f]", minVals[4], maxVals[4]);
    LOG_DEBUG("  feat5 (kp0_x): [%.1f, %.1f]", minVals[5], maxVals[5]);
    LOG_DEBUG("  feat6 (kp0_y): [%.1f, %.1f]", minVals[6], maxVals[6]);
    LOG_DEBUG("  feat7 (kp0_c): [%.3f, %.3f]", minVals[7], maxVals[7]);
#endif
    
    // Find top 5 confidences for debugging (no sigmoid needed, ONNX output already 0-1)
#ifdef DEBUG
    float topConfs[5] = {0};
    int topIdxs[5] = {-1};
    for (int i = 0; i < numBoxes; i++) {
        float conf = data[4 * numBoxes + i];  // Already in 0-1 range
        for (int j = 0; j < 5; j++) {
            if (conf > topConfs[j]) {
                for (int k = 4; k > j; k--) {
                    topConfs[k] = topConfs[k-1];
                    topIdxs[k] = topIdxs[k-1];
                }
                topConfs[j] = conf;
                topIdxs[j] = i;
                break;
            }
        }
    }
    LOG_DEBUG("Top 5 confs: %.3f %.3f %.3f %.3f %.3f",
             topConfs[0], topConfs[1], topConfs[2], topConfs[3], topConfs[4]);
#endif
    
    // Find best detection (no sigmoid needed)
    float bestConf = 0.f;
    int bestIdx = -1;
    
    for (int i = 0; i < numBoxes; i++) {
        // Confidence is at feature index 4, box index i (already 0-1)
        float conf = data[4 * numBoxes + i];
        
        if (conf > bestConf && conf > confThreshold) {
            bestConf = conf;
            bestIdx = i;
        }
    }
    
        if (bestIdx < 0) {
            LOG_DEBUG("No detection above threshold %.2f", confThreshold);
            return;
        }
    
        LOG_DEBUG("Best detection: idx=%d conf=%.3f", bestIdx, bestConf);    
    // Extract bbox coordinates from model output
    // Model outputs: [cx, cy, w, h] (center x, center y, width, height)
    // These are absolute coordinates in 640x640 input space
    float cx = data[0 * numBoxes + bestIdx];
    float cy = data[1 * numBoxes + bestIdx];
    float w  = data[2 * numBoxes + bestIdx];
    float h  = data[3 * numBoxes + bestIdx];
    
    // Convert xywh to x1y1x2y2
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = cx + w / 2.0f;
    float y2 = cy + h / 2.0f;
    
    // Debug: print raw bbox values
    LOG_DEBUG("Raw bbox: cx=%.1f cy=%.1f w=%.1f h=%.1f -> x1=%.1f y1=%.1f x2=%.1f y2=%.1f", 
             cx, cy, w, h, x1, y1, x2, y2);
    
    // Map back to original image coordinates
    result.box[0] = (x1 - padX) / scale;
    result.box[1] = (y1 - padY) / scale;
    result.box[2] = (x2 - padX) / scale;
    result.box[3] = (y2 - padY) / scale;
    result.confidence = bestConf;
    
    // Extract keypoints (17 points, each with x, y, conf)
    result.keypoints.resize(17);
    for (int k = 0; k < 17; k++) {
        // Keypoint k starts at feature index 5 + k*3
        float kx = data[(5 + k * 3) * numBoxes + bestIdx];
        float ky = data[(5 + k * 3 + 1) * numBoxes + bestIdx];
        float kc = data[(5 + k * 3 + 2) * numBoxes + bestIdx];  // Already in 0-1 range
        
        result.keypoints[k].x = (kx - padX) / scale;
        result.keypoints[k].y = (ky - padY) / scale;
        result.keypoints[k].confidence = kc;
        
        // Debug: print first few keypoints
        if (k < 3) {
            LOG_DEBUG("KP%d raw: (%.1f, %.1f) -> mapped: (%.1f, %.1f) conf=%.2f", 
                     k, kx, ky, result.keypoints[k].x, result.keypoints[k].y, kc);
        }
    }
}

} // namespace yolo
