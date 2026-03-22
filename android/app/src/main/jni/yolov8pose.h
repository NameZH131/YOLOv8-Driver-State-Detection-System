#ifndef YOLOV8POSE_H
#define YOLOV8POSE_H

#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace yolo {

struct KeyPoint {
    float x;
    float y;
    float confidence;
};

struct DetectionResult {
    std::vector<KeyPoint> keypoints;  // 17 keypoints
    float box[4];  // x1, y1, x2, y2
    float confidence;
};

class YOLOv8Pose {
public:
    YOLOv8Pose();
    ~YOLOv8Pose();
    
    bool init(const std::string& paramPath, const std::string& binPath, bool useGPU = true);
    DetectionResult detect(const cv::Mat& image, float confThreshold = 0.6f, float iouThreshold = 0.45f);
    
private:
    void* net_;  // ncnn::Net*
    bool useGPU_;
    int targetSize_ = 640;
    
    cv::Mat preprocess(const cv::Mat& image, float& scale, float& padX, float& padY);
    void postprocess(const void* output, int outputSize, float scale, float padX, float padY, 
                     DetectionResult& result, float confThreshold, float iouThreshold);
};

} // namespace yolo

#endif
