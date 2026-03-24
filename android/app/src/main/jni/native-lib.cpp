#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include <vector>
#include <net.h>  // ncnn for Vulkan detection
#include <gpu.h>  // ncnn GPU functions
#include "yolov8pose.h"

#define LOG_TAG "YOLOv8Pose"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Thread-safe detector management
static std::mutex g_detector_mutex;
static yolo::YOLOv8Pose* g_detector = nullptr;
static bool g_gpu_enabled = false;  // 实际使用的 GPU 状态

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_yolo_driver_analyzer_KeypointDetector_nativeInit(
        JNIEnv* env,
        jobject thiz,
        jstring paramPath,
        jstring binPath,
        jboolean useGPU) {
    
    std::lock_guard<std::mutex> lock(g_detector_mutex);
    
    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
    }
    
    const char* param = env->GetStringUTFChars(paramPath, nullptr);
    const char* bin = env->GetStringUTFChars(binPath, nullptr);
    
    g_detector = new yolo::YOLOv8Pose();
    bool success = g_detector->init(param, bin, useGPU);
    
    // 如果 GPU 初始化失败，尝试 CPU 降级
    if (!success && useGPU) {
        LOGI("GPU init failed, falling back to CPU");
        success = g_detector->init(param, bin, false);
        g_gpu_enabled = false;
    } else {
        g_gpu_enabled = success && useGPU;
    }
    
    env->ReleaseStringUTFChars(paramPath, param);
    env->ReleaseStringUTFChars(binPath, bin);
    
    LOGI("YOLOv8-Pose init: %s, GPU: %s (requested: %s)", 
         success ? "success" : "failed", 
         g_gpu_enabled ? "yes" : "no",
         useGPU ? "yes" : "no");
    return success;
}

JNIEXPORT jboolean JNICALL
Java_com_yolo_driver_analyzer_KeypointDetector_nativeCheckVulkanSupport(
        JNIEnv* env,
        jobject thiz) {
    // 检测 Vulkan 是否可用
    // 通过创建一个临时 Net 来测试
    ncnn::Net testNet;
    testNet.opt.use_vulkan_compute = true;
    
    // 尝试获取 Vulkan 设备
    bool hasVulkan = (ncnn::get_gpu_count() > 0);
    
    LOGI("Vulkan support check: %s (GPU count: %d)", 
         hasVulkan ? "available" : "not available",
         ncnn::get_gpu_count());
    
    return hasVulkan ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_yolo_driver_analyzer_KeypointDetector_nativeIsGPUEnabled(
        JNIEnv* env,
        jobject thiz) {
    std::lock_guard<std::mutex> lock(g_detector_mutex);
    return g_gpu_enabled ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloatArray JNICALL
Java_com_yolo_driver_analyzer_KeypointDetector_nativeDetect(
        JNIEnv* env,
        jobject thiz,
        jbyteArray imageData,
        jint width,
        jint height,
        jfloat confThreshold,
        jfloat iouThreshold) {
    
    std::lock_guard<std::mutex> lock(g_detector_mutex);
    
    if (!g_detector) {
        LOGE("Detector not initialized");
        return nullptr;
    }
    
    // Get image data and copy to local buffer for safety
    jbyte* data = env->GetByteArrayElements(imageData, nullptr);
    if (!data) {
        LOGE("Failed to get image data");
        return nullptr;
    }
    
    // Calculate NV21 buffer size
    size_t nv21Size = static_cast<size_t>(width * height * 3 / 2);
    std::vector<uint8_t> buffer(data, data + nv21Size);
    
    // Release JNI reference immediately after copying
    env->ReleaseByteArrayElements(imageData, data, JNI_ABORT);
    
    // Create cv::Mat from local buffer (NV21 -> RGB)
    cv::Mat nv21(height + height/2, width, CV_8UC1, buffer.data());
    cv::Mat rgb;
    cv::cvtColor(nv21, rgb, cv::COLOR_YUV2RGB_NV21);
    
    // Detect
    yolo::DetectionResult result = g_detector->detect(rgb, confThreshold, iouThreshold);
    
    if (result.keypoints.empty()) {
        return nullptr;
    }
    
    // Output format: [17 keypoints * 3 (x, y, conf)] + [4 bbox] + [1 conf] = 56 floats
    jfloatArray output = env->NewFloatArray(56);
    if (!output) {
        LOGE("Failed to create output array");
        return nullptr;
    }
    
    jfloat* outputData = env->GetFloatArrayElements(output, nullptr);
    if (!outputData) {
        LOGE("Failed to get output array elements");
        return nullptr;
    }
    
    // Keypoints
    for (int i = 0; i < 17; i++) {
        outputData[i * 3] = result.keypoints[i].x;
        outputData[i * 3 + 1] = result.keypoints[i].y;
        outputData[i * 3 + 2] = result.keypoints[i].confidence;
    }
    
    // Bbox
    outputData[51] = result.box[0];
    outputData[52] = result.box[1];
    outputData[53] = result.box[2];
    outputData[54] = result.box[3];
    outputData[55] = result.confidence;
    
    env->ReleaseFloatArrayElements(output, outputData, 0);
    
    return output;
}

JNIEXPORT void JNICALL
Java_com_yolo_driver_analyzer_KeypointDetector_nativeRelease(
        JNIEnv* env,
        jobject thiz) {
    
    std::lock_guard<std::mutex> lock(g_detector_mutex);
    
    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
    }
    LOGI("YOLOv8-Pose released");
}

} // extern "C"
