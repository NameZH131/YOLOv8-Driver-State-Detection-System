package com.yolo.driver.analyzer

import android.util.Log

/**
 * YOLOv8-Pose 关键点检测器 JNI 接口
 * 使用引用计数管理的单例模式
 */
class KeypointDetector private constructor() {
    
    companion object {
        private const val TAG = "KeypointDetector"
        
        init {
            System.loadLibrary("yolov8pose")
        }
        
        @Volatile
        private var instance: KeypointDetector? = null
        private var refCount = 0
        private val lock = Any()
        
        /**
         * 获取单例实例（引用计数 +1）
         */
        fun getInstance(): KeypointDetector {
            synchronized(lock) {
                val det = instance ?: KeypointDetector().also { instance = it }
                refCount++
                return det
            }
        }
        
        /**
         * 释放实例（引用计数 -1，计数为 0 时真正释放）
         * 应在 Activity/Fragment 销毁时调用
         */
        fun releaseInstance() {
            synchronized(lock) {
                refCount--
                if (refCount <= 0) {
                    instance?.apply {
                        nativeRelease()
                        isInitialized = false  // 重置初始化标志
                    }
                    instance = null
                    refCount = 0
                    Log.i(TAG, "Detector instance released")
                }
            }
        }
        
        /**
         * 强制释放（用于应用退出时）
         */
        fun forceRelease() {
            synchronized(lock) {
                instance?.apply {
                    nativeRelease()
                    isInitialized = false  // 重置初始化标志
                }
                instance = null
                refCount = 0
                Log.i(TAG, "Detector force released")
            }
        }
    }
    
    // YOLOv8-Pose 17 关键点索引
    object KeypointIndex {
        const val NOSE = 0
        const val LEFT_EYE = 1
        const val RIGHT_EYE = 2
        const val LEFT_EAR = 3
        const val RIGHT_EAR = 4
        const val LEFT_SHOULDER = 5
        const val RIGHT_SHOULDER = 6
        const val LEFT_ELBOW = 7
        const val RIGHT_ELBOW = 8
        const val LEFT_WRIST = 9
        const val RIGHT_WRIST = 10
        const val LEFT_HIP = 11
        const val RIGHT_HIP = 12
        const val LEFT_KNEE = 13
        const val RIGHT_KNEE = 14
        const val LEFT_ANKLE = 15
        const val RIGHT_ANKLE = 16
    }
    
    /**
     * 关键点数据
     */
    data class KeyPoint(
        val x: Float,
        val y: Float,
        val confidence: Float
    )
    
    /**
     * 检测失败原因
     */
    enum class DetectError {
        NOT_INITIALIZED,    // 检测器未初始化
        NO_PERSON,          // 未检测到人物
        LOW_CONFIDENCE,     // 置信度过低
        INVALID_DATA,       // 数据无效
        NATIVE_ERROR        // Native 层错误
    }
    
    /**
     * 检测结果（Sealed Class 支持成功/失败状态）
     */
    sealed class DetectResult {
        data class Success(
            val keypoints: List<KeyPoint>,
            val bbox: FloatArray,  // x1, y1, x2, y2
            val confidence: Float
        ) : DetectResult()
        
        data class Failed(
            val error: DetectError,
            val message: String
        ) : DetectResult()
    }
    
    /**
     * 检测结果（旧版兼容）
     */
    data class DetectionResult(
        val keypoints: List<KeyPoint>,
        val bbox: FloatArray,  // x1, y1, x2, y2
        val confidence: Float
    )
    
    private var isInitialized = false
    
    /**
     * 初始化检测器
     * @param paramPath NCNN param 文件路径
     * @param binPath NCNN bin 文件路径
     * @param useGPU 是否使用 GPU (Vulkan)
     */
    fun init(paramPath: String, binPath: String, useGPU: Boolean = true): Boolean {
        if (isInitialized) return true  // 已初始化
        
        return try {
            isInitialized = nativeInit(paramPath, binPath, useGPU)
            Log.i(TAG, "Detector init: $isInitialized")
            isInitialized
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init detector", e)
            isInitialized = false
            false
        }
    }
    
    /**
     * 检测图像中的关键点
     * @param imageData NV21 格式的图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param confThreshold 置信度阈值
     * @param iouThreshold IOU 阈值
     * @return 检测结果，失败返回 null
     */
    fun detect(
        imageData: ByteArray,
        width: Int,
        height: Int,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f
    ): DetectionResult? {
        val result = detectWithResult(imageData, width, height, confThreshold, iouThreshold)
        return when (result) {
            is DetectResult.Success -> DetectionResult(
                keypoints = result.keypoints,
                bbox = result.bbox,
                confidence = result.confidence
            )
            is DetectResult.Failed -> null
        }
    }
    
    /**
     * 检测图像中的关键点（带详细错误信息）
     * @param imageData NV21 格式的图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param confThreshold 置信度阈值
     * @param iouThreshold IOU 阈值
     * @return DetectResult (Success 或 Failed)
     */
    fun detectWithResult(
        imageData: ByteArray,
        width: Int,
        height: Int,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f
    ): DetectResult {
        if (!isInitialized) {
            return DetectResult.Failed(DetectError.NOT_INITIALIZED, "检测器未初始化")
        }
        
        return try {
            val output = nativeDetect(imageData, width, height, confThreshold, iouThreshold)
                ?: return DetectResult.Failed(DetectError.NO_PERSON, "未检测到人物")
            
            // 边界检查
            if (output.size < 56) {
                Log.e(TAG, "Invalid output size: ${output.size}, expected >= 56")
                return DetectResult.Failed(DetectError.INVALID_DATA, "检测数据无效")
            }
            
            val confidence = output[55]
            if (confidence < confThreshold) {
                return DetectResult.Failed(DetectError.LOW_CONFIDENCE, "检测置信度过低: ${"%.2f".format(confidence)}")
            }
            
            // 解析输出数据 (56 floats)
            val keypoints = mutableListOf<KeyPoint>()
            for (i in 0 until 17) {
                keypoints.add(KeyPoint(
                    x = output[i * 3],
                    y = output[i * 3 + 1],
                    confidence = output[i * 3 + 2]
                ))
            }
            
            DetectResult.Success(
                keypoints = keypoints,
                bbox = floatArrayOf(output[51], output[52], output[53], output[54]),
                confidence = confidence
            )
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            DetectResult.Failed(DetectError.NATIVE_ERROR, "检测异常: ${e.message}")
        }
    }
    
    /**
     * 检查是否已初始化
     */
    fun isInitialized(): Boolean = isInitialized
    
    // Native methods
    private external fun nativeInit(paramPath: String, binPath: String, useGPU: Boolean): Boolean
    private external fun nativeDetect(
        imageData: ByteArray,
        width: Int,
        height: Int,
        confThreshold: Float,
        iouThreshold: Float
    ): FloatArray?
    private external fun nativeRelease()
}