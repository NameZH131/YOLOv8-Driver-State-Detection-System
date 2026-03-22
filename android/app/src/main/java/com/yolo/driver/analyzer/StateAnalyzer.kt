package com.yolo.driver.analyzer

import com.yolo.driver.analyzer.KeypointDetector.KeypointIndex
import com.yolo.driver.analyzer.KeypointDetector.KeyPoint
import com.yolo.driver.data.CalibrationData
import com.yolo.driver.data.CalibrationThresholds
import kotlin.math.abs

/**
 * 驾驶员状态分析器
 * 基于关键点检测分析头部姿态和坐姿，判断疲劳程度
 */
class StateAnalyzer {
    
    companion object {
        // 疲劳状态阈值 (直接使用最终值，避免混淆)
        private const val TIRED_THRESHOLD = 1.44f           // 疲劳状态阈值 (1.2 * 1.2)
        private const val SLIGHTLY_TIRED_THRESHOLD = 0.96f  // 轻度疲劳阈值 (1.2 * 0.8)
        
        // 疲劳评分累积系数
        private const val HEAD_POSE_SCORE = 1.2f            // 抬头/低头累积系数
        private const val POSTURE_DEVIATION_SCORE = 2.0f    // 坐姿倾斜累积系数
        private const val UNKNOWN_STATE_SCORE = 0.5f        // 未知状态累积系数
        
        // 置信度阈值
        private const val DEFAULT_EYE_CONFIDENCE = 0.5f
        private const val DEFAULT_SHOULDER_CONFIDENCE = 0.5f
        private const val DEFAULT_NOSE_CONFIDENCE = 0.5f
    }
    
    // 疲劳状态
    enum class DriverState {
        NORMAL,
        SLIGHTLY_TIRED,
        TIRED
    }
    
    // 头部姿态类型
    enum class HeadPose {
        FACING_FORWARD,
        HEAD_UP,
        HEAD_DOWN,
        HEAD_OFFSET,      // 头部左右偏移
        HEAD_TURNED,      // 头部扭转 (侧脸)
        POSTURE_DEVIATION // 坐姿倾斜
    }
    
    // 检测结果
    data class AnalysisResult(
        val driverState: DriverState,
        val headPoses: Set<HeadPose>,
        val tiredFrameCount: Float,
        val frameCount: Int,
        val timestamp: Long = System.currentTimeMillis(),
        val calibrated: Boolean
    )
    
    // 置信度阈值
    private val eyeConfidenceThreshold = DEFAULT_EYE_CONFIDENCE
    private val shoulderConfidenceThreshold = DEFAULT_SHOULDER_CONFIDENCE
    private val noseConfidenceThreshold = DEFAULT_NOSE_CONFIDENCE
    
    // 默认阈值 (校准前使用)
    private var thresholds = CalibrationThresholds(
        headUp = -250f,
        headDown = -200f,
        headOffset = 38f,
        headTurned = 20f,
        postureDeviation = 33f
    )
    
    // 状态变量
    private var tiredFrameCount = 0f
    private var frameCount = 0
    private var driverState = DriverState.NORMAL
    private var calibrated = false
    
    /**
     * 设置校准阈值
     */
    fun setCalibration(calibrationData: CalibrationData) {
        thresholds = calibrationData.calculatedThresholds
        calibrated = true
    }
    
    /**
     * 分析驾驶员状态
     */
    fun analyze(keypoints: List<KeyPoint>?): AnalysisResult {
        frameCount++
        
        if (keypoints == null || keypoints.size < 17) {
            tiredFrameCount += UNKNOWN_STATE_SCORE
            return buildResult(emptySet())
        }
        
        // 分析头部姿态
        val headPoses = analyzeHeadPose(keypoints)
        
        // 疲劳评分累积
        tiredFrameCount = 0f
        
        if (headPoses.contains(HeadPose.HEAD_DOWN) || headPoses.contains(HeadPose.HEAD_UP)) {
            tiredFrameCount += HEAD_POSE_SCORE
        }
        if (headPoses.contains(HeadPose.POSTURE_DEVIATION)) {
            tiredFrameCount += POSTURE_DEVIATION_SCORE
        }
        if (headPoses.isEmpty() || 
            (headPoses.size == 1 && headPoses.contains(HeadPose.FACING_FORWARD))) {
            // 正常状态，重置计数
            tiredFrameCount = 0f
        }
        
        // 判断疲劳状态
        driverState = when {
            tiredFrameCount >= TIRED_THRESHOLD -> DriverState.TIRED
            tiredFrameCount > SLIGHTLY_TIRED_THRESHOLD -> DriverState.SLIGHTLY_TIRED
            else -> DriverState.NORMAL
        }
        
        return buildResult(headPoses)
    }
    
    /**
     * 分析头部姿态
     */
    private fun analyzeHeadPose(keypoints: List<KeyPoint>): Set<HeadPose> {
        val poses = mutableSetOf<HeadPose>()
        
        val nose = keypoints[KeypointIndex.NOSE]
        val leftEye = keypoints[KeypointIndex.LEFT_EYE]
        val rightEye = keypoints[KeypointIndex.RIGHT_EYE]
        val leftShoulder = keypoints[KeypointIndex.LEFT_SHOULDER]
        val rightShoulder = keypoints[KeypointIndex.RIGHT_SHOULDER]
        
        // 检查关键点置信度
        if (nose.confidence < noseConfidenceThreshold ||
            leftShoulder.confidence < shoulderConfidenceThreshold ||
            rightShoulder.confidence < shoulderConfidenceThreshold ||
            leftEye.confidence < eyeConfidenceThreshold ||
            rightEye.confidence < eyeConfidenceThreshold) {
            return poses
        }
        
        // 计算肩膀中点
        val shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2
        val shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2
        
        // 计算头部垂直偏移 (鼻子相对于肩膀中点)
        val dyNoseShoulder = nose.y - shoulderMidY
        
        // 计算头部水平偏移
        val dxNoseShoulder = abs(nose.x - shoulderMidX)
        
        // 计算肩膀高度差
        val dyShoulder = abs(leftShoulder.y - rightShoulder.y)
        
        // 计算两眼水平距离 (用于检测侧脸)
        val eyeDistance = abs(leftEye.x - rightEye.x)
        
        // 判断抬头/低头
        when {
            dyNoseShoulder < thresholds.headUp -> {
                poses.add(HeadPose.HEAD_UP)
            }
            dyNoseShoulder > thresholds.headDown -> {
                poses.add(HeadPose.HEAD_DOWN)
            }
            else -> {
                poses.add(HeadPose.FACING_FORWARD)
            }
        }
        
        // 判断头部左右偏移
        if (dxNoseShoulder > thresholds.headOffset) {
            poses.add(HeadPose.HEAD_OFFSET)
        }
        
        // 判断头部扭转 (侧脸)
        if (eyeDistance < thresholds.headTurned) {
            poses.add(HeadPose.HEAD_TURNED)
        }
        
        // 判断坐姿倾斜
        if (dyShoulder > thresholds.postureDeviation) {
            poses.add(HeadPose.POSTURE_DEVIATION)
        }
        
        return poses
    }
    
    private fun buildResult(headPoses: Set<HeadPose>): AnalysisResult {
        return AnalysisResult(
            driverState = driverState,
            headPoses = headPoses,
            tiredFrameCount = tiredFrameCount,
            frameCount = frameCount,
            calibrated = calibrated
        )
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        tiredFrameCount = 0f
        frameCount = 0
        driverState = DriverState.NORMAL
    }
    
    /**
     * 清除校准
     */
    fun clearCalibration() {
        thresholds = CalibrationThresholds(
            headUp = -250f,
            headDown = -200f,
            headOffset = 38f,
            headTurned = 20f,
            postureDeviation = 33f
        )
        calibrated = false
    }
}
