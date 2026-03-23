package com.yolo.driver.analyzer

import com.yolo.driver.analyzer.KeypointDetector.KeypointIndex
import com.yolo.driver.analyzer.KeypointDetector.KeyPoint
import com.yolo.driver.data.CalibrationData
import com.yolo.driver.data.CalibrationThresholds
import kotlin.math.abs

/**
 * 驾驶员状态分析器
 * 基于关键点检测分析头部姿态和坐姿，判断疲劳程度
 * 
 * 支持用户自定义姿态状态映射：
 * - 逐帧模式：检测到某姿态直接返回对应状态
 * - 滑动窗模式：累积后根据最高优先级状态返回
 */
class StateAnalyzer {
    
    companion object {
        // 默认置信度阈值
        private const val DEFAULT_EYE_CONFIDENCE = 0.5f
        private const val DEFAULT_SHOULDER_CONFIDENCE = 0.5f
        private const val DEFAULT_NOSE_CONFIDENCE = 0.5f
    }
    
    // 可配置的置信度阈值（用于分析）
    private var analysisConfidenceThreshold = DEFAULT_NOSE_CONFIDENCE
    
    // 疲劳状态
    enum class DriverState {
        NORMAL,
        SLIGHTLY_TIRED,
        TIRED;
        
        fun toInt(): Int = when (this) {
            NORMAL -> 0
            SLIGHTLY_TIRED -> 1
            TIRED -> 2
        }
        
        companion object {
            fun fromInt(value: Int): DriverState = when (value) {
                1 -> SLIGHTLY_TIRED
                2 -> TIRED
                else -> NORMAL
            }
        }
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
    
    // 姿态状态映射
    data class PoseStateMapping(
        val headUpDown: DriverState = DriverState.TIRED,               // 抬头/低头 -> 疲劳
        val headLeftRight: DriverState = DriverState.NORMAL,           // 左右摆头 -> 正常
        val postureDeviation: DriverState = DriverState.TIRED          // 姿态偏移 -> 疲劳
    )
    
    // 检测结果
    data class AnalysisResult(
        val driverState: DriverState,
        val headPoses: Set<HeadPose>,
        val frameCount: Int,
        val timestamp: Long = System.currentTimeMillis(),
        val calibrated: Boolean
    )
    
    // 默认阈值 (校准前使用，基于实际校准数据优化)
    private var thresholds = CalibrationThresholds(
        headUp = -180f,
        headDown = -150.84f,
        headOffset = 27.08f,
        headTurned = 16.88f,
        postureDeviation = 38.5f
    )
    
    // 姿态状态映射（逐帧模式和滑动窗模式分别设置）
    private var framePoseMapping = PoseStateMapping()
    private var slidingPoseMapping = PoseStateMapping()
    
    // 当前是否使用滑动窗模式
    private var isSlidingWindowMode = false
    
    // 状态变量
    private var frameCount = 0
    private var calibrated = false
    
    /**
     * 设置逐帧模式姿态状态映射
     */
    fun setFramePoseMapping(mapping: PoseStateMapping) {
        this.framePoseMapping = mapping
    }
    
    /**
     * 设置滑动窗模式姿态状态映射
     */
    fun setSlidingPoseMapping(mapping: PoseStateMapping) {
        this.slidingPoseMapping = mapping
    }
    
    /**
     * 设置当前检测模式
     */
    fun setSlidingWindowMode(enabled: Boolean) {
        this.isSlidingWindowMode = enabled
    }
    
    /**
     * 设置关键点置信度阈值
     * @param drawThreshold 绘制阈值（暂时未使用，由 KeypointDrawer 控制）
     * @param analysisThreshold 分析阈值，低于此值的关键点不参与姿态判断
     */
    fun setKeypointThresholds(drawThreshold: Float, analysisThreshold: Float) {
        this.analysisConfidenceThreshold = analysisThreshold
    }
    
    /**
     * 设置姿态状态映射（已废弃，使用 setAllPoseMappings）
     */
    @Deprecated("Use setAllPoseMappings instead")
    fun setPoseMapping(mapping: PoseStateMapping) {
        // 兼容旧代码，同时设置两个映射
        this.framePoseMapping = mapping
        this.slidingPoseMapping = mapping
    }
    
    /**
     * 设置所有姿态映射（同时设置两种模式）
     */
    fun setAllPoseMappings(frameMapping: PoseStateMapping, slidingMapping: PoseStateMapping) {
        this.framePoseMapping = frameMapping
        this.slidingPoseMapping = slidingMapping
    }
    
    /**
     * 获取当前使用的姿态状态映射
     */
    fun getPoseMapping(): PoseStateMapping = if (isSlidingWindowMode) slidingPoseMapping else framePoseMapping
    
    /**
     * 获取逐帧模式姿态映射
     */
    fun getFramePoseMapping(): PoseStateMapping = framePoseMapping
    
    /**
     * 获取滑动窗模式姿态映射
     */
    fun getSlidingPoseMapping(): PoseStateMapping = slidingPoseMapping
    
    /**
     * 设置校准阈值
     */
    fun setCalibration(calibrationData: CalibrationData) {
        thresholds = calibrationData.calculatedThresholds
        calibrated = true
    }
    
    /**
     * 分析驾驶员状态（逐帧模式，直接使用姿态映射）
     */
    fun analyze(keypoints: List<KeyPoint>?): AnalysisResult {
        frameCount++
        
        // 分析头部姿态
        val headPoses = if (keypoints != null && keypoints.size >= 17) {
            analyzeHeadPose(keypoints)
        } else {
            emptySet()
        }
        
        // 根据姿态映射确定状态（取最高优先级）
        val driverState = determineStateFromPoses(headPoses)
        
        return AnalysisResult(
            driverState = driverState,
            headPoses = headPoses,
            frameCount = frameCount,
            calibrated = calibrated
        )
    }
    
    /**
     * 根据检测到的姿态和用户映射确定状态
     * 取最高优先级状态（疲劳 > 轻度疲劳 > 正常）
     */
    private fun determineStateFromPoses(headPoses: Set<HeadPose>): DriverState {
        // 根据当前模式选择映射
        val poseMapping = if (isSlidingWindowMode) slidingPoseMapping else framePoseMapping
        
        var maxState = DriverState.NORMAL
        
        for (pose in headPoses) {
            val mappedState = when (pose) {
                HeadPose.HEAD_UP, HeadPose.HEAD_DOWN -> poseMapping.headUpDown
                HeadPose.HEAD_OFFSET, HeadPose.HEAD_TURNED -> poseMapping.headLeftRight
                HeadPose.POSTURE_DEVIATION -> poseMapping.postureDeviation
                HeadPose.FACING_FORWARD -> DriverState.NORMAL
            }
            
            // 取优先级最高的状态
            if (mappedState.toInt() > maxState.toInt()) {
                maxState = mappedState
            }
        }
        
        return maxState
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
        
        // 检查关键点置信度（使用统一的可配置阈值）
        if (nose.confidence < analysisConfidenceThreshold ||
            leftShoulder.confidence < analysisConfidenceThreshold ||
            rightShoulder.confidence < analysisConfidenceThreshold ||
            leftEye.confidence < analysisConfidenceThreshold ||
            rightEye.confidence < analysisConfidenceThreshold) {
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
    
    /**
     * 重置状态
     */
    fun reset() {
        frameCount = 0
    }
    
    /**
     * 清除校准
     */
    fun clearCalibration() {
        thresholds = CalibrationThresholds(
            headUp = -180f,
            headDown = -150.84f,
            headOffset = 27.08f,
            headTurned = 16.88f,
            postureDeviation = 38.5f
        )
        calibrated = false
    }
}