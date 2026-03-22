package com.yolo.driver.data

import com.google.gson.annotations.SerializedName
import kotlin.math.abs

/**
 * 校准数据模型
 */
data class CalibrationData(
    @SerializedName("timestamp")
    val timestamp: Long = System.currentTimeMillis(),
    
    @SerializedName("duration_setting")
    val durationSetting: String = "3s",
    
    // 基准值 (正常驾驶姿势)
    @SerializedName("base_values")
    val baseValues: BaseValues,
    
    // 动作边界值
    @SerializedName("action_limits")
    val actionLimits: ActionLimits,
    
    // 计算后的阈值
    @SerializedName("calculated_thresholds")
    val calculatedThresholds: CalibrationThresholds
) {
    data class BaseValues(
        @SerializedName("head_vertical")
        val headVertical: Float,      // dy_nose_shoulder 基准
        
        @SerializedName("head_horizontal")
        val headHorizontal: Float,    // dx_nose_shoulder 基准
        
        @SerializedName("posture")
        val posture: Float,           // dy_shoulder 基准
        
        @SerializedName("eye_distance")
        val eyeDistance: Float        // 双眼距离基准 (用于检测侧脸)
    )
    
    data class ActionLimits(
        @SerializedName("head_up")
        val headUp: Float,            // 抬头边界 (dy_nose_shoulder 最小值)
        
        @SerializedName("head_down")
        val headDown: Float,          // 低头边界 (dy_nose_shoulder 最大值)
        
        @SerializedName("head_left")
        val headLeft: Float,          // 左看边界 (dx_nose_shoulder 左偏最大)
        
        @SerializedName("head_right")
        val headRight: Float,         // 右看边界 (dx_nose_shoulder 右偏最大)
        
        @SerializedName("posture_deviation")
        val postureDeviation: Float,  // 姿态偏移边界 (dy_shoulder 最大值)
        
        @SerializedName("head_turned")
        val headTurned: Float         // 侧脸时双眼距离边界
    )
    
    companion object {
        /**
         * 从采集数据计算校准阈值
         * threshold = base + factor * (limit - base)
         */
        fun calculateThresholds(
            baseValues: BaseValues,
            actionLimits: ActionLimits,
            factor: Float = 0.7f
        ): CalibrationThresholds {
            // 抬头阈值: base + 0.7 * (up - base)，值越小 = 头越后仰
            val headUpThreshold = baseValues.headVertical + factor * (actionLimits.headUp - baseValues.headVertical)
            
            // 低头阈值: base + 0.7 * (down - base)，值越大 = 头越下垂
            val headDownThreshold = baseValues.headVertical + factor * (actionLimits.headDown - baseValues.headVertical)
            
            // 头部偏移阈值: 取左右偏移的最大值 * 0.7
            val headOffsetThreshold = maxOf(actionLimits.headLeft, actionLimits.headRight) * factor
            
            // 姿态偏移阈值
            val postureDeviationThreshold = baseValues.posture + factor * (actionLimits.postureDeviation - baseValues.posture)
            
            // 头部扭转阈值: 基准双眼距离 * 0.3 (侧脸时双眼距离约为正面的30%)
            val headTurnedThreshold = if (actionLimits.headTurned > 0) {
                actionLimits.headTurned
            } else {
                baseValues.eyeDistance * 0.3f
            }
            
            return CalibrationThresholds(
                headUp = headUpThreshold,
                headDown = headDownThreshold,
                headOffset = headOffsetThreshold,
                headTurned = headTurnedThreshold,
                postureDeviation = postureDeviationThreshold
            )
        }
    }
}

/**
 * 校准阈值
 */
data class CalibrationThresholds(
    @SerializedName("head_up")
    val headUp: Float,
    
    @SerializedName("head_down")
    val headDown: Float,
    
    @SerializedName("head_offset")
    val headOffset: Float,
    
    @SerializedName("head_turned")
    val headTurned: Float,
    
    @SerializedName("posture_deviation")
    val postureDeviation: Float
)

/**
 * 采集帧数据 (用于校准)
 */
data class CollectedFrame(
    val dyNoseShoulder: Float,    // 头部垂直偏移
    val dxNoseShoulder: Float,    // 头部水平偏移 (带方向)
    val dyShoulder: Float,        // 肩膀高度差
    val eyeDistance: Float,       // 双眼水平距离
    val isValid: Boolean          // 数据是否有效
)
