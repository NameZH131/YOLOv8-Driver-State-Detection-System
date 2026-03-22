package com.yolo.driver.analyzer

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.yolo.driver.data.CalibrationData
import com.yolo.driver.data.CalibrationThresholds
import com.yolo.driver.data.CollectedFrame
import com.yolo.driver.analyzer.KeypointDetector.KeypointIndex
import com.yolo.driver.analyzer.KeypointDetector.KeyPoint
import java.io.File
import kotlin.math.abs

/**
 * 校准管理器
 * 管理校准状态机、数据采集和阈值计算
 */
class CalibrationManager(private val context: Context) {
    
    companion object {
        private const val TAG = "CalibrationManager"
        private const val CALIBRATION_FILE = "calibration.json"
    }
    
    // 校准状态
    enum class State {
        IDLE,               // 空闲
        BASE_COLLECT,       // 基准采集
        HEAD_UP,            // 抬头采集
        HEAD_DOWN,          // 低头采集
        HEAD_LEFT,          // 左看采集
        HEAD_RIGHT,         // 右看采集
        POSTURE_DEVIATION,  // 姿态偏移采集
        DONE                // 完成
    }
    
    // 校准时长选项
    enum class Duration(val seconds: Int, val displayName: String, val maxFrames: Int) {
        FAST(2, "快速 (2秒)", 60),
        NORMAL(3, "标准 (3秒)", 90),
        ACCURATE(5, "精确 (5秒)", 150)
    }
    
    // 采集数据结构
    private data class CollectionData(
        val frames: MutableList<CollectedFrame> = mutableListOf()
    )
    
    // 状态变量
    private var currentState = State.IDLE
    private var selectedDuration = Duration.NORMAL
    private val collectionData = CollectionData()
    
    // 采集到的数据
    private var baseValues: CalibrationData.BaseValues? = null
    private var actionLimits = mutableMapOf<State, Float>()
    
    // 状态转换顺序
    private val stateOrder = listOf(
        State.BASE_COLLECT,
        State.HEAD_UP,
        State.HEAD_DOWN,
        State.HEAD_LEFT,
        State.HEAD_RIGHT,
        State.POSTURE_DEVIATION
    )
    
    // 置信度阈值
    private val confidenceThreshold = 0.5f
    
    /**
     * 获取当前状态
     */
    fun getState(): State = currentState
    
    /**
     * 获取选定的时长
     */
    fun getSelectedDuration(): Duration = selectedDuration
    
    /**
     * 设置时长
     */
    fun setDuration(duration: Duration) {
        selectedDuration = duration
    }
    
    /**
     * 获取状态显示名称
     */
    fun getStateDisplayName(state: State): String {
        return when (state) {
            State.IDLE -> "等待开始"
            State.BASE_COLLECT -> "保持正常驾驶姿势"
            State.HEAD_UP -> "请抬头"
            State.HEAD_DOWN -> "请低头"
            State.HEAD_LEFT -> "请向左看"
            State.HEAD_RIGHT -> "请向右看"
            State.POSTURE_DEVIATION -> "请左右倾斜身体"
            State.DONE -> "校准完成"
        }
    }
    
    /**
     * 开始校准
     */
    fun startCalibration(): Boolean {
        if (currentState != State.IDLE) {
            Log.w(TAG, "Calibration already in progress")
            return false
        }
        
        currentState = State.BASE_COLLECT
        collectionData.frames.clear()
        actionLimits.clear()
        baseValues = null
        
        Log.i(TAG, "Calibration started, duration: ${selectedDuration.seconds}s")
        return true
    }
    
    /**
     * 处理帧数据
     * @return 是否完成当前状态的采集
     */
    fun processFrame(keypoints: List<KeyPoint>?): Boolean {
        if (currentState == State.IDLE || currentState == State.DONE) {
            return false
        }
        
        // 检查帧数限制，防止内存无限增长
        if (collectionData.frames.size >= selectedDuration.maxFrames) {
            return false
        }
        
        val frame = extractFrameData(keypoints)
        if (frame.isValid) {
            collectionData.frames.add(frame)
        }
        
        return false  // 由外部计时器控制状态转换
    }
    
    /**
     * 提取帧数据
     */
    private fun extractFrameData(keypoints: List<KeyPoint>?): CollectedFrame {
        if (keypoints == null || keypoints.size < 17) {
            return CollectedFrame(0f, 0f, 0f, 0f, false)
        }
        
        val nose = keypoints[KeypointIndex.NOSE]
        val leftEye = keypoints[KeypointIndex.LEFT_EYE]
        val rightEye = keypoints[KeypointIndex.RIGHT_EYE]
        val leftShoulder = keypoints[KeypointIndex.LEFT_SHOULDER]
        val rightShoulder = keypoints[KeypointIndex.RIGHT_SHOULDER]
        
        // 检查置信度
        if (nose.confidence < confidenceThreshold ||
            leftShoulder.confidence < confidenceThreshold ||
            rightShoulder.confidence < confidenceThreshold) {
            return CollectedFrame(0f, 0f, 0f, 0f, false)
        }
        
        // 计算各项指标
        val shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2
        val shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2
        
        val dyNoseShoulder = nose.y - shoulderMidY
        val dxNoseShoulder = nose.x - shoulderMidX  // 带方向
        val dyShoulder = abs(leftShoulder.y - rightShoulder.y)
        
        // 计算双眼水平距离 (用于检测侧脸)
        val eyeDistance = if (leftEye.confidence >= confidenceThreshold && 
                              rightEye.confidence >= confidenceThreshold) {
            abs(leftEye.x - rightEye.x)
        } else {
            0f
        }
        
        return CollectedFrame(
            dyNoseShoulder = dyNoseShoulder,
            dxNoseShoulder = dxNoseShoulder,
            dyShoulder = dyShoulder,
            eyeDistance = eyeDistance,
            isValid = true
        )
    }
    
    /**
     * 完成当前状态的采集，进入下一状态
     */
    fun completeCurrentState(): State {
        if (currentState == State.IDLE || currentState == State.DONE) {
            return currentState
        }
        
        // 处理采集的数据
        processCollectedData()
        
        // 清空采集缓冲
        collectionData.frames.clear()
        
        // 进入下一状态
        val currentIndex = stateOrder.indexOf(currentState)
        if (currentIndex < stateOrder.size - 1) {
            currentState = stateOrder[currentIndex + 1]
        } else {
            // 所有采集完成，计算最终阈值
            finalizeCalibration()
            currentState = State.DONE
        }
        
        return currentState
    }
    
    /**
     * 处理采集的数据
     */
    private fun processCollectedData() {
        val validFrames = collectionData.frames.filter { it.isValid }
        if (validFrames.isEmpty()) {
            Log.w(TAG, "No valid frames for state: $currentState")
            return
        }
        
        when (currentState) {
            State.BASE_COLLECT -> {
                // 基准值取中位数
                val sortedDy = validFrames.map { it.dyNoseShoulder }.sorted()
                val sortedDx = validFrames.map { abs(it.dxNoseShoulder) }.sorted()
                val sortedPosture = validFrames.map { it.dyShoulder }.sorted()
                val sortedEyeDistance = validFrames.map { it.eyeDistance }.filter { it > 0 }.sorted()
                
                baseValues = CalibrationData.BaseValues(
                    headVertical = sortedDy[sortedDy.size / 2],
                    headHorizontal = sortedDx[sortedDx.size / 2],
                    posture = sortedPosture[sortedPosture.size / 2],
                    eyeDistance = if (sortedEyeDistance.isNotEmpty()) sortedEyeDistance[sortedEyeDistance.size / 2] else 60f
                )
                Log.i(TAG, "Base values: $baseValues")
            }
            
            State.HEAD_UP -> {
                // 抬头: dy_nose_shoulder 最小值
                actionLimits[State.HEAD_UP] = validFrames.minOf { it.dyNoseShoulder }
            }
            
            State.HEAD_DOWN -> {
                // 低头: dy_nose_shoulder 最大值
                actionLimits[State.HEAD_DOWN] = validFrames.maxOf { it.dyNoseShoulder }
            }
            
            State.HEAD_LEFT -> {
                // 左看: dx_nose_shoulder 最小值 (负值最大)
                actionLimits[State.HEAD_LEFT] = abs(validFrames.minOf { it.dxNoseShoulder })
            }
            
            State.HEAD_RIGHT -> {
                // 右看: dx_nose_shoulder 最大值
                actionLimits[State.HEAD_RIGHT] = abs(validFrames.maxOf { it.dxNoseShoulder })
            }
            
            State.POSTURE_DEVIATION -> {
                // 姿态偏移: dy_shoulder 最大值
                actionLimits[State.POSTURE_DEVIATION] = validFrames.maxOf { it.dyShoulder }
            }
            
            else -> {}
        }
        
        Log.i(TAG, "State $currentState completed, limit: ${actionLimits[currentState]}")
    }
    
    /**
     * 完成校准，计算阈值并保存
     */
    private fun finalizeCalibration() {
        val base = baseValues ?: run {
            Log.e(TAG, "No base values available")
            return
        }
        
        val limits = CalibrationData.ActionLimits(
            headUp = actionLimits[State.HEAD_UP] ?: base.headVertical - 100f,
            headDown = actionLimits[State.HEAD_DOWN] ?: base.headVertical + 100f,
            headLeft = actionLimits[State.HEAD_LEFT] ?: 50f,
            headRight = actionLimits[State.HEAD_RIGHT] ?: 50f,
            postureDeviation = actionLimits[State.POSTURE_DEVIATION] ?: base.posture + 30f,
            headTurned = base.eyeDistance * 0.3f  // 侧脸时双眼距离约为正面的30%
        )
        
        val thresholds = CalibrationData.calculateThresholds(base, limits)
        
        val calibrationData = CalibrationData(
            timestamp = System.currentTimeMillis(),
            durationSetting = selectedDuration.displayName,
            baseValues = base,
            actionLimits = limits,
            calculatedThresholds = thresholds
        )
        
        // 保存到本地
        saveCalibration(calibrationData)
        
        Log.i(TAG, "Calibration completed: $calibrationData")
    }
    
    /**
     * 保存校准数据到本地
     */
    private fun saveCalibration(data: CalibrationData) {
        try {
            val json = Gson().toJson(data)
            File(context.filesDir, CALIBRATION_FILE).writeText(json)
            Log.i(TAG, "Calibration saved")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save calibration", e)
        }
    }
    
    /**
     * 加载校准数据
     */
    fun loadCalibration(): CalibrationData? {
        return try {
            val file = File(context.filesDir, CALIBRATION_FILE)
            if (!file.exists()) return null
            
            val json = file.readText()
            Gson().fromJson(json, CalibrationData::class.java)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load calibration", e)
            null
        }
    }
    
    /**
     * 清除校准数据
     */
    fun clearCalibration() {
        File(context.filesDir, CALIBRATION_FILE).delete()
        currentState = State.IDLE
        collectionData.frames.clear()
        actionLimits.clear()
        baseValues = null
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        currentState = State.IDLE
        collectionData.frames.clear()
    }
}
