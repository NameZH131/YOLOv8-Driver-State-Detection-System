package com.yolo.driver.ui.viewmodel

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.analyzer.KeypointDetector
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * @writer: zhangheng
 * 校准界面 ViewModel (Compose 版本)
 * 集成 CalibrationManager 进行真正的校准数据采集和保存
 */
class CalibrationViewModel(
    private val calibrationManager: CalibrationManager,
    private val sharedPreferences: SharedPreferences
) : ViewModel() {
    
    companion object {
        private const val KEY_MANUAL_ROTATION = "calibration_manual_rotation"
        private const val TAG = "CalibrationViewModel"
    }
    
    // 校准阶段
    enum class CalibrationPhase {
        IDLE,               // 空闲
        COLLECTING_BASE,    // 收集基准数据
        COLLECTING_ACTION,  // 收集动作边界
        COMPLETED           // 完成
    }
    
    // 校准动作
    enum class CalibrationAction {
        HEAD_UP,
        HEAD_DOWN,
        LOOK_LEFT,
        LOOK_RIGHT,
        POSTURE_DEVIATION
    }
    
    // UI 状态
    data class UiState(
        val currentPhase: CalibrationPhase = CalibrationPhase.IDLE,
        val currentAction: CalibrationAction? = null,
        val countdown: Int = 0,
        val progress: Float = 0f,
        val collectedFrames: Int = 0,
        val requiredFrames: Int = 90,
        val isLoading: Boolean = false,
        val errorMessage: String? = null,
        val manualRotation: Int = 0,  // 手动旋转角度 (0, 90, 180, 270, 360)
        // 关键点数据
        val keypoints: List<KeypointDetector.KeyPoint> = emptyList(),
        val frameWidth: Int = 640,
        val frameHeight: Int = 480
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 关键点状态
    private val _keypoints = MutableStateFlow<List<KeypointDetector.KeyPoint>>(emptyList())
    val keypoints: StateFlow<List<KeypointDetector.KeyPoint>> = _keypoints.asStateFlow()
    
    // 校准动作队列
    private val actionQueue = ArrayDeque<CalibrationAction>()
    
    // 倒计时 Job
    private var countdownJob: Job? = null
    
    // 校准完成回调（由外部设置，在用户确认保存时调用）
    var onCalibrationComplete: (() -> Unit)? = null
    
    init {
        // 加载保存的旋转角度
        val savedRotation = sharedPreferences.getInt(KEY_MANUAL_ROTATION, 0)
        _uiState.value = _uiState.value.copy(manualRotation = savedRotation)
    }
    
    /**
     * 开始校准
     */
    fun startCalibration(durationSeconds: Int = 3) {
        // 设置校准时长
        val duration = when (durationSeconds) {
            2 -> CalibrationManager.Duration.FAST
            5 -> CalibrationManager.Duration.ACCURATE
            else -> CalibrationManager.Duration.NORMAL
        }
        calibrationManager.setDuration(duration)
        
        // 启动 CalibrationManager
        if (!calibrationManager.startCalibration()) {
            Log.e(TAG, "Failed to start calibration")
            return
        }
        
        _uiState.value = _uiState.value.copy(
            currentPhase = CalibrationPhase.COLLECTING_BASE,
            countdown = durationSeconds,
            progress = 0f,
            collectedFrames = 0,
            requiredFrames = duration.maxFrames
        )
        
        // 初始化动作队列
        actionQueue.clear()
        actionQueue.addAll(CalibrationAction.values())
        
        // 启动倒计时
        startCountdown(durationSeconds)
        
        Log.i(TAG, "Calibration started, duration: ${durationSeconds}s")
    }
    
    /**
     * 启动倒计时
     */
    private fun startCountdown(durationSeconds: Int) {
        countdownJob?.cancel()
        
        countdownJob = viewModelScope.launch {
            var remaining = durationSeconds
            while (isActive && remaining > 0) {
                _uiState.value = _uiState.value.copy(countdown = remaining)
                delay(1000)
                remaining--
            }
            
            if (remaining <= 0) {
                // 当前阶段完成
                completeCurrentPhase()
            }
        }
    }
    
    /**
     * 完成当前阶段
     */
    private fun completeCurrentPhase() {
        // 通知 CalibrationManager 完成当前状态
        val newState = calibrationManager.completeCurrentState()
        
        Log.i(TAG, "CalibrationManager state: $newState")
        
        when (newState) {
            CalibrationManager.State.HEAD_UP -> {
                _uiState.value = _uiState.value.copy(
                    currentPhase = CalibrationPhase.COLLECTING_ACTION,
                    currentAction = CalibrationAction.HEAD_UP,
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationManager.State.HEAD_DOWN -> {
                _uiState.value = _uiState.value.copy(
                    currentAction = CalibrationAction.HEAD_DOWN,
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationManager.State.HEAD_LEFT -> {
                _uiState.value = _uiState.value.copy(
                    currentAction = CalibrationAction.LOOK_LEFT,
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationManager.State.HEAD_RIGHT -> {
                _uiState.value = _uiState.value.copy(
                    currentAction = CalibrationAction.LOOK_RIGHT,
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationManager.State.POSTURE_DEVIATION -> {
                _uiState.value = _uiState.value.copy(
                    currentAction = CalibrationAction.POSTURE_DEVIATION,
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationManager.State.DONE -> {
                // 校准完成
                completeCalibration()
            }
            else -> {}
        }
    }
    
    /**
     * 跳过当前动作
     */
    fun skipCurrentAction() {
        // 通知 CalibrationManager 完成当前状态（跳过）
        val newState = calibrationManager.completeCurrentState()
        
        if (newState == CalibrationManager.State.DONE) {
            completeCalibration()
        } else {
            // 更新 UI 状态
            val nextAction = when (newState) {
                CalibrationManager.State.HEAD_UP -> CalibrationAction.HEAD_UP
                CalibrationManager.State.HEAD_DOWN -> CalibrationAction.HEAD_DOWN
                CalibrationManager.State.HEAD_LEFT -> CalibrationAction.LOOK_LEFT
                CalibrationManager.State.HEAD_RIGHT -> CalibrationAction.LOOK_RIGHT
                CalibrationManager.State.POSTURE_DEVIATION -> CalibrationAction.POSTURE_DEVIATION
                else -> null
            }
            
            _uiState.value = _uiState.value.copy(
                currentAction = nextAction,
                collectedFrames = 0,
                progress = 0f
            )
            startCountdown(3)
        }
    }
    
    /**
     * 完成校准
     */
    private fun completeCalibration() {
        countdownJob?.cancel()
        _uiState.value = _uiState.value.copy(
            currentPhase = CalibrationPhase.COMPLETED,
            currentAction = null,
            isLoading = false
        )
        
        // 注意：不在这里调用回调，让用户在 UI 上确认保存后再调用
        
        Log.i(TAG, "Calibration completed, waiting for user confirmation")
    }
    
    /**
     * 处理帧
     */
    fun processFrame() {
        val state = _uiState.value
        if (state.currentPhase == CalibrationPhase.COLLECTING_BASE ||
            state.currentPhase == CalibrationPhase.COLLECTING_ACTION) {
            
            // 获取当前关键点并传递给 CalibrationManager
            val keypoints = _keypoints.value
            calibrationManager.processFrame(keypoints)
            
            val newCollectedFrames = state.collectedFrames + 1
            val newProgress = newCollectedFrames.toFloat() / state.requiredFrames
            
            _uiState.value = _uiState.value.copy(
                collectedFrames = newCollectedFrames,
                progress = newProgress
            )
        }
    }
    
    /**
     * 更新关键点（用于 Compose 绘制）
     */
    fun updateKeypoints(kps: List<KeypointDetector.KeyPoint>, width: Int, height: Int) {
        _keypoints.value = kps
        _uiState.value = _uiState.value.copy(
            keypoints = kps,
            frameWidth = width,
            frameHeight = height
        )
    }
    
    /**
     * 清除关键点
     */
    fun clearKeypoints() {
        _keypoints.value = emptyList()
        _uiState.value = _uiState.value.copy(keypoints = emptyList())
    }
    
    /**
     * 切换手动旋转 (0 -> 90 -> 180 -> 270 -> 360 -> 0)
     */
    fun toggleManualRotation() {
        val oldRotation = _uiState.value.manualRotation
        val newRotation = when (oldRotation) {
            0 -> 90
            90 -> 180
            180 -> 270
            270 -> 360
            else -> 0
        }
        _uiState.value = _uiState.value.copy(manualRotation = newRotation)
        
        // 保存到 SharedPreferences
        sharedPreferences.edit()
            .putInt(KEY_MANUAL_ROTATION, newRotation)
            .apply()
            
        Log.d(TAG, "toggleManualRotation: $oldRotation -> $newRotation")
    }
    
    /**
     * 重置
     */
    fun reset() {
        countdownJob?.cancel()
        actionQueue.clear()
        calibrationManager.reset()
        _uiState.value = UiState()
    }
    
    override fun onCleared() {
        super.onCleared()
        countdownJob?.cancel()
        actionQueue.clear()
    }
    
    /**
     * Factory for creating CalibrationViewModel with dependencies
     */
    class Factory(
        private val context: Context
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            val sharedPreferences = context.getSharedPreferences("driver_monitor_prefs", Context.MODE_PRIVATE)
            val calibrationManager = CalibrationManager(context)
            return CalibrationViewModel(calibrationManager, sharedPreferences) as T
        }
    }
}