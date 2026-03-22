package com.yolo.driver.ui

import android.content.Context
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
 * 校准界面 ViewModel
 * 管理校准状态机、倒计时和帧采集
 */
class CalibrationViewModel(
    private val calibrationManager: CalibrationManager
) : ViewModel() {
    
    // UI 状态
    data class UiState(
        val state: CalibrationManager.State = CalibrationManager.State.IDLE,
        val stateDisplayName: String = "等待开始",
        val duration: CalibrationManager.Duration = CalibrationManager.Duration.NORMAL,
        val countdownSeconds: Float = 0f,
        val progress: Int = 0,
        val maxProgress: Int = 6,
        val isCalibrating: Boolean = false,
        val isCompleted: Boolean = false,
        val canStart: Boolean = true,
        val errorMessage: String? = null
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 倒计时 Job
    private var countdownJob: Job? = null
    
    // 状态转换顺序（用于计算进度）
    private val stateOrder = listOf(
        CalibrationManager.State.BASE_COLLECT,
        CalibrationManager.State.HEAD_UP,
        CalibrationManager.State.HEAD_DOWN,
        CalibrationManager.State.HEAD_LEFT,
        CalibrationManager.State.HEAD_RIGHT,
        CalibrationManager.State.POSTURE_DEVIATION
    )
    
    /**
     * 设置校准时长
     */
    fun setDuration(duration: CalibrationManager.Duration) {
        calibrationManager.setDuration(duration)
        _uiState.value = _uiState.value.copy(duration = duration)
    }
    
    /**
     * 开始校准
     */
    fun startCalibration() {
        if (!calibrationManager.startCalibration()) {
            return
        }
        
        _uiState.value = _uiState.value.copy(
            isCalibrating = true,
            canStart = false,
            isCompleted = false
        )
        
        startStateCountdown()
    }
    
    /**
     * 开始当前状态的倒计时
     */
    private fun startStateCountdown() {
        val state = calibrationManager.getState()
        
        if (state == CalibrationManager.State.IDLE || state == CalibrationManager.State.DONE) {
            return
        }
        
        // 更新 UI 状态
        val progress = stateOrder.indexOf(state).takeIf { it >= 0 } ?: 0
        _uiState.value = _uiState.value.copy(
            state = state,
            stateDisplayName = calibrationManager.getStateDisplayName(state),
            progress = progress,
            countdownSeconds = calibrationManager.getSelectedDuration().seconds.toFloat()
        )
        
        // 取消之前的倒计时
        countdownJob?.cancel()
        
        // 启动新倒计时
        val durationMs = calibrationManager.getSelectedDuration().seconds * 1000L
        
        countdownJob = viewModelScope.launch {
            val startTime = System.currentTimeMillis()
            
            while (isActive) {
                val elapsed = System.currentTimeMillis() - startTime
                val remaining = (durationMs - elapsed) / 1000f
                
                if (remaining <= 0) {
                    // 倒计时结束，完成当前状态
                    completeCurrentState()
                    break
                }
                
                _uiState.value = _uiState.value.copy(countdownSeconds = remaining)
                delay(100)
            }
        }
    }
    
    /**
     * 完成当前状态，进入下一状态
     */
    private fun completeCurrentState() {
        val nextState = calibrationManager.completeCurrentState()
        
        if (nextState == CalibrationManager.State.DONE) {
            // 校准完成
            countdownJob?.cancel()
            _uiState.value = _uiState.value.copy(
                state = CalibrationManager.State.DONE,
                stateDisplayName = "校准完成!",
                isCalibrating = false,
                isCompleted = true,
                canStart = true
            )
        } else {
            // 进入下一状态
            startStateCountdown()
        }
    }
    
    /**
     * 处理帧数据
     */
    fun processFrame(keypoints: List<KeypointDetector.KeyPoint>?) {
        if (!_uiState.value.isCalibrating) {
            return
        }
        
        calibrationManager.processFrame(keypoints)
    }
    
    /**
     * 取消校准
     */
    fun cancelCalibration() {
        countdownJob?.cancel()
        calibrationManager.reset()
        _uiState.value = UiState()
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        countdownJob?.cancel()
        calibrationManager.reset()
        _uiState.value = UiState()
    }
    
    override fun onCleared() {
        super.onCleared()
        countdownJob?.cancel()
    }
    
    /**
     * Factory for creating CalibrationViewModel with Context dependency
     */
    class Factory(
        private val context: Context
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            val calibrationManager = CalibrationManager(context.applicationContext)
            return CalibrationViewModel(calibrationManager) as T
        }
    }
}
