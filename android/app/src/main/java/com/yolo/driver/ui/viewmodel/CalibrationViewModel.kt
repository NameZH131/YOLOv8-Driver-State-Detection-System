package com.yolo.driver.ui.viewmodel

import android.content.Context
import android.content.SharedPreferences
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * 校准界面 ViewModel (Compose 版本)
 */
class CalibrationViewModel(
    private val sharedPreferences: SharedPreferences
) : ViewModel() {
    
    companion object {
        private const val KEY_MANUAL_ROTATION = "manual_rotation"
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
        val manualRotation: Int = 0  // 手动旋转角度 (0, 90, 180, 270)
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 校准动作队列
    private val actionQueue = ArrayDeque<CalibrationAction>()
    
    // 倒计时 Job
    private var countdownJob: Job? = null
    
    /**
     * 开始校准
     */
    fun startCalibration(durationSeconds: Int = 3) {
        _uiState.value = _uiState.value.copy(
            currentPhase = CalibrationPhase.COLLECTING_BASE,
            countdown = durationSeconds,
            progress = 0f,
            collectedFrames = 0,
            requiredFrames = durationSeconds * 30
        )
        
        // 初始化动作队列
        actionQueue.clear()
        actionQueue.addAll(CalibrationAction.values())
        
        // 启动倒计时
        startCountdown(durationSeconds)
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
        when (_uiState.value.currentPhase) {
            CalibrationPhase.COLLECTING_BASE -> {
                // 进入动作收集阶段
                _uiState.value = _uiState.value.copy(
                    currentPhase = CalibrationPhase.COLLECTING_ACTION,
                    currentAction = actionQueue.firstOrNull(),
                    collectedFrames = 0,
                    progress = 0f
                )
                startCountdown(3)
            }
            CalibrationPhase.COLLECTING_ACTION -> {
                // 跳到下一个动作
                skipCurrentAction()
            }
            else -> {}
        }
    }
    
    /**
     * 跳过当前动作
     */
    fun skipCurrentAction() {
        if (actionQueue.isNotEmpty()) {
            actionQueue.removeFirst()
        }
        
        if (actionQueue.isEmpty()) {
            completeCalibration()
        } else {
            _uiState.value = _uiState.value.copy(
                currentAction = actionQueue.first(),
                collectedFrames = 0
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
    }
    
    /**
     * 处理帧
     */
    fun processFrame() {
        val state = _uiState.value
        if (state.currentPhase == CalibrationPhase.COLLECTING_BASE ||
            state.currentPhase == CalibrationPhase.COLLECTING_ACTION) {
            
            val newCollectedFrames = state.collectedFrames + 1
            val newProgress = newCollectedFrames.toFloat() / state.requiredFrames
            
            _uiState.value = _uiState.value.copy(
                collectedFrames = newCollectedFrames,
                progress = newProgress
            )
        }
    }
    
    /**
     * 切换手动旋转 (0 -> 90 -> 180 -> 270 -> 360 -> 0)
     */
    fun toggleManualRotation() {
        val oldRotation = _uiState.value.manualRotation
        // 支持 360 度，循环：0 -> 90 -> 180 -> 270 -> 360 -> 0
        val newRotation = when (oldRotation) {
            0 -> 90
            90 -> 180
            180 -> 270
            270 -> 360
            else -> 0  // 360 或其他值回到 0
        }
        _uiState.value = _uiState.value.copy(manualRotation = newRotation)
        // 保存到 SharedPreferences
        sharedPreferences.edit()
            .putInt(KEY_MANUAL_ROTATION, newRotation)
            .apply()
        android.util.Log.d(TAG, "toggleManualRotation: $oldRotation -> $newRotation")
    }
    
    /**
     * 重置
     */
    fun reset() {
        countdownJob?.cancel()
        actionQueue.clear()
        _uiState.value = UiState()
    }
    
    override fun onCleared() {
        super.onCleared()
        countdownJob?.cancel()
        actionQueue.clear()
    }
    
    /**
     * Factory for creating CalibrationViewModel with Context dependency
     */
    class Factory(
        private val context: Context
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            val sharedPreferences = context.getSharedPreferences("driver_monitor_prefs", Context.MODE_PRIVATE)
            return CalibrationViewModel(sharedPreferences) as T
        }
    }
}
