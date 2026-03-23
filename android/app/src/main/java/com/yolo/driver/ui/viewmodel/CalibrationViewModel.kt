package com.yolo.driver.ui.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 校准界面 ViewModel (Compose 版本)
 */
class CalibrationViewModel : ViewModel() {
    
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
        val manualRotation: Int = -1
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 校准动作队列
    private val actionQueue = ArrayDeque<CalibrationAction>()
    
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
        
        // TODO: 启动倒计时协程
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
        }
    }
    
    /**
     * 完成校准
     */
    private fun completeCalibration() {
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
            
            if (newCollectedFrames >= state.requiredFrames) {
                // 当前阶段完成
                if (state.currentPhase == CalibrationPhase.COLLECTING_BASE) {
                    // 进入动作收集阶段
                    _uiState.value = _uiState.value.copy(
                        currentPhase = CalibrationPhase.COLLECTING_ACTION,
                        currentAction = actionQueue.firstOrNull(),
                        collectedFrames = 0,
                        progress = 0f
                    )
                } else {
                    // 跳到下一个动作
                    skipCurrentAction()
                }
            } else {
                _uiState.value = _uiState.value.copy(
                    collectedFrames = newCollectedFrames,
                    progress = newProgress
                )
            }
        }
    }
    
    /**
     * 切换手动旋转
     */
    fun toggleManualRotation() {
        val current = _uiState.value.manualRotation
        val newRotation = when (current) {
            -1 -> 0
            0 -> 90
            90 -> 180
            180 -> 270
            270 -> -1
            else -> -1
        }
        _uiState.value = _uiState.value.copy(manualRotation = newRotation)
    }
    
    /**
     * 重置
     */
    fun reset() {
        actionQueue.clear()
        _uiState.value = UiState()
    }
    
    override fun onCleared() {
        super.onCleared()
        actionQueue.clear()
    }
}
