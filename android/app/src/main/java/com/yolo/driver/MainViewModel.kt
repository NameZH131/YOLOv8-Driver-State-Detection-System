package com.yolo.driver

import android.graphics.Color
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.analyzer.StateAnalyzer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * 主界面 ViewModel
 * 管理驾驶员状态、校准状态等业务逻辑
 */
class MainViewModel : ViewModel() {
    
    // 驾驶员状态
    sealed class DriverState {
        object Normal : DriverState()
        object SlightlyTired : DriverState()
        object Tired : DriverState()
    }
    
    // 获取状态显示名称
    fun getDriverStateDisplayName(state: DriverState): String = when (state) {
        is DriverState.Normal -> "正常"
        is DriverState.SlightlyTired -> "轻度疲劳"
        is DriverState.Tired -> "疲劳"
    }
    
    // 获取状态颜色
    fun getDriverStateColor(state: DriverState): Int = when (state) {
        is DriverState.Normal -> Color.GREEN
        is DriverState.SlightlyTired -> Color.YELLOW
        is DriverState.Tired -> Color.RED
    }
    
    // UI 状态
    data class UiState(
        val driverState: DriverState = DriverState.Normal,
        val headPoses: List<String> = emptyList(),
        val frameCount: Int = 0,
        val isCalibrated: Boolean = false,
        val manualRotation: Int = 0,
        val errorMessage: String? = null,
        val detectionError: String? = null,  // 检测错误信息
        val noPersonCount: Int = 0  // 连续未检测到人物的帧数
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 分析器（由 Activity 注入）
    private var analyzer: StateAnalyzer? = null
    private var calibrationManager: CalibrationManager? = null
    
    /**
     * 初始化分析器
     */
    fun initAnalyzer(analyzer: StateAnalyzer, calibrationManager: CalibrationManager) {
        this.analyzer = analyzer
        this.calibrationManager = calibrationManager
        updateCalibrationStatus()
    }
    
    /**
     * 处理分析结果
     */
    fun processAnalysis(analysis: StateAnalyzer.AnalysisResult?) {
        analysis?.let { a ->
            val driverState = when (a.driverState) {
                StateAnalyzer.DriverState.NORMAL -> DriverState.Normal
                StateAnalyzer.DriverState.SLIGHTLY_TIRED -> DriverState.SlightlyTired
                StateAnalyzer.DriverState.TIRED -> DriverState.Tired
            }
            
            val headPoses = a.headPoses.map { pose ->
                when (pose) {
                    StateAnalyzer.HeadPose.FACING_FORWARD -> "正视前方"
                    StateAnalyzer.HeadPose.HEAD_UP -> "抬头"
                    StateAnalyzer.HeadPose.HEAD_DOWN -> "低头"
                    StateAnalyzer.HeadPose.HEAD_OFFSET -> "头部偏移"
                    StateAnalyzer.HeadPose.HEAD_TURNED -> "侧脸"
                    StateAnalyzer.HeadPose.POSTURE_DEVIATION -> "坐姿倾斜"
                }
            }
            
            _uiState.value = _uiState.value.copy(
                driverState = driverState,
                headPoses = headPoses,
                frameCount = a.frameCount,
                detectionError = null,  // 清除错误
                noPersonCount = 0  // 重置计数
            )
        }
    }
    
    /**
     * 处理检测结果（带错误处理）
     */
    fun processDetectResult(result: KeypointDetector.DetectResult) {
        when (result) {
            is KeypointDetector.DetectResult.Success -> {
                // 清除错误状态
                _uiState.value = _uiState.value.copy(
                    detectionError = null,
                    noPersonCount = 0
                )
            }
            is KeypointDetector.DetectResult.Failed -> {
                val newNoPersonCount = when (result.error) {
                    KeypointDetector.DetectError.NO_PERSON -> _uiState.value.noPersonCount + 1
                    else -> _uiState.value.noPersonCount
                }
                
                // 只在连续多次未检测到人物时显示错误
                val showError = when (result.error) {
                    KeypointDetector.DetectError.NO_PERSON -> newNoPersonCount > 30  // 约1秒
                    KeypointDetector.DetectError.NOT_INITIALIZED -> true
                    else -> false
                }
                
                _uiState.value = _uiState.value.copy(
                    detectionError = if (showError) result.message else null,
                    noPersonCount = newNoPersonCount
                )
            }
        }
    }
    
    /**
     * 更新校准状态
     */
    fun updateCalibrationStatus() {
        val isCalibrated = calibrationManager?.loadCalibration() != null
        _uiState.value = _uiState.value.copy(isCalibrated = isCalibrated)
    }
    
    /**
     * 切换手动旋转角度
     */
    fun toggleManualRotation() {
        val newRotation = (_uiState.value.manualRotation + 90) % 360
        _uiState.value = _uiState.value.copy(manualRotation = newRotation)
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        analyzer?.reset()
        calibrationManager?.clearCalibration()
        analyzer?.clearCalibration()
        _uiState.value = UiState()
    }
    
    /**
     * 设置错误消息
     */
    fun setError(message: String?) {
        _uiState.value = _uiState.value.copy(errorMessage = message)
    }
    
    /**
     * 清除错误消息
     */
    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }
    
    /**
     * 获取当前手动旋转角度
     */
    fun getManualRotation(): Int = _uiState.value.manualRotation
    
    /**
     * 获取校准状态文本
     */
    fun getCalibrationStatusText(): String = if (_uiState.value.isCalibrated) "已校准" else "未校准"
    
    /**
     * 获取校准状态颜色
     */
    fun getCalibrationStatusColor(): Int = if (_uiState.value.isCalibrated) Color.GREEN else Color.YELLOW
}