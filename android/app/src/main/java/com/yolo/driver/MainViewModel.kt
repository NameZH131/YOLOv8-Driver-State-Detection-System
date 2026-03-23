package com.yolo.driver

import android.content.Context
import android.graphics.Color
import android.net.Uri
import androidx.lifecycle.ViewModel
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.analyzer.SlidingWindowAnalyzer
import com.yolo.driver.analyzer.StateAnalyzer
import com.yolo.driver.util.AudioPlayer
import com.yolo.driver.util.VibrationController
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * @writer: zhangheng
 * 主界面 ViewModel
 * 管理驾驶员状态、校准状态、滑动窗分析、音频/震动控制等业务逻辑
 */
class MainViewModel : ViewModel() {
    
    // 驾驶员状态
    sealed class DriverState {
        object Normal : DriverState()
        object SlightlyTired : DriverState()
        object Tired : DriverState()
    }
    
    // 获取状态显示名称 (由外部提供 Context 以支持国际化)
    fun getDriverStateDisplayName(context: Context, state: DriverState): String = when (state) {
        is DriverState.Normal -> context.getString(R.string.driver_state_normal)
        is DriverState.SlightlyTired -> context.getString(R.string.driver_state_slightly_tired)
        is DriverState.Tired -> context.getString(R.string.driver_state_tired)
    }
    
    // 获取状态颜色
    fun getDriverStateColor(state: DriverState): Int = when (state) {
        is DriverState.Normal -> Color.GREEN
        is DriverState.SlightlyTired -> Color.YELLOW
        is DriverState.Tired -> Color.RED
    }
    
    // 姿态状态映射（检测到某姿态后对应的疲劳状态）
    data class PoseStateMapping(
        val headUpDown: DriverState = DriverState.Tired,               // 抬头/低头 -> 疲劳
        val headLeftRight: DriverState = DriverState.Normal,           // 左右摆头 -> 正常
        val postureDeviation: DriverState = DriverState.Tired          // 姿态偏移 -> 疲劳
    )
    
    // 设置状态
    data class SettingsState(
        val vibrationEnabled: Boolean = true,       // 震动开关，默认开启
        val vibrationMode: Int = 0,                 // 震动模式: 0=短震, 1=长震, 2=双击, 3=脉冲
        val audioEnabled: Boolean = true,           // 音频开关，默认开启
        val audioVolume: Int = 100,                 // 音量 0-100
        val tiredAudioUri: String? = null,          // 自定义疲劳音频 URI
        val slightlyTiredAudioUri: String? = null,  // 自定义轻度疲劳音频 URI
        val windowDurationMs: Long = 5000L,         // 滑动窗时长，默认5秒
        val languageMode: Int = 0,                  // 语言模式: 0=自动, 1=中文, 2=英文
        val isSlidingWindowMode: Boolean = false,   // 检测模式: false=逐帧检测, true=滑动窗模式
        // 姿态状态映射（逐帧模式和滑动窗模式分别设置）
        val framePoseMapping: PoseStateMapping = PoseStateMapping(),
        val slidingPoseMapping: PoseStateMapping = PoseStateMapping(),
        // 关键点置信度阈值
        val drawThreshold: Float = 0.5f,            // 绘制阈值 (0.3-0.8)
        val analysisThreshold: Float = 0.5f         // 分析阈值 (0.3-0.8)
    )
    
    // UI 状态
    data class UiState(
        val driverState: DriverState = DriverState.Normal,
        val headPoses: List<String> = emptyList(),
        val frameCount: Int = 0,
        val isCalibrated: Boolean = false,
        val manualRotation: Int = 0,
        val errorMessage: String? = null,
        val detectionError: String? = null,
        val noPersonCount: Int = 0,
        val settingsState: SettingsState = SettingsState(),
        val windowFrameCount: Int = 0,              // 滑动窗内帧数
        // 关键点数据（用于 Compose 绘制）
        val keypoints: List<KeypointDetector.KeyPoint> = emptyList(),
        val frameWidth: Int = 640,
        val frameHeight: Int = 480
    )
    
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    // 关键点状态（单独暴露，减少重组）
    private val _keypoints = MutableStateFlow<List<KeypointDetector.KeyPoint>>(emptyList())
    val keypoints: StateFlow<List<KeypointDetector.KeyPoint>> = _keypoints.asStateFlow()
    
    // 帧尺寸
    private val _frameSize = MutableStateFlow(Pair(640, 480))
    val frameSize: StateFlow<Pair<Int, Int>> = _frameSize.asStateFlow()
    
    // 分析器（由 Activity 注入）
    private var analyzer: StateAnalyzer? = null
    private var calibrationManager: CalibrationManager? = null
    
    // 滑动窗分析器
    private var slidingWindowAnalyzer: SlidingWindowAnalyzer = SlidingWindowAnalyzer(5000L)
    
    // 音频播放器 (由 Activity 注入 Context)
    private var audioPlayer: AudioPlayer? = null
    
    // 震动控制器 (由 Activity 注入 Context)
    private var vibrationController: VibrationController? = null
    
    // 上一次触发的状态 (用于避免重复播放)
    private var lastTriggeredState: DriverState = DriverState.Normal
    
    /**
     * 初始化分析器
     */
    fun initAnalyzer(analyzer: StateAnalyzer, calibrationManager: CalibrationManager) {
        this.analyzer = analyzer
        this.calibrationManager = calibrationManager
        updateCalibrationStatus()
    }
    
    /**
     * 初始化音频和震动控制器
     */
    fun initMediaControllers(context: Context) {
        audioPlayer = AudioPlayer(context)
        vibrationController = VibrationController(context)
        
        // 应用当前设置
        val settings = _uiState.value.settingsState
        vibrationController?.setEnabled(settings.vibrationEnabled)
        vibrationController?.setMode(settings.vibrationMode)
        audioPlayer?.setEnabled(settings.audioEnabled)
        audioPlayer?.setVolume(settings.audioVolume)
        
        // 设置自定义音频 Uri (安全解析)
        settings.tiredAudioUri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { uri -> audioPlayer?.setTiredAudioUri(uri) }
        }
        settings.slightlyTiredAudioUri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { uri -> audioPlayer?.setSlightlyTiredAudioUri(uri) }
        }
    }
    
    /**
     * 处理分析结果
     */
    fun processAnalysis(analysis: StateAnalyzer.AnalysisResult?) {
        analysis?.let { a ->
            // 根据检测模式处理
            val driverState = if (_uiState.value.settingsState.isSlidingWindowMode) {
                // 滑动窗模式：添加到窗口并获取聚合结果
                slidingWindowAnalyzer.addState(a)
                val aggregatedResult = slidingWindowAnalyzer.getAggregatedState()
                
                // 更新窗口帧数
                _uiState.value = _uiState.value.copy(
                    windowFrameCount = aggregatedResult.windowSize
                )
                
                // 转换状态
                when (aggregatedResult.driverState) {
                    StateAnalyzer.DriverState.NORMAL -> DriverState.Normal
                    StateAnalyzer.DriverState.SLIGHTLY_TIRED -> DriverState.SlightlyTired
                    StateAnalyzer.DriverState.TIRED -> DriverState.Tired
                }
            } else {
                // 逐帧检测模式：直接使用单帧结果
                _uiState.value = _uiState.value.copy(windowFrameCount = 0)
                
                when (a.driverState) {
                    StateAnalyzer.DriverState.NORMAL -> DriverState.Normal
                    StateAnalyzer.DriverState.SLIGHTLY_TIRED -> DriverState.SlightlyTired
                    StateAnalyzer.DriverState.TIRED -> DriverState.Tired
                }
            }
            
            // 转换头部姿态 (需要 Context，由 Activity 处理国际化)
            val headPoses = a.headPoses.map { pose ->
                pose.name  // 使用枚举名称，Activity 会转换
            }
            
            // 更新 UI 状态
            _uiState.value = _uiState.value.copy(
                driverState = driverState,
                headPoses = headPoses,
                frameCount = a.frameCount,
                detectionError = null,
                noPersonCount = 0
            )
            
            // 触发音频和震动
            triggerAlerts(driverState)
        }
    }
    
    /**
     * 更新关键点和帧尺寸（用于 Compose 绘制）
     */
    fun updateKeypoints(kps: List<KeypointDetector.KeyPoint>, width: Int, height: Int) {
        _keypoints.value = kps
        _frameSize.value = Pair(width, height)
        _uiState.value = _uiState.value.copy(
            keypoints = kps,
            frameWidth = width,
            frameHeight = height
        )
    }
    
    /**
     * 清除关键点（无人检测到时）
     */
    fun clearKeypoints() {
        _keypoints.value = emptyList()
        _uiState.value = _uiState.value.copy(keypoints = emptyList())
    }
    
    /**
     * 触发音频和震动提醒
     */
    private fun triggerAlerts(newState: DriverState) {
        android.util.Log.d("MainViewModel", "triggerAlerts: newState=$newState, lastTriggeredState=$lastTriggeredState, vibrationEnabled=${vibrationController?.isEnabled()}")
        
        // 避免重复触发相同状态
        if (newState == lastTriggeredState && newState == DriverState.Normal) {
            android.util.Log.d("MainViewModel", "Skip: same state and Normal")
            return
        }
        
        when (newState) {
            is DriverState.Tired -> {
                // 疲劳状态：播放疲劳音频，触发当前模式的震动
                android.util.Log.d("MainViewModel", "Tired: playing audio and vibrating")
                audioPlayer?.playTired()
                vibrationController?.vibrate()
            }
            is DriverState.SlightlyTired -> {
                // 轻度疲劳：播放轻度疲劳音频，触发当前模式的震动
                android.util.Log.d("MainViewModel", "SlightlyTired: playing audio and vibrating")
                audioPlayer?.playSlightlyTired()
                vibrationController?.vibrate()
            }
            is DriverState.Normal -> {
                // 正常状态：不播放音频
                android.util.Log.d("MainViewModel", "Normal: no alerts")
            }
        }
        
        lastTriggeredState = newState
    }
    
    /**
     * 处理检测结果（带错误处理）
     */
    fun processDetectResult(result: KeypointDetector.DetectResult) {
        when (result) {
            is KeypointDetector.DetectResult.Success -> {
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
                
                val showError = when (result.error) {
                    KeypointDetector.DetectError.NO_PERSON -> newNoPersonCount > 30
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
     * 切换手动旋转角度 (0 -> 90 -> 180 -> 270 -> 360 -> 0)
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
        android.util.Log.d("MainViewModel", "toggleManualRotation: $oldRotation -> $newRotation, uiState.manualRotation=${_uiState.value.manualRotation}")
    }
    
    /**
     * 设置手动旋转角度
     */
    fun setManualRotation(rotation: Int) {
        _uiState.value = _uiState.value.copy(manualRotation = rotation)
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        analyzer?.reset()
        calibrationManager?.clearCalibration()
        analyzer?.clearCalibration()
        slidingWindowAnalyzer.clear()
        lastTriggeredState = DriverState.Normal
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
     * 获取校准状态文本 (由 Activity 提供国际化)
     */
    fun getCalibrationStatusText(context: Context): String = 
        if (_uiState.value.isCalibrated) context.getString(R.string.calibrated) 
        else context.getString(R.string.not_calibrated)
    
    /**
     * 获取校准状态颜色
     */
    fun getCalibrationStatusColor(): Int = if (_uiState.value.isCalibrated) Color.GREEN else Color.YELLOW
    
    // ========== 设置相关方法 ==========
    
    /**
     * 更新设置状态
     */
    fun updateSettings(newSettings: SettingsState) {
        // 更新震动控制器
        vibrationController?.setEnabled(newSettings.vibrationEnabled)
        vibrationController?.setMode(newSettings.vibrationMode)
        
        // 更新音频控制器
        audioPlayer?.setEnabled(newSettings.audioEnabled)
        audioPlayer?.setVolume(newSettings.audioVolume)
        
        // 更新自定义音频 Uri (安全解析)
        newSettings.tiredAudioUri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { uri -> audioPlayer?.setTiredAudioUri(uri) }
        }
        newSettings.slightlyTiredAudioUri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { uri -> audioPlayer?.setSlightlyTiredAudioUri(uri) }
        }
        
        // 更新滑动窗时长
        if (newSettings.windowDurationMs != _uiState.value.settingsState.windowDurationMs) {
            slidingWindowAnalyzer = SlidingWindowAnalyzer(newSettings.windowDurationMs)
        }
        
        // 更新 StateAnalyzer 的姿态映射（同时设置两种模式的映射）
        analyzer?.setAllPoseMappings(
            StateAnalyzer.PoseStateMapping(
                headUpDown = convertDriverState(newSettings.framePoseMapping.headUpDown),
                headLeftRight = convertDriverState(newSettings.framePoseMapping.headLeftRight),
                postureDeviation = convertDriverState(newSettings.framePoseMapping.postureDeviation)
            ),
            StateAnalyzer.PoseStateMapping(
                headUpDown = convertDriverState(newSettings.slidingPoseMapping.headUpDown),
                headLeftRight = convertDriverState(newSettings.slidingPoseMapping.headLeftRight),
                postureDeviation = convertDriverState(newSettings.slidingPoseMapping.postureDeviation)
            )
        )
        // 设置当前检测模式
        analyzer?.setSlidingWindowMode(newSettings.isSlidingWindowMode)
        
        // 更新关键点置信度阈值
        analyzer?.setKeypointThresholds(newSettings.drawThreshold, newSettings.analysisThreshold)
        
        // 更新 UI 状态
        _uiState.value = _uiState.value.copy(settingsState = newSettings)
    }
    
    /**
     * 转换 DriverState (ViewModel -> StateAnalyzer)
     */
    private fun convertDriverState(state: DriverState): StateAnalyzer.DriverState {
        return when (state) {
            is DriverState.Normal -> StateAnalyzer.DriverState.NORMAL
            is DriverState.SlightlyTired -> StateAnalyzer.DriverState.SLIGHTLY_TIRED
            is DriverState.Tired -> StateAnalyzer.DriverState.TIRED
        }
    }
    
    /**
     * 设置震动开关
     */
    fun setVibrationEnabled(enabled: Boolean) {
        vibrationController?.setEnabled(enabled)
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(vibrationEnabled = enabled)
        )
    }
    
    /**
     * 设置震动模式
     */
    fun setVibrationMode(mode: Int) {
        vibrationController?.setMode(mode)
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(vibrationMode = mode)
        )
    }
    
    /**
     * 设置音频开关
     */
    fun setAudioEnabled(enabled: Boolean) {
        audioPlayer?.setEnabled(enabled)
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(audioEnabled = enabled)
        )
    }
    
    /**
     * 设置音量
     */
    fun setAudioVolume(volume: Int) {
        audioPlayer?.setVolume(volume)
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(audioVolume = volume)
        )
    }
    
    /**
     * 设置疲劳音频 Uri
     */
    fun setTiredAudioUri(uri: String?) {
        uri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { parsedUri -> audioPlayer?.setTiredAudioUri(parsedUri) }
        }
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(tiredAudioUri = uri)
        )
    }
    
    /**
     * 设置轻度疲劳音频 Uri
     */
    fun setSlightlyTiredAudioUri(uri: String?) {
        uri?.takeIf { it.isNotEmpty() }?.let { 
            safeParseUri(it)?.let { parsedUri -> audioPlayer?.setSlightlyTiredAudioUri(parsedUri) }
        }
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(slightlyTiredAudioUri = uri)
        )
    }
    
    /**
     * 设置滑动窗时长
     */
    fun setWindowDuration(durationMs: Long) {
        slidingWindowAnalyzer = SlidingWindowAnalyzer(durationMs)
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(windowDurationMs = durationMs)
        )
    }
    
    /**
     * 切换检测模式
     */
    fun setSlidingWindowMode(enabled: Boolean) {
        if (!enabled) {
            slidingWindowAnalyzer.clear()
        }
        _uiState.value = _uiState.value.copy(
            settingsState = _uiState.value.settingsState.copy(isSlidingWindowMode = enabled),
            windowFrameCount = if (enabled) _uiState.value.windowFrameCount else 0
        )
    }
    
    /**
     * 获取当前设置状态
     */
    fun getSettingsState(): SettingsState = _uiState.value.settingsState
    
    /**
     * 清理资源
     */
    override fun onCleared() {
        super.onCleared()
        audioPlayer?.destroy()
        vibrationController?.cancel()
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 安全解析 URI 字符串
     * @return 解析成功返回 Uri，失败返回 null
     */
    private fun safeParseUri(uriString: String): Uri? {
        return try {
            if (uriString.isBlank()) null
            else Uri.parse(uriString)
        } catch (e: Exception) {
            android.util.Log.e("MainViewModel", "Failed to parse URI: $uriString", e)
            null
        }
    }
}