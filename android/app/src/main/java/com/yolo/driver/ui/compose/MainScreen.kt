package com.yolo.driver.ui.compose

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.ZoomState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicReference
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import com.yolo.driver.DriverApplication
import com.yolo.driver.MainViewModel
import com.yolo.driver.R
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.analyzer.StateAnalyzer
import com.yolo.driver.ui.compose.components.CameraPreview
import com.yolo.driver.ui.compose.components.ControlBar
import com.yolo.driver.ui.compose.components.KeypointOverlay
import com.yolo.driver.ui.compose.components.SettingsDialog
import com.yolo.driver.ui.compose.components.StatePanel
import com.yolo.driver.ui.compose.theme.DriverMonitorTheme
import com.yolo.driver.util.AudioPlayer
import com.yolo.driver.util.CameraUtils
import com.yolo.driver.util.VibrationController
import java.io.File

/**
 * @writer: zhangheng
 * 主界面 Composable
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel(),
    onNavigateToCalibration: () -> Unit = {},
    onExitApp: () -> Unit = {}
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    
    // 权限状态
    var hasCameraPermission by remember { 
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == 
                PackageManager.PERMISSION_GRANTED
        )
    }
    
    // 权限请求
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            Toast.makeText(context, R.string.camera_permission_denied, Toast.LENGTH_LONG).show()
        }
    }
    
    // 检测器生命周期管理
    var detector by remember { mutableStateOf<KeypointDetector?>(null) }
    var analyzer by remember { mutableStateOf<StateAnalyzer?>(null) }
    var calibrationManager by remember { mutableStateOf<CalibrationManager?>(null) }
    var audioPlayer by remember { mutableStateOf<AudioPlayer?>(null) }
    var vibrationController by remember { mutableStateOf<VibrationController?>(null) }
    
    // NV21 缓冲区复用 (线程安全)
    val nv21BufferRef = remember { AtomicReference<ByteArray?>(null) }
    
    // 设置对话框
    var showSettingsDialog by remember { mutableStateOf(false) }
    
    // CameraControl 用于缩放控制
    var cameraControl by remember { mutableStateOf<CameraControl?>(null) }
    var zoomRatio by remember { mutableStateOf(1.0f) }
    var minZoomRatio by remember { mutableStateOf(1.0f) }
    var maxZoomRatio by remember { mutableStateOf(5.0f) }
    
    // 前后摄像头切换
    var useFrontCamera by remember { mutableStateOf(true) }
    
    // 镜像关键点
    var mirrorKeypoints by remember { mutableStateOf(true) }
    
    // 生命周期监听 - 用于从校准页面返回时刷新状态
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                // 从校准页面返回时刷新校准状态和设置
                viewModel.updateCalibrationStatus()
                viewModel.loadSettingsFromPrefs(context)
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
    
    // 初始化
    LaunchedEffect(Unit) {
        // 请求相机权限
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        
        // 加载保存的设置（包括语言设置）
        viewModel.loadSettingsFromPrefs(context)
        
        // 初始化检测器
        detector = KeypointDetector.getInstance()
        
        // 复制模型文件到内部存储
        copyAssetsIfNeeded(context)
        
        val paramPath = File(context.filesDir, "yolov8n_pose.ncnn.param").absolutePath
        val binPath = File(context.filesDir, "yolov8n_pose.ncnn.bin").absolutePath
        
        if (detector?.init(paramPath, binPath, useGPU = true) != true) {
            Toast.makeText(context, "检测器初始化失败", Toast.LENGTH_LONG).show()
        }
        
        // 初始化分析器
        analyzer = StateAnalyzer()
        calibrationManager = CalibrationManager(context)
        
        // 初始化音频和震动控制器
        audioPlayer = AudioPlayer(context)
        vibrationController = VibrationController(context)
        
        // 注入到 ViewModel
        analyzer?.let { a ->
            calibrationManager?.let { cm ->
                viewModel.initAnalyzer(a, cm)
            }
        }
        viewModel.initMediaControllers(context)
        
        // 加载校准数据
        viewModel.updateCalibrationStatus()
    }
    
    // 清理资源
    DisposableEffect(Unit) {
        onDispose {
            KeypointDetector.releaseInstance()
            detector = null
            analyzer = null
            calibrationManager = null
            audioPlayer?.destroy()
            vibrationController?.cancel()
        }
    }
    
    // 帧处理回调
    val onFrameAnalysis: (ImageProxy) -> Unit = { imageProxy ->
        detector?.let { det ->
            if (det.isInitialized()) {
                val reuseBuffer = nv21BufferRef.get()
                val nv21 = CameraUtils.imageProxyToNV21(imageProxy, reuseBuffer)
                nv21BufferRef.set(nv21)
                
                val result = det.detectWithResult(
                    nv21,
                    imageProxy.width,
                    imageProxy.height
                )
                
                when (result) {
                    is KeypointDetector.DetectResult.Success -> {
                        // 更新关键点状态
                        viewModel.updateKeypoints(
                            result.keypoints,
                            imageProxy.width,
                            imageProxy.height
                        )
                        
                        // 进行疲劳分析
                        val analysis = analyzer?.analyze(result.keypoints)
                        viewModel.processAnalysis(analysis)
                    }
                    is KeypointDetector.DetectResult.Failed -> {
                        viewModel.processDetectResult(result)
                        viewModel.clearKeypoints()
                    }
                }
            }
        }
        imageProxy.close()
    }
    
    DriverMonitorTheme(darkTheme = true) {
        Box(modifier = Modifier.fillMaxSize()) {
            // 相机预览（需要权限）
            if (hasCameraPermission) {
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    useFrontCamera = useFrontCamera,
                    onFrameAnalysis = onFrameAnalysis,
                    onCameraControlReady = { control ->
                        cameraControl = control
                        // 切换摄像头后重置缩放
                        zoomRatio = 1.0f
                    },
                    onCameraReady = { camera ->
                        // 获取相机缩放范围
                        camera.cameraInfo.zoomState.value?.let { zoomState ->
                            minZoomRatio = zoomState.minZoomRatio
                            maxZoomRatio = zoomState.maxZoomRatio
                        }
                    }
                )
            }
            
            // 关键点覆盖层
            KeypointOverlay(
                keypoints = uiState.keypoints,
                frameWidth = uiState.frameWidth,
                frameHeight = uiState.frameHeight,
                rotation = 0,  // 系统旋转，由 CameraX 处理
                manualRotation = uiState.manualRotation,
                mirror = mirrorKeypoints,
                modifier = Modifier.fillMaxSize()
            )
            
            // 状态信息面板
            StatePanel(
                driverState = uiState.driverState,
                headPoses = uiState.headPoses,
                frameCount = uiState.frameCount,
                isCalibrated = uiState.isCalibrated,
                manualRotation = uiState.manualRotation,
                windowFrameCount = uiState.windowFrameCount,
                isSlidingWindowMode = uiState.settingsState.isSlidingWindowMode,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
            )
            
            // 底部控制栏
            ControlBar(
                rotationText = "${uiState.manualRotation}°",
                isCalibrated = uiState.isCalibrated,
                onRotate = { viewModel.toggleManualRotation() },
                onCalibrate = onNavigateToCalibration,
                onReset = { viewModel.reset() },
                onSettings = { showSettingsDialog = true },
                zoomRatio = zoomRatio,
                minZoomRatio = minZoomRatio,
                onZoomIn = {
                    cameraControl?.let { control ->
                        val newZoom = (zoomRatio * 1.2f).coerceAtMost(maxZoomRatio)
                        control.setZoomRatio(newZoom)
                        zoomRatio = newZoom
                    }
                },
                onZoomOut = {
                    cameraControl?.let { control ->
                        val newZoom = if (useFrontCamera && minZoomRatio >= 1.0f) {
                            // 前置摄像头：尝试缩小到 0.5x
                            (zoomRatio / 1.2f).coerceAtLeast(0.5f)
                        } else {
                            // 后置摄像头：正常缩小
                            (zoomRatio / 1.2f).coerceAtLeast(minZoomRatio)
                        }
                        control.setZoomRatio(newZoom)
                        zoomRatio = newZoom
                    }
                },
                useFrontCamera = useFrontCamera,
                onSwitchCamera = {
                    useFrontCamera = !useFrontCamera
                },
                mirrorKeypoints = mirrorKeypoints,
                onToggleMirror = {
                    mirrorKeypoints = !mirrorKeypoints
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
            )
            
            // 设置弹窗
            if (showSettingsDialog) {
                SettingsDialog(
                    currentSettings = uiState.settingsState,
                    onDismiss = { showSettingsDialog = false },
                    onSave = { newSettings ->
                        // 检查语言是否变化
                        val languageChanged = newSettings.languageMode != uiState.settingsState.languageMode
                        
                        // 保存语言设置到 SharedPreferences
                        if (languageChanged) {
                            DriverApplication.saveLanguageMode(context, newSettings.languageMode)
                        }
                        
                        // 保存姿态映射设置到 SharedPreferences
                        val frameMappingChanged = newSettings.framePoseMapping != uiState.settingsState.framePoseMapping
                        val slidingMappingChanged = newSettings.slidingPoseMapping != uiState.settingsState.slidingPoseMapping
                        
                        if (frameMappingChanged || slidingMappingChanged) {
                            DriverApplication.savePoseMappings(
                                context,
                                DriverApplication.PoseMappingData(
                                    headUpDown = driverStateToInt(newSettings.framePoseMapping.headUpDown),
                                    headLeftRight = driverStateToInt(newSettings.framePoseMapping.headLeftRight),
                                    postureDeviation = driverStateToInt(newSettings.framePoseMapping.postureDeviation)
                                ),
                                DriverApplication.PoseMappingData(
                                    headUpDown = driverStateToInt(newSettings.slidingPoseMapping.headUpDown),
                                    headLeftRight = driverStateToInt(newSettings.slidingPoseMapping.headLeftRight),
                                    postureDeviation = driverStateToInt(newSettings.slidingPoseMapping.postureDeviation)
                                )
                            )
                        }
                        
                        // 保存所有设置到 SharedPreferences（音频 URI、音量、震动等）
                        DriverApplication.saveAllSettings(
                            context = context,
                            vibrationEnabled = newSettings.vibrationEnabled,
                            vibrationMode = newSettings.vibrationMode,
                            audioEnabled = newSettings.audioEnabled,
                            audioVolume = newSettings.audioVolume,
                            tiredAudioUri = newSettings.tiredAudioUri,
                            slightlyTiredAudioUri = newSettings.slightlyTiredAudioUri,
                            windowDurationMs = newSettings.windowDurationMs,
                            isSlidingWindowMode = newSettings.isSlidingWindowMode,
                            drawThreshold = newSettings.drawThreshold,
                            analysisThreshold = newSettings.analysisThreshold,
                            alertRepeatMode = newSettings.alertRepeatMode
                        )
                        
                        viewModel.updateSettings(newSettings)
                        showSettingsDialog = false
                        
                        // 语言变化后重启 Activity 使配置生效
                        if (languageChanged) {
                            (context as? android.app.Activity)?.recreate()
                        }
                    },
                    onAutoSave = { newSettings ->
                        // 检查语言是否变化
                        val languageChanged = newSettings.languageMode != uiState.settingsState.languageMode
                        
                        // 保存语言设置到 SharedPreferences
                        if (languageChanged) {
                            DriverApplication.saveLanguageMode(context, newSettings.languageMode)
                        }
                        
                        // 保存姿态映射设置到 SharedPreferences
                        DriverApplication.savePoseMappings(
                            context,
                            DriverApplication.PoseMappingData(
                                headUpDown = driverStateToInt(newSettings.framePoseMapping.headUpDown),
                                headLeftRight = driverStateToInt(newSettings.framePoseMapping.headLeftRight),
                                postureDeviation = driverStateToInt(newSettings.framePoseMapping.postureDeviation)
                            ),
                            DriverApplication.PoseMappingData(
                                headUpDown = driverStateToInt(newSettings.slidingPoseMapping.headUpDown),
                                headLeftRight = driverStateToInt(newSettings.slidingPoseMapping.headLeftRight),
                                postureDeviation = driverStateToInt(newSettings.slidingPoseMapping.postureDeviation)
                            )
                        )
                        
                        // 保存所有设置到 SharedPreferences
                        DriverApplication.saveAllSettings(
                            context = context,
                            vibrationEnabled = newSettings.vibrationEnabled,
                            vibrationMode = newSettings.vibrationMode,
                            audioEnabled = newSettings.audioEnabled,
                            audioVolume = newSettings.audioVolume,
                            tiredAudioUri = newSettings.tiredAudioUri,
                            slightlyTiredAudioUri = newSettings.slightlyTiredAudioUri,
                            windowDurationMs = newSettings.windowDurationMs,
                            isSlidingWindowMode = newSettings.isSlidingWindowMode,
                            drawThreshold = newSettings.drawThreshold,
                            analysisThreshold = newSettings.analysisThreshold,
                            alertRepeatMode = newSettings.alertRepeatMode
                        )
                        
                        // 更新 ViewModel 状态（不关闭对话框）
                        viewModel.updateSettings(newSettings)
                        
                        // 语言变化后重启 Activity 使配置生效
                        if (languageChanged) {
                            (context as? android.app.Activity)?.recreate()
                        }
                    }
                )
            }
            
            // 错误提示
            uiState.detectionError?.let { error ->
                LaunchedEffect(error) {
                    Toast.makeText(context, error, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
}

/**
 * 复制 assets 文件到内部存储
 */
private fun copyAssetsIfNeeded(context: android.content.Context) {
    val files = listOf("yolov8n_pose.ncnn.param", "yolov8n_pose.ncnn.bin")
    files.forEach { fileName ->
        val targetFile = File(context.filesDir, fileName)
        if (!targetFile.exists()) {
            context.assets.open(fileName).use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }
}

/**
 * DriverState 转整数（用于保存到 SharedPreferences）
 */
private fun driverStateToInt(state: MainViewModel.DriverState): Int = when (state) {
    is MainViewModel.DriverState.Normal -> DriverApplication.STATE_NORMAL
    is MainViewModel.DriverState.SlightlyTired -> DriverApplication.STATE_SLIGHTLY_TIRED
    is MainViewModel.DriverState.Tired -> DriverApplication.STATE_TIRED
}