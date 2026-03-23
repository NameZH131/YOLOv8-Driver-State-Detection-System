package com.yolo.driver.ui.compose

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageProxy
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
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
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
    
    // NV21 缓冲区复用
    var nv21Buffer by remember { mutableStateOf<ByteArray?>(null) }
    
    // 设置对话框
    var showSettingsDialog by remember { mutableStateOf(false) }
    
    // 初始化
    LaunchedEffect(Unit) {
        // 请求相机权限
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        
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
                val nv21 = CameraUtils.imageProxyToNV21(imageProxy, nv21Buffer)
                nv21Buffer = nv21
                
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
                    onFrameAnalysis = onFrameAnalysis
                )
            }
            
            // 关键点覆盖层
            KeypointOverlay(
                keypoints = uiState.keypoints,
                frameWidth = uiState.frameWidth,
                frameHeight = uiState.frameHeight,
                rotation = 0,  // 系统旋转，由 CameraX 处理
                manualRotation = uiState.manualRotation,
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
                        viewModel.updateSettings(newSettings)
                        showSettingsDialog = false
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