package com.yolo.driver.ui.compose

import android.widget.Toast
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.yolo.driver.MainViewModel
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.ui.compose.components.CameraPreview
import com.yolo.driver.ui.compose.components.ControlBar
import com.yolo.driver.ui.compose.components.KeypointOverlay
import com.yolo.driver.ui.compose.components.SettingsDialog
import com.yolo.driver.ui.compose.components.StatePanel
import com.yolo.driver.ui.compose.theme.DriverMonitorTheme
import com.yolo.driver.util.CameraUtils

/**
 * 主界面 Composable
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel(),
    onNavigateToCalibration: () -> Unit = {},
    onSaveSettings: (MainViewModel.SettingsState) -> Unit = {}
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    
    // 检测器生命周期管理
    var detector by remember { mutableStateOf<KeypointDetector?>(null) }
    var showSettingsDialog by remember { mutableStateOf(false) }
    
    // 初始化检测器
    DisposableEffect(Unit) {
        detector = KeypointDetector.getInstance()
        
        // 初始化模型
        val paramPath = context.filesDir.resolve("yolov8n_pose.ncnn.param").absolutePath
        val binPath = context.filesDir.resolve("yolov8n_pose.ncnn.bin").absolutePath
        
        // 复制模型文件到内部存储（如果不存在）
        copyAssetsIfNeeded(context)
        
        if (detector?.init(paramPath, binPath, useGPU = true) != true) {
            Toast.makeText(context, "检测器初始化失败", Toast.LENGTH_LONG).show()
        }
        
        onDispose {
            KeypointDetector.releaseInstance()
            detector = null
        }
    }
    
    // NV21 缓冲区复用
    var nv21Buffer by remember { mutableStateOf<ByteArray?>(null) }
    
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
                        viewModel.processDetectResult(result)
                        // 这里需要调用 analyze 来获取分析结果
                        // 暂时跳过，因为需要 StateAnalyzer 实例
                    }
                    is KeypointDetector.DetectResult.Failed -> {
                        viewModel.processDetectResult(result)
                    }
                }
            }
        }
        imageProxy.close()
    }
    
    DriverMonitorTheme(darkTheme = true) {
        Box(modifier = Modifier.fillMaxSize()) {
            // 相机预览
            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                onFrameAnalysis = onFrameAnalysis
            )
            
            // 关键点覆盖层
            KeypointOverlay(
                keypoints = emptyList(),  // TODO: 从 ViewModel 获取
                frameWidth = 640,
                frameHeight = 480,
                rotation = 0,
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
                        onSaveSettings(newSettings)
                        showSettingsDialog = false
                    }
                )
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
        val targetFile = context.filesDir.resolve(fileName)
        if (!targetFile.exists()) {
            context.assets.open(fileName).use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }
}
