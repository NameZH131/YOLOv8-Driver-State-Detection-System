package com.yolo.driver.ui.compose

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.RotateRight
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.yolo.driver.R
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.ui.compose.components.CameraPreview
import com.yolo.driver.ui.compose.components.KeypointOverlay
import com.yolo.driver.ui.compose.theme.DriverMonitorTheme
import com.yolo.driver.ui.viewmodel.CalibrationViewModel
import com.yolo.driver.util.CameraUtils
import java.io.File

/**
 * @writer: zhangheng
 * 校准界面 Composable
 */
@Composable
fun CalibrationScreen(
    viewModel: CalibrationViewModel = viewModel(factory = CalibrationViewModel.Factory(LocalContext.current)),
    onNavigateBack: () -> Unit = {}
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
    
    // NV21 缓冲区复用
    var nv21Buffer by remember { mutableStateOf<ByteArray?>(null) }
    
    // 初始化
    DisposableEffect(Unit) {
        // 请求相机权限
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        
        // 初始化检测器
        detector = KeypointDetector.getInstance()
        
        val paramPath = File(context.filesDir, "yolov8n_pose.ncnn.param").absolutePath
        val binPath = File(context.filesDir, "yolov8n_pose.ncnn.bin").absolutePath
        
        if (detector?.init(paramPath, binPath, useGPU = true) != true) {
            Toast.makeText(context, "检测器初始化失败", Toast.LENGTH_LONG).show()
        }
        
        onDispose {
            KeypointDetector.releaseInstance()
            detector = null
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
                        viewModel.updateKeypoints(
                            result.keypoints,
                            imageProxy.width,
                            imageProxy.height
                        )
                        viewModel.processFrame()
                    }
                    is KeypointDetector.DetectResult.Failed -> {
                        viewModel.clearKeypoints()
                    }
                }
            }
        }
        imageProxy.close()
    }
    
    DriverMonitorTheme(darkTheme = true) {
        Box(modifier = Modifier.fillMaxSize()) {
            // 相机预览
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
                rotation = 0,
                manualRotation = uiState.manualRotation,
                modifier = Modifier.fillMaxSize()
            )
            
            // 顶部信息栏
            Column(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(16.dp)
            ) {
                // 当前阶段提示
                val phaseText = when (uiState.currentPhase) {
                    CalibrationViewModel.CalibrationPhase.IDLE ->
                        stringResource(R.string.calibrate)
                    CalibrationViewModel.CalibrationPhase.COLLECTING_BASE ->
                        stringResource(R.string.calibration_phase_base)
                    CalibrationViewModel.CalibrationPhase.COLLECTING_ACTION ->
                        stringResource(R.string.calibration_phase_action)
                    CalibrationViewModel.CalibrationPhase.COMPLETED ->
                        stringResource(R.string.calibration_success)
                }
                
                Text(
                    text = phaseText,
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    modifier = Modifier
                        .clip(MaterialTheme.shapes.medium)
                        .background(Color.Black.copy(alpha = 0.6f))
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                )
                
                // 进度条和倒计时
                if (uiState.currentPhase == CalibrationViewModel.CalibrationPhase.COLLECTING_BASE ||
                    uiState.currentPhase == CalibrationViewModel.CalibrationPhase.COLLECTING_ACTION) {
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    LinearProgressIndicator(
                        progress = { uiState.progress.coerceIn(0f, 1f) },
                        modifier = Modifier
                            .fillMaxWidth(0.8f)
                            .height(8.dp),
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "${uiState.countdown}s",
                        style = MaterialTheme.typography.headlineMedium,
                        color = Color.White
                    )
                }
                
                // 当前动作提示
                uiState.currentAction?.let { action ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = getActionPrompt(action),
                        style = MaterialTheme.typography.bodyLarge,
                        color = Color.Yellow,
                        modifier = Modifier
                            .clip(MaterialTheme.shapes.medium)
                            .background(Color.Black.copy(alpha = 0.6f))
                            .padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                }
            }
            
            // 底部控制栏
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // 取消按钮
                TextButton(onClick = {
                    viewModel.reset()
                    onNavigateBack()
                }) {
                    Text(
                        text = stringResource(R.string.cancel),
                        color = Color.White
                    )
                }
                
                // 旋转按钮
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = {
                        viewModel.toggleManualRotation()
                    }) {
                        Icon(
                            imageVector = Icons.Default.RotateRight,
                            contentDescription = stringResource(R.string.rotate),
                            tint = Color.White
                        )
                    }
                    Text(
                        text = "${uiState.manualRotation}°",
                        color = Color.White
                    )
                }
                
                // 开始/完成按钮
                Button(
                    onClick = {
                        when (uiState.currentPhase) {
                            CalibrationViewModel.CalibrationPhase.IDLE -> {
                                viewModel.startCalibration()
                            }
                            CalibrationViewModel.CalibrationPhase.COMPLETED -> {
                                viewModel.reset()
                                onNavigateBack()
                            }
                            else -> {
                                viewModel.skipCurrentAction()
                            }
                        }
                    }
                ) {
                    Text(
                        text = when (uiState.currentPhase) {
                            CalibrationViewModel.CalibrationPhase.IDLE ->
                                stringResource(R.string.calibrate)
                            CalibrationViewModel.CalibrationPhase.COMPLETED ->
                                stringResource(R.string.save)
                            else -> stringResource(R.string.skip)
                        }
                    )
                }
            }
            
            // 加载指示器
            if (uiState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.Center),
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

/**
 * 获取动作提示文本
 */
@Composable
private fun getActionPrompt(action: CalibrationViewModel.CalibrationAction): String {
    return when (action) {
        CalibrationViewModel.CalibrationAction.HEAD_UP ->
            stringResource(R.string.calibration_action_head_up)
        CalibrationViewModel.CalibrationAction.HEAD_DOWN ->
            stringResource(R.string.calibration_action_head_down)
        CalibrationViewModel.CalibrationAction.LOOK_LEFT ->
            stringResource(R.string.calibration_action_look_left)
        CalibrationViewModel.CalibrationAction.LOOK_RIGHT ->
            stringResource(R.string.calibration_action_look_right)
        CalibrationViewModel.CalibrationAction.POSTURE_DEVIATION ->
            stringResource(R.string.calibration_action_posture_deviation)
    }
}