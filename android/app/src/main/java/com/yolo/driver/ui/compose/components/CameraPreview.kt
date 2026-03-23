package com.yolo.driver.ui.compose.components

import android.view.View
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * 相机预览 Composable
 * 使用 CameraX 和 AndroidView 包装 PreviewView
 */
@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onFrameAnalysis: (ImageProxy) -> Unit,
    cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    val previewView = remember { PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
    } }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    
    // 相机初始化
    LaunchedEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
        
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                    onFrameAnalysis(imageProxy)
                }
            }
        
        try {
            // 解绑所有用例后再绑定
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    // 清理资源
    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }
    
    AndroidView(
        factory = { previewView },
        modifier = modifier
    )
}

/**
 * 相机预览状态
 */
data class CameraPreviewState(
    val isReady: Boolean = false,
    val hasPermission: Boolean = false,
    val error: String? = null
)
