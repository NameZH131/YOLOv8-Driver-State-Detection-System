package com.yolo.driver.util

import android.content.Context
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * 相机控制器
 * 封装 CameraX 相机管理，支持帧回调
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "CameraController"
    }
    
    // 帧数据回调
    interface FrameCallback {
        fun onFrame(nv21: ByteArray, width: Int, height: Int, rotation: Int)
    }
    
    // 相机组件
    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    
    // 执行器
    private var cameraExecutor: ExecutorService? = null
    
    // NV21 buffer 复用
    private var nv21Buffer: ByteArray? = null
    
    // 帧回调
    private var frameCallback: FrameCallback? = null
    
    // 是否已启动
    private var isStarted = false
    
    /**
     * 设置帧回调
     */
    fun setFrameCallback(callback: FrameCallback) {
        this.frameCallback = callback
    }
    
    /**
     * 启动相机
     * @param previewView 预览视图
     * @param useFrontCamera 是否使用前置摄像头
     */
    fun startCamera(previewView: PreviewView, useFrontCamera: Boolean = true): Boolean {
        if (isStarted) {
            Log.w(TAG, "Camera already started")
            return true
        }
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                
                // 初始化执行器
                cameraExecutor = Executors.newSingleThreadExecutor()
                
                // 配置预览
                preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                
                // 配置图像分析
                imageAnalyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                            processFrame(imageProxy)
                        }
                    }
                
                // 选择摄像头
                val cameraSelector = if (useFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                
                // 绑定生命周期
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
                
                isStarted = true
                Log.i(TAG, "Camera started successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(context))
        
        return true
    }
    
    /**
     * 处理帧数据
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val nv21 = CameraUtils.imageProxyToNV21(imageProxy, nv21Buffer)
            nv21Buffer = nv21
            
            val rotation = imageProxy.imageInfo.rotationDegrees
            
            // 回调
            frameCallback?.onFrame(nv21, imageProxy.width, imageProxy.height, rotation)
            
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }
    
    /**
     * 停止相机
     */
    fun stopCamera() {
        if (!isStarted) return
        
        cameraProvider?.unbindAll()
        cameraExecutor?.shutdown()
        
        cameraProvider = null
        preview = null
        imageAnalyzer = null
        cameraExecutor = null
        nv21Buffer = null
        
        isStarted = false
        Log.i(TAG, "Camera stopped")
    }
    
    /**
     * 检查是否已启动
     */
    fun isRunning(): Boolean = isStarted
    
    /**
     * 检查相机权限
     */
    fun hasCameraPermission(): Boolean = CameraUtils.hasCameraPermission(context)
}
