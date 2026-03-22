package com.yolo.driver

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.analyzer.StateAnalyzer
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.databinding.ActivityMainBinding
import com.yolo.driver.util.CameraUtils
import com.yolo.driver.util.KeypointDrawer
import com.yolo.driver.ui.CalibrationActivity
import android.app.Activity
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
    }
    
    // 帧数据容器（线程安全）
    private data class FrameData(
        val keypoints: List<KeypointDetector.KeyPoint>,
        val width: Int,
        val height: Int,
        val rotation: Int
    )
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    
    // ViewModel
    private val viewModel: MainViewModel by viewModels()
    
    // 检测器和分析器（在 ViewModel 初始化前需要本地持有）
    private var detector: KeypointDetector? = null
    private var analyzer: StateAnalyzer? = null
    private var calibrationManager: CalibrationManager? = null
    
    // 相机相关
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var preview: Preview? = null
    
    // 帧数据（线程安全）
    private val frameDataRef = AtomicReference<FrameData?>(null)
    
    // NV21 buffer 复用
    private var nv21Buffer: ByteArray? = null
    
    // Paint 对象复用
    private val pointPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val linePaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // 权限请求
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            handlePermissionDenied()
        }
    }
    
    private fun handlePermissionDenied() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            AlertDialog.Builder(this)
                .setTitle("需要相机权限")
                .setMessage("驾驶员状态检测需要使用摄像头来分析您的姿态，请授予权限")
                .setPositiveButton("重新授权") { _, _ ->
                    requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                }
                .setNegativeButton("退出") { _, _ -> finish() }
                .setCancelable(false)
                .show()
        } else {
            showSettingsDialog()
        }
    }
    
    private fun showSettingsDialog() {
        val uri = Uri.fromParts("package", packageName, null)
        AlertDialog.Builder(this)
            .setTitle("需要相机权限")
            .setMessage("检测到您已禁止相机权限，请在设置中手动开启")
            .setPositiveButton("去设置") { _, _ ->
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, uri))
            }
            .setNegativeButton("退出") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }
    
    // 校准结果回调
    private val calibrationLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            loadCalibration()
            viewModel.updateCalibrationStatus()
            Toast.makeText(this, "校准成功", Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // 初始化
        cameraExecutor = Executors.newSingleThreadExecutor()
        analyzer = StateAnalyzer()
        calibrationManager = CalibrationManager(this)
        
        // 初始化 ViewModel 依赖
        analyzer?.let { a ->
            calibrationManager?.let { cm ->
                viewModel.initAnalyzer(a, cm)
            }
        }
        
        // 加载校准数据
        loadCalibration()
        
        // 初始化检测器
        initDetector()
        
        // 设置 UI
        setupUI()
        
        // 观察 ViewModel 状态
        observeViewModel()
        
        // 检查权限
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    
    private fun initDetector() {
        try {
            // 如果 detector 为空才获取新实例（避免 refCount 重复增加）
            if (detector == null) {
                detector = KeypointDetector.getInstance()
            }
            val paramPath = copyAssetToFilesDir("yolov8n_pose.ncnn.param")
            val binPath = copyAssetToFilesDir("yolov8n_pose.ncnn.bin")
            
            val success = detector?.init(paramPath, binPath, useGPU = true) ?: false
            
            if (!success) {
                Toast.makeText(this, "模型加载失败", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init detector", e)
        }
    }
    
    private fun copyAssetToFilesDir(assetName: String): String {
        val outFile = File(filesDir, assetName)
        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }
    
    private fun loadCalibration() {
        val calibrationData = calibrationManager?.loadCalibration()
        if (calibrationData != null) {
            analyzer?.setCalibration(calibrationData)
        }
    }
    
    private fun setupUI() {
        // 旋转按钮
        binding.btnRotate.setOnClickListener {
            viewModel.toggleManualRotation()
        }
        
        binding.btnCalibrate.setOnClickListener {
            val intent = Intent(this, CalibrationActivity::class.java)
            calibrationLauncher.launch(intent)
        }
        
        binding.btnReset.setOnClickListener {
            viewModel.reset()
            Toast.makeText(this, "已重置", Toast.LENGTH_SHORT).show()
        }
        
        // 设置关键点绘制回调
        binding.overlayView.setKeypointDrawCallback { canvas, viewWidth, viewHeight ->
            frameDataRef.get()?.let { data ->
                KeypointDrawer.drawKeypoints(
                    canvas, data.keypoints, data.width, data.height, 
                    data.rotation, viewWidth, viewHeight,
                    pointPaint, linePaint, viewModel.getManualRotation()
                )
            }
        }
    }
    
    private fun observeViewModel() {
        lifecycleScope.launch {
            viewModel.uiState.collect { state ->
                updateUI(state)
            }
        }
    }
    
    private fun updateUI(state: MainViewModel.UiState) {
        // 驾驶员状态
        binding.tvDriverState.text = viewModel.getDriverStateDisplayName(state.driverState)
        binding.tvDriverState.setTextColor(viewModel.getDriverStateColor(state.driverState))
        
        // 头部姿态
        binding.tvHeadPose.text = if (state.headPoses.isEmpty()) "未知" else state.headPoses.joinToString(", ")
        
        // 帧计数
        binding.tvFrameCount.text = "帧: ${state.frameCount}"
        
        // 校准状态
        binding.tvCalibrationStatus.text = viewModel.getCalibrationStatusText()
        binding.tvCalibrationStatus.setTextColor(viewModel.getCalibrationStatusColor())
        
        // 手动旋转角度
        binding.tvRotation.text = "旋转: ${state.manualRotation}°"
        
        // 检测错误信息
        state.detectionError?.let { error ->
            binding.tvHeadPose.text = error
            binding.tvHeadPose.setTextColor(Color.RED)
        } ?: run {
            binding.tvHeadPose.setTextColor(Color.WHITE)
        }
        
        // 错误消息（Toast）
        state.errorMessage?.let { message ->
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
            viewModel.clearError()
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
            // 先解绑旧的 use cases，避免冲突
            try {
                cameraProvider?.unbindAll()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to unbind camera", e)
            }
            
            preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }
            
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }
            
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            
            try {
                cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processFrame(imageProxy: ImageProxy) {
        if (detector == null || !detector!!.isInitialized() || analyzer == null) {
            imageProxy.close()
            return
        }
        
        try {
            val nv21 = CameraUtils.imageProxyToNV21(imageProxy, nv21Buffer)
            nv21Buffer = nv21
            val width = imageProxy.width
            val height = imageProxy.height
            val rotation = imageProxy.imageInfo.rotationDegrees
            
            // 使用带错误处理的检测方法
            val detectResult = detector?.detectWithResult(nv21, width, height)
            
            // 处理检测结果
            detectResult?.let { result ->
                viewModel.processDetectResult(result)
                
                if (result is KeypointDetector.DetectResult.Success) {
                    // 保存帧数据
                    frameDataRef.set(FrameData(result.keypoints, width, height, rotation))
                    
                    // 分析状态
                    val analysis = analyzer?.analyze(result.keypoints)
                    viewModel.processAnalysis(analysis)
                }
            }
            
            runOnUiThread {
                binding.overlayView.invalidate()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        } finally {
            imageProxy.close()
        }
    }
    
    private fun allPermissionsGranted() = CameraUtils.hasCameraPermission(this)
    
    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume: checking detector status...")
        
        // 检查检测器状态，如果未初始化则重新获取
        if (detector == null) {
            Log.d(TAG, "Detector is null, acquiring new instance...")
            detector = KeypointDetector.getInstance()
        }
        
        if (!detector!!.isInitialized()) {
            Log.d(TAG, "Detector not initialized, initializing...")
            initDetector()
        }
        
        // 相机在 onCreate 中绑定到 Lifecycle，由系统自动管理启停，无需手动重启
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        // 显式解绑相机资源
        cameraProvider?.unbindAll()
        cameraProvider = null
        preview = null
        imageAnalyzer = null
        KeypointDetector.releaseInstance()
    }
    
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        Log.d(TAG, "Configuration changed: ${if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) "LANDSCAPE" else "PORTRAIT"}")
    }
}
