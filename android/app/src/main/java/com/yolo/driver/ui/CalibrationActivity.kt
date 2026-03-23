package com.yolo.driver.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.yolo.driver.DriverApplication
import com.yolo.driver.analyzer.CalibrationManager
import com.yolo.driver.analyzer.KeypointDetector
import com.yolo.driver.databinding.ActivityCalibrationBinding
import com.yolo.driver.util.CameraUtils
import com.yolo.driver.util.KeypointDrawer
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

class CalibrationActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "CalibrationActivity"
    }
    
    override fun attachBaseContext(newBase: Context) {
        // 使用 newBase 获取 application，而不是 applicationContext（此时可能为 null）
        val app = newBase.applicationContext as? DriverApplication
        val context = app?.applyLanguageToContext(newBase) ?: newBase
        super.attachBaseContext(context)
    }
    
    // 帧数据容器（线程安全）
    // 注意：imageProxy 需要在绘制完成后关闭
    private data class FrameData(
        val keypoints: List<KeypointDetector.KeyPoint>,
        val imageProxy: ImageProxy
    )
    
    private lateinit var binding: ActivityCalibrationBinding
    private lateinit var cameraExecutor: ExecutorService
    
    // ViewModel
    private val viewModel: CalibrationViewModel by viewModels { CalibrationViewModel.Factory(applicationContext) }
    
    private var detector: KeypointDetector? = null
    
    // 帧数据（线程安全）
    private val frameDataRef = AtomicReference<FrameData?>(null)
    
    // NV21 buffer 复用
    private var nv21Buffer: ByteArray? = null
    
    // Camera resources (member variables for proper lifecycle management)
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    
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
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityCalibrationBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // 使用单例获取检测器（与 MainActivity 共享）
        detector = KeypointDetector.getInstance()
        
        setupUI()
        observeViewModel()
        viewModel.loadManualRotation()  // 从 SharedPreferences 加载旋转角度
        
        if (CameraUtils.hasCameraPermission(this)) {
            startCamera()
        }
    }
    
    private fun setupUI() {
        // 时长选择
        binding.btnDuration2s.setOnClickListener {
            viewModel.setDuration(CalibrationManager.Duration.FAST)
        }
        binding.btnDuration3s.setOnClickListener {
            viewModel.setDuration(CalibrationManager.Duration.NORMAL)
        }
        binding.btnDuration5s.setOnClickListener {
            viewModel.setDuration(CalibrationManager.Duration.ACCURATE)
        }
        
        // 旋转按钮
        binding.btnRotate.setOnClickListener {
            viewModel.toggleManualRotation()
        }
        
        // 开始校准
        binding.btnStartCalibration.setOnClickListener {
            viewModel.startCalibration()
        }
        
        // 取消
        binding.btnCancel.setOnClickListener {
            viewModel.cancelCalibration()
            finish()
        }
        
        // 设置关键点绘制回调
        @androidx.camera.core.ExperimentalGetImage
        binding.overlayView.setKeypointDrawCallback { canvas, viewWidth, viewHeight ->
            frameDataRef.get()?.let { data ->
                try {
                    // 使用 CoordinateTransform API
                    KeypointDrawer.drawKeypointsWithTransform(
                        canvas, data.keypoints, data.imageProxy, binding.previewView,
                        pointPaint, linePaint, viewModel.getManualRotation(), mirror = true
                    )
                } finally {
                    // 绘制完成后关闭 ImageProxy
                    data.imageProxy.close()
                    frameDataRef.set(null)
                }
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
    
    private fun updateUI(state: CalibrationViewModel.UiState) {
        // 更新时长选择按钮
        binding.btnDuration2s.isSelected = state.duration == CalibrationManager.Duration.FAST
        binding.btnDuration3s.isSelected = state.duration == CalibrationManager.Duration.NORMAL
        binding.btnDuration5s.isSelected = state.duration == CalibrationManager.Duration.ACCURATE
        
        // 更新旋转按钮文本
        val rotationText = if (state.manualRotation == -1) "自动" else "${state.manualRotation}°"
        binding.btnRotate.text = "旋转: $rotationText"
        
        // 更新状态文本
        binding.tvState.text = state.stateDisplayName
        
        // 更新倒计时
        if (state.isCalibrating) {
            binding.tvCountdown.visibility = View.VISIBLE
            binding.tvCountdown.text = String.format("%.1f", state.countdownSeconds)
        } else {
            binding.tvCountdown.visibility = View.GONE
        }
        
        // 更新进度条
        binding.progressBar.progress = state.progress
        binding.progressBar.max = state.maxProgress
        
        // 显示对应动作示意图
        binding.ivActionGuide.visibility = when (state.state) {
            CalibrationManager.State.IDLE -> View.GONE
            CalibrationManager.State.DONE -> View.GONE
            else -> View.VISIBLE
        }
        
        // 校准中状态
        if (state.isCalibrating) {
            binding.btnStartCalibration.visibility = View.GONE
            binding.btnCancel.text = "取消"
            binding.durationSelector.visibility = View.GONE
        }
        
        // 完成状态
        if (state.isCompleted) {
            binding.btnStartCalibration.visibility = View.VISIBLE
            binding.btnCancel.text = "返回"
            setResult(RESULT_OK)
            Toast.makeText(this, "校准成功!", Toast.LENGTH_SHORT).show()
        }
        
        // 初始状态
        if (state.state == CalibrationManager.State.IDLE && !state.isCompleted) {
            binding.btnStartCalibration.visibility = View.VISIBLE
            binding.btnCancel.text = "取消"
            binding.durationSelector.visibility = View.VISIBLE
        }
        
        // 错误消息
        state.errorMessage?.let { message ->
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
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
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processFrame(imageProxy: ImageProxy) {
        val uiState = viewModel.uiState.value
        if (!uiState.isCalibrating) {
            imageProxy.close()
            return
        }
        
        // 关闭上一帧的 ImageProxy（如果有）
        frameDataRef.getAndSet(null)?.imageProxy?.close()
        
        try {
            val nv21 = CameraUtils.imageProxyToNV21(imageProxy, nv21Buffer)
            nv21Buffer = nv21
            val result = detector?.detect(nv21, imageProxy.width, imageProxy.height)
            
            // 保存帧数据（包含 ImageProxy，绘制后关闭）
            result?.keypoints?.let { kps ->
                frameDataRef.set(FrameData(kps, imageProxy))
                
                // 更新覆盖层
                runOnUiThread {
                    binding.overlayView.invalidate()
                }
            }
            
            // 通过 ViewModel 处理帧数据
            viewModel.processFrame(result?.keypoints)
            
            // 如果没有关键点，立即关闭
            if (result?.keypoints == null) {
                imageProxy.close()
                frameDataRef.set(null)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
            imageProxy.close()
            frameDataRef.set(null)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        // ProcessCameraProvider 绑定到 Lifecycle，系统自动管理，无需手动解绑
        // cameraProvider?.unbindAll()  // 移除！避免破坏 MainActivity 的相机绑定
        cameraProvider = null
        preview = null
        imageAnalyzer = null
        // 使用引用计数释放
        KeypointDetector.releaseInstance()
    }
}
