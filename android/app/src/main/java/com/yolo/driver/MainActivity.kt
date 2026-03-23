package com.yolo.driver

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
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
import com.yolo.driver.databinding.DialogSettingsBinding
import com.yolo.driver.util.CameraUtils
import com.yolo.driver.util.KeypointDrawer
import com.yolo.driver.util.VibrationController
import com.yolo.driver.ui.CalibrationActivity
import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    override fun attachBaseContext(newBase: Context) {
        // 使用 newBase 获取 application，而不是 applicationContext（此时可能为 null）
        val app = newBase.applicationContext as? DriverApplication
        val context = app?.applyLanguageToContext(newBase) ?: newBase
        super.attachBaseContext(context)
    }
    
    companion object {
        private const val TAG = "MainActivity"
        
        // SharedPreferences Keys
        private const val PREFS_NAME = "driver_monitor_prefs"
        private const val KEY_VIBRATION_ENABLED = "vibration_enabled"
        private const val KEY_VIBRATION_MODE = "vibration_mode"
        private const val KEY_AUDIO_ENABLED = "audio_enabled"
        private const val KEY_AUDIO_VOLUME = "audio_volume"
        private const val KEY_TIRED_AUDIO_URI = "tired_audio_uri"
        private const val KEY_SLIGHTLY_TIRED_AUDIO_URI = "slightly_tired_audio_uri"
        private const val KEY_WINDOW_DURATION = "window_duration"
        private const val KEY_LANGUAGE_MODE = "language_mode"
        private const val KEY_SLIDING_WINDOW_MODE = "sliding_window_mode"
        private const val KEY_MANUAL_ROTATION = "manual_rotation"
        
        // 姿态状态映射 Keys - 逐帧模式
        private const val KEY_FRAME_HEAD_UP_DOWN = "frame_head_up_down"
        private const val KEY_FRAME_HEAD_LEFT_RIGHT = "frame_head_left_right"
        private const val KEY_FRAME_POSTURE_DEVIATION = "frame_posture_deviation"
        
        // 姿态状态映射 Keys - 滑动窗模式
        private const val KEY_SLIDING_HEAD_UP_DOWN = "sliding_head_up_down"
        private const val KEY_SLIDING_HEAD_LEFT_RIGHT = "sliding_head_left_right"
        private const val KEY_SLIDING_POSTURE_DEVIATION = "sliding_posture_deviation"
        
        // 关键点置信度阈值 Keys
        private const val KEY_DRAW_THRESHOLD = "draw_threshold"
        private const val KEY_ANALYSIS_THRESHOLD = "analysis_threshold"
        
        // 音频选择请求码
        private const val REQUEST_TIRED_AUDIO = 1001
        private const val REQUEST_SLIGHTLY_TIRED_AUDIO = 1002
    }
    
    // 帧数据容器（线程安全）
    // 注意：imageProxy 需要在绘制完成后关闭，由绘制回调负责
    private data class FrameData(
        val keypoints: List<KeypointDetector.KeyPoint>,
        val imageProxy: ImageProxy
    )
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sharedPreferences: SharedPreferences
    
    // ViewModel
    private val viewModel: MainViewModel by viewModels()
    
    // 检测器和分析器（在 ViewModel 初始化前需要本地持有）
    private var detector: KeypointDetector? = null
    private var analyzer: StateAnalyzer? = null
    private var calibrationManager: CalibrationManager? = null
    
    // 相机相关
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
    
    // 存储权限请求（用于读取本地音频）
    private val requestStoragePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, R.string.storage_permission_denied, Toast.LENGTH_SHORT).show()
        }
    }
    
    // 音频文件选择器
    private val tiredAudioPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.setTiredAudioUri(it.toString())
            saveAudioUri(KEY_TIRED_AUDIO_URI, it.toString())
        }
    }
    
    private val slightlyTiredAudioPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.setSlightlyTiredAudioUri(it.toString())
            saveAudioUri(KEY_SLIGHTLY_TIRED_AUDIO_URI, it.toString())
        }
    }
    
    private fun handlePermissionDenied() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            AlertDialog.Builder(this)
                .setTitle(R.string.camera_permission_title)
                .setMessage(R.string.camera_permission_message)
                .setPositiveButton(R.string.reauthorize) { _, _ ->
                    requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                }
                .setNegativeButton(R.string.exit) { _, _ -> finish() }
                .setCancelable(false)
                .show()
        } else {
            showPermissionSettingsDialog()
        }
    }
    
    private fun showPermissionSettingsDialog() {
        val uri = Uri.fromParts("package", packageName, null)
        AlertDialog.Builder(this)
            .setTitle(R.string.camera_permission_title)
            .setMessage(R.string.camera_permission_denied)
            .setPositiveButton(R.string.go_to_settings) { _, _ ->
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, uri))
            }
            .setNegativeButton(R.string.exit) { _, _ -> finish() }
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
            Toast.makeText(this, R.string.calibration_success, Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // 初始化 SharedPreferences
        sharedPreferences = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        
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
        
        // 初始化音频和震动控制器
        viewModel.initMediaControllers(this)
        
        // 加载设置
        loadSettings()
        
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
    
    /**
     * 加载设置
     */
    private fun loadSettings() {
        val vibrationEnabled = sharedPreferences.getBoolean(KEY_VIBRATION_ENABLED, false)
        val vibrationMode = sharedPreferences.getInt(KEY_VIBRATION_MODE, 0)
        val audioEnabled = sharedPreferences.getBoolean(KEY_AUDIO_ENABLED, true)
        val audioVolume = sharedPreferences.getInt(KEY_AUDIO_VOLUME, 100)
        val tiredAudioUri = sharedPreferences.getString(KEY_TIRED_AUDIO_URI, null)
        val slightlyTiredAudioUri = sharedPreferences.getString(KEY_SLIGHTLY_TIRED_AUDIO_URI, null)
        val windowDuration = sharedPreferences.getLong(KEY_WINDOW_DURATION, 5000L)
        val languageMode = sharedPreferences.getInt(KEY_LANGUAGE_MODE, 0)
        val slidingWindowMode = sharedPreferences.getBoolean(KEY_SLIDING_WINDOW_MODE, false)
        val manualRotation = sharedPreferences.getInt(KEY_MANUAL_ROTATION, 0)
        
        // 加载姿态状态映射（默认：抬头/低头=疲劳, 侧看=正常, 姿态偏移=疲劳）
        val framePoseMapping = MainViewModel.PoseStateMapping(
            headUpDown = intToDriverState(sharedPreferences.getInt(KEY_FRAME_HEAD_UP_DOWN, 2)), // 默认疲劳
            headLeftRight = intToDriverState(sharedPreferences.getInt(KEY_FRAME_HEAD_LEFT_RIGHT, 0)), // 默认正常
            postureDeviation = intToDriverState(sharedPreferences.getInt(KEY_FRAME_POSTURE_DEVIATION, 2)) // 默认疲劳
        )
        
        val slidingPoseMapping = MainViewModel.PoseStateMapping(
            headUpDown = intToDriverState(sharedPreferences.getInt(KEY_SLIDING_HEAD_UP_DOWN, 2)), // 默认疲劳
            headLeftRight = intToDriverState(sharedPreferences.getInt(KEY_SLIDING_HEAD_LEFT_RIGHT, 0)), // 默认正常
            postureDeviation = intToDriverState(sharedPreferences.getInt(KEY_SLIDING_POSTURE_DEVIATION, 2)) // 默认疲劳
        )
        
        // 加载关键点置信度阈值（默认 0.5）
        val drawThreshold = sharedPreferences.getFloat(KEY_DRAW_THRESHOLD, 0.5f)
        val analysisThreshold = sharedPreferences.getFloat(KEY_ANALYSIS_THRESHOLD, 0.5f)
        
        Log.d(TAG, "loadSettings: slidingWindowMode=$slidingWindowMode (false=逐帧, true=滑动窗)")
        
        val settings = MainViewModel.SettingsState(
            vibrationEnabled = vibrationEnabled,
            vibrationMode = vibrationMode,
            audioEnabled = audioEnabled,
            audioVolume = audioVolume,
            tiredAudioUri = tiredAudioUri,
            slightlyTiredAudioUri = slightlyTiredAudioUri,
            windowDurationMs = windowDuration,
            languageMode = languageMode,
            isSlidingWindowMode = slidingWindowMode,
            framePoseMapping = framePoseMapping,
            slidingPoseMapping = slidingPoseMapping,
            drawThreshold = drawThreshold,
            analysisThreshold = analysisThreshold
        )
        
        viewModel.updateSettings(settings)
        viewModel.setManualRotation(manualRotation)
    }
    
    /**
     * 整数转 DriverState
     * 0 = Normal, 1 = SlightlyTired, 2 = Tired
     */
    private fun intToDriverState(value: Int): MainViewModel.DriverState {
        return when (value) {
            0 -> MainViewModel.DriverState.Normal
            1 -> MainViewModel.DriverState.SlightlyTired
            2 -> MainViewModel.DriverState.Tired
            else -> MainViewModel.DriverState.SlightlyTired
        }
    }
    
    /**
     * DriverState 转整数
     */
    private fun driverStateToInt(state: MainViewModel.DriverState): Int {
        return when (state) {
            is MainViewModel.DriverState.Normal -> 0
            is MainViewModel.DriverState.SlightlyTired -> 1
            is MainViewModel.DriverState.Tired -> 2
        }
    }
    
    /**
     * 保存设置
     */
    private fun saveSettings(settings: MainViewModel.SettingsState) {
        Log.d(TAG, "saveSettings: isSlidingWindowMode=${settings.isSlidingWindowMode}")
        sharedPreferences.edit()
            .putBoolean(KEY_VIBRATION_ENABLED, settings.vibrationEnabled)
            .putInt(KEY_VIBRATION_MODE, settings.vibrationMode)
            .putBoolean(KEY_AUDIO_ENABLED, settings.audioEnabled)
            .putInt(KEY_AUDIO_VOLUME, settings.audioVolume)
            .putLong(KEY_WINDOW_DURATION, settings.windowDurationMs)
            .putInt(KEY_LANGUAGE_MODE, settings.languageMode)
            .putBoolean(KEY_SLIDING_WINDOW_MODE, settings.isSlidingWindowMode)
            .putString(KEY_TIRED_AUDIO_URI, settings.tiredAudioUri)
            .putString(KEY_SLIGHTLY_TIRED_AUDIO_URI, settings.slightlyTiredAudioUri)
            // 保存姿态映射 - 逐帧模式
            .putInt(KEY_FRAME_HEAD_UP_DOWN, driverStateToInt(settings.framePoseMapping.headUpDown))
            .putInt(KEY_FRAME_HEAD_LEFT_RIGHT, driverStateToInt(settings.framePoseMapping.headLeftRight))
            .putInt(KEY_FRAME_POSTURE_DEVIATION, driverStateToInt(settings.framePoseMapping.postureDeviation))
            // 保存姿态映射 - 滑动窗模式
            .putInt(KEY_SLIDING_HEAD_UP_DOWN, driverStateToInt(settings.slidingPoseMapping.headUpDown))
            .putInt(KEY_SLIDING_HEAD_LEFT_RIGHT, driverStateToInt(settings.slidingPoseMapping.headLeftRight))
            .putInt(KEY_SLIDING_POSTURE_DEVIATION, driverStateToInt(settings.slidingPoseMapping.postureDeviation))
            // 保存关键点阈值
            .putFloat(KEY_DRAW_THRESHOLD, settings.drawThreshold)
            .putFloat(KEY_ANALYSIS_THRESHOLD, settings.analysisThreshold)
            .apply()
    }
    
    /**
     * 保存音频 Uri
     */
    private fun saveAudioUri(key: String, uri: String) {
        sharedPreferences.edit().putString(key, uri).apply()
    }
    
    private fun initDetector() {
        try {
            detector = KeypointDetector.getInstance()
            val paramPath = copyAssetToFilesDir("yolov8n_pose.ncnn.param")
            val binPath = copyAssetToFilesDir("yolov8n_pose.ncnn.bin")
            
            val success = detector?.init(paramPath, binPath, useGPU = true) ?: false
            
            if (!success) {
                Toast.makeText(this, R.string.model_load_failed, Toast.LENGTH_LONG).show()
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
        // 设置按钮
        binding.btnSettings.setOnClickListener {
            showSettingsDialog()
        }
        
        // 旋转按钮
        binding.btnRotate.setOnClickListener {
            val oldRotation = viewModel.getManualRotation()
            viewModel.toggleManualRotation()
            val newRotation = viewModel.getManualRotation()
            Log.d(TAG, "btnRotate clicked: $oldRotation -> $newRotation")
            // 保存旋转角度到 SharedPreferences
            sharedPreferences.edit()
                .putInt(KEY_MANUAL_ROTATION, newRotation)
                .apply()
            // 触发重绘以应用新的旋转角度
            binding.overlayView.invalidate()
        }
        
        binding.btnCalibrate.setOnClickListener {
            val intent = Intent(this, CalibrationActivity::class.java)
            calibrationLauncher.launch(intent)
        }
        
        binding.btnReset.setOnClickListener {
            viewModel.reset()
            Toast.makeText(this, R.string.reset_success, Toast.LENGTH_SHORT).show()
        }
        
        // 设置关键点绘制回调
        @androidx.camera.core.ExperimentalGetImage
        binding.overlayView.setKeypointDrawCallback { canvas, viewWidth, viewHeight ->
            frameDataRef.get()?.let { data ->
                try {
                    // 使用 CoordinateTransform API
                    val settings = viewModel.getSettingsState()
                    KeypointDrawer.drawKeypointsWithTransform(
                        canvas, data.keypoints, data.imageProxy, binding.previewView,
                        pointPaint, linePaint, viewModel.getManualRotation(), mirror = true,
                        confidenceThreshold = settings.drawThreshold
                    )
                } finally {
                    // 绘制完成后关闭 ImageProxy
                    data.imageProxy.close()
                    frameDataRef.set(null)
                }
            }
        }
    }
    
    /**
     * 显示设置弹窗（自动保存模式）
     */
    private fun showSettingsDialog() {
        val dialogBinding = DialogSettingsBinding.inflate(layoutInflater)
        
        // 加载当前设置
        val currentSettings = viewModel.getSettingsState()
        Log.d(TAG, "showSettingsDialog: currentSettings.isSlidingWindowMode=${currentSettings.isSlidingWindowMode}")
        
        // 标记是否正在初始化（避免初始化时触发自动保存）
        var isInitializing = true
        
        // ===== 初始化各控件的值 =====
        
        // 检测模式
        if (currentSettings.isSlidingWindowMode) {
            dialogBinding.rbModeSliding.isChecked = true
            dialogBinding.layoutWindowDuration.visibility = View.VISIBLE
            dialogBinding.layoutFramePoseMapping.visibility = View.GONE
            dialogBinding.layoutSlidingPoseMapping.visibility = View.VISIBLE
        } else {
            dialogBinding.rbModeFrame.isChecked = true
            dialogBinding.layoutWindowDuration.visibility = View.GONE
            dialogBinding.layoutFramePoseMapping.visibility = View.VISIBLE
            dialogBinding.layoutSlidingPoseMapping.visibility = View.GONE
        }
        
        // 窗口时长
        when (currentSettings.windowDurationMs) {
            3000L -> dialogBinding.rbWindow3s.isChecked = true
            5000L -> dialogBinding.rbWindow5s.isChecked = true
            10000L -> dialogBinding.rbWindow10s.isChecked = true
        }
        
        // 震动设置
        dialogBinding.switchVibration.isChecked = currentSettings.vibrationEnabled
        dialogBinding.layoutVibrationMode.visibility = 
            if (currentSettings.vibrationEnabled) View.VISIBLE else View.GONE
        
        // 震动模式 Spinner
        val vibrationModes = arrayOf(
            getString(R.string.vibration_short),
            getString(R.string.vibration_long),
            getString(R.string.vibration_double),
            getString(R.string.vibration_pulse)
        )
        val vibrationAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, vibrationModes)
        vibrationAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        dialogBinding.spinnerVibrationMode.adapter = vibrationAdapter
        dialogBinding.spinnerVibrationMode.setSelection(currentSettings.vibrationMode)
        
        // 音频设置
        dialogBinding.switchAudio.isChecked = currentSettings.audioEnabled
        val audioVisibility = if (currentSettings.audioEnabled) View.VISIBLE else View.GONE
        dialogBinding.layoutAudioVolume.visibility = audioVisibility
        dialogBinding.layoutTiredAudio.visibility = audioVisibility
        dialogBinding.layoutSlightlyTiredAudio.visibility = audioVisibility
        
        // 音量
        dialogBinding.seekBarVolume.progress = currentSettings.audioVolume
        dialogBinding.tvVolumeValue.text = "${currentSettings.audioVolume}%"
        
        // 音频选择 Spinner
        val audioOptions = arrayOf(
            getString(R.string.default_audio),
            getString(R.string.select_custom_audio)
        )
        val audioAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, audioOptions)
        audioAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        
        // 疲劳音频 - 先设置 adapter 和 selection
        dialogBinding.spinnerTiredAudio.adapter = audioAdapter
        if (!currentSettings.tiredAudioUri.isNullOrEmpty()) {
            dialogBinding.spinnerTiredAudio.setSelection(1, false)
        }
        
        // 轻度疲劳音频 - 先设置 adapter 和 selection
        dialogBinding.spinnerSlightlyTiredAudio.adapter = audioAdapter
        if (!currentSettings.slightlyTiredAudioUri.isNullOrEmpty()) {
            dialogBinding.spinnerSlightlyTiredAudio.setSelection(1, false)
        }
        
        // 语言
        when (currentSettings.languageMode) {
            0 -> dialogBinding.rbLanguageAuto.isChecked = true
            1 -> dialogBinding.rbLanguageZh.isChecked = true
            2 -> dialogBinding.rbLanguageEn.isChecked = true
        }
        
        // 关键点置信度阈值
        // SeekBar 范围 0-50, 对应阈值 0.3-0.8 (公式: threshold = 0.3 + progress * 0.01)
        val drawProgress = ((currentSettings.drawThreshold - 0.3f) / 0.01f).toInt().coerceIn(0, 50)
        val analysisProgress = ((currentSettings.analysisThreshold - 0.3f) / 0.01f).toInt().coerceIn(0, 50)
        dialogBinding.seekBarDrawThreshold.progress = drawProgress
        dialogBinding.seekBarAnalysisThreshold.progress = analysisProgress
        dialogBinding.tvDrawThresholdValue.text = String.format("%.2f", currentSettings.drawThreshold)
        dialogBinding.tvAnalysisThresholdValue.text = String.format("%.2f", currentSettings.analysisThreshold)
        
        // 姿态状态映射 Spinner 初始化
        val stateOptions = arrayOf(
            getString(R.string.driver_state_normal),
            getString(R.string.driver_state_slightly_tired),
            getString(R.string.driver_state_tired)
        )
        val stateAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, stateOptions)
        stateAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        
        // 逐帧模式姿态映射
        dialogBinding.spinnerFrameHeadUpDown.adapter = stateAdapter
        dialogBinding.spinnerFrameHeadUpDown.setSelection(driverStateToInt(currentSettings.framePoseMapping.headUpDown))
        
        dialogBinding.spinnerFrameHeadLeftRight.adapter = stateAdapter
        dialogBinding.spinnerFrameHeadLeftRight.setSelection(driverStateToInt(currentSettings.framePoseMapping.headLeftRight))
        
        dialogBinding.spinnerFramePostureDeviation.adapter = stateAdapter
        dialogBinding.spinnerFramePostureDeviation.setSelection(driverStateToInt(currentSettings.framePoseMapping.postureDeviation))
        
        // 滑动窗模式姿态映射
        dialogBinding.spinnerSlidingHeadUpDown.adapter = stateAdapter
        dialogBinding.spinnerSlidingHeadUpDown.setSelection(driverStateToInt(currentSettings.slidingPoseMapping.headUpDown))
        
        dialogBinding.spinnerSlidingHeadLeftRight.adapter = stateAdapter
        dialogBinding.spinnerSlidingHeadLeftRight.setSelection(driverStateToInt(currentSettings.slidingPoseMapping.headLeftRight))
        
        dialogBinding.spinnerSlidingPostureDeviation.adapter = stateAdapter
        dialogBinding.spinnerSlidingPostureDeviation.setSelection(driverStateToInt(currentSettings.slidingPoseMapping.postureDeviation))
        
        // 初始化完成
        isInitializing = false
        
        val dialog = AlertDialog.Builder(this)
            .setView(dialogBinding.root)
            .create()
        
        // ===== 自动保存辅助方法 =====
        fun autoSaveSettings(andRecreate: Boolean = false) {
            if (isInitializing) return
            
            // 从 ViewModel 获取最新的音频 URI（音频选择器回调已更新）
            val latestSettings = viewModel.getSettingsState()
            
            // 构建姿态映射
            val framePoseMapping = MainViewModel.PoseStateMapping(
                headUpDown = intToDriverState(dialogBinding.spinnerFrameHeadUpDown.selectedItemPosition),
                headLeftRight = intToDriverState(dialogBinding.spinnerFrameHeadLeftRight.selectedItemPosition),
                postureDeviation = intToDriverState(dialogBinding.spinnerFramePostureDeviation.selectedItemPosition)
            )
            
            val slidingPoseMapping = MainViewModel.PoseStateMapping(
                headUpDown = intToDriverState(dialogBinding.spinnerSlidingHeadUpDown.selectedItemPosition),
                headLeftRight = intToDriverState(dialogBinding.spinnerSlidingHeadLeftRight.selectedItemPosition),
                postureDeviation = intToDriverState(dialogBinding.spinnerSlidingPostureDeviation.selectedItemPosition)
            )
            
            val newSettings = MainViewModel.SettingsState(
                vibrationEnabled = dialogBinding.switchVibration.isChecked,
                vibrationMode = dialogBinding.spinnerVibrationMode.selectedItemPosition,
                audioEnabled = dialogBinding.switchAudio.isChecked,
                audioVolume = dialogBinding.seekBarVolume.progress,
                tiredAudioUri = latestSettings.tiredAudioUri,
                slightlyTiredAudioUri = latestSettings.slightlyTiredAudioUri,
                windowDurationMs = when {
                    dialogBinding.rbWindow3s.isChecked -> 3000L
                    dialogBinding.rbWindow10s.isChecked -> 10000L
                    else -> 5000L
                },
                languageMode = when {
                    dialogBinding.rbLanguageZh.isChecked -> 1
                    dialogBinding.rbLanguageEn.isChecked -> 2
                    else -> 0
                },
                isSlidingWindowMode = dialogBinding.rbModeSliding.isChecked,
                framePoseMapping = framePoseMapping,
                slidingPoseMapping = slidingPoseMapping,
                // 关键点阈值: SeekBar progress 0-50 -> threshold 0.3-0.8
                drawThreshold = 0.3f + dialogBinding.seekBarDrawThreshold.progress * 0.01f,
                analysisThreshold = 0.3f + dialogBinding.seekBarAnalysisThreshold.progress * 0.01f
            )
            
            saveSettings(newSettings)
            viewModel.updateSettings(newSettings)
            
            if (andRecreate) {
                Toast.makeText(this, R.string.settings_saved_restart, Toast.LENGTH_SHORT).show()
                dialog.dismiss()
                binding.root.postDelayed({ recreate() }, 150)
            }
        }
        
        // ===== 设置各控件的监听器（自动保存） =====
        
        // 检测模式切换 -> 自动保存
        dialogBinding.rgDetectionMode.setOnCheckedChangeListener { _, checkedId ->
            val isSlidingMode = (checkedId == R.id.rbModeSliding)
            dialogBinding.layoutWindowDuration.visibility = if (isSlidingMode) View.VISIBLE else View.GONE
            dialogBinding.layoutFramePoseMapping.visibility = if (isSlidingMode) View.GONE else View.VISIBLE
            dialogBinding.layoutSlidingPoseMapping.visibility = if (isSlidingMode) View.VISIBLE else View.GONE
            autoSaveSettings()
        }
        
        // 窗口时长切换 -> 自动保存
        dialogBinding.rgWindowDuration.setOnCheckedChangeListener { _, _ ->
            autoSaveSettings()
        }
        
        // ===== 姿态映射 Spinner 监听器 =====
        
        // 确认选择"正常"的对话框
        fun confirmNormalSelection(spinner: android.widget.Spinner, originalPosition: Int, onConfirmed: () -> Unit) {
            AlertDialog.Builder(this)
                .setTitle(R.string.confirm_normal_title)
                .setMessage(R.string.confirm_normal_message)
                .setPositiveButton(R.string.yes) { _, _ ->
                    onConfirmed()
                    autoSaveSettings()
                }
                .setNegativeButton(R.string.no) { _, _ ->
                    // 恢复原来的选择
                    spinner.setSelection(originalPosition, false)
                }
                .setCancelable(false)
                .show()
        }
        
        // 逐帧模式姿态映射监听
        dialogBinding.spinnerFrameHeadUpDown.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerFrameHeadUpDown, 2) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        dialogBinding.spinnerFrameHeadLeftRight.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerFrameHeadLeftRight, 0) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        dialogBinding.spinnerFramePostureDeviation.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerFramePostureDeviation, 2) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // 滑动窗模式姿态映射监听
        dialogBinding.spinnerSlidingHeadUpDown.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerSlidingHeadUpDown, 2) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        dialogBinding.spinnerSlidingHeadLeftRight.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerSlidingHeadLeftRight, 0) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        dialogBinding.spinnerSlidingPostureDeviation.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (firstSelection) { firstSelection = false; return }
                if (position == 0) {
                    confirmNormalSelection(dialogBinding.spinnerSlidingPostureDeviation, 2) {}
                } else {
                    autoSaveSettings()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // 震动开关切换 -> 自动保存
        dialogBinding.switchVibration.setOnCheckedChangeListener { _, isChecked ->
            dialogBinding.layoutVibrationMode.visibility = if (isChecked) View.VISIBLE else View.GONE
            autoSaveSettings()
        }
        
        // 震动模式选择 -> 自动保存
        dialogBinding.spinnerVibrationMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                autoSaveSettings()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // 音频开关切换 -> 自动保存
        dialogBinding.switchAudio.setOnCheckedChangeListener { _, isChecked ->
            val visibility = if (isChecked) View.VISIBLE else View.GONE
            dialogBinding.layoutAudioVolume.visibility = visibility
            dialogBinding.layoutTiredAudio.visibility = visibility
            dialogBinding.layoutSlightlyTiredAudio.visibility = visibility
            autoSaveSettings()
        }
        
        // 音量滑块 -> 松手时自动保存
        dialogBinding.seekBarVolume.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                dialogBinding.tvVolumeValue.text = "$progress%"
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                autoSaveSettings()
            }
        })
        
        // 疲劳音频监听器
        dialogBinding.spinnerTiredAudio.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 1 && viewModel.getSettingsState().tiredAudioUri.isNullOrEmpty()) {
                    requestStoragePermissionAndPickAudio(REQUEST_TIRED_AUDIO)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // 轻度疲劳音频监听器
        dialogBinding.spinnerSlightlyTiredAudio.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 1 && viewModel.getSettingsState().slightlyTiredAudioUri.isNullOrEmpty()) {
                    requestStoragePermissionAndPickAudio(REQUEST_SLIGHTLY_TIRED_AUDIO)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // 关键点绘制阈值滑块 -> 松手时自动保存
        dialogBinding.seekBarDrawThreshold.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val threshold = 0.3f + progress * 0.01f
                dialogBinding.tvDrawThresholdValue.text = String.format("%.2f", threshold)
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                autoSaveSettings()
            }
        })
        
        // 关键点分析阈值滑块 -> 松手时自动保存
        dialogBinding.seekBarAnalysisThreshold.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val threshold = 0.3f + progress * 0.01f
                dialogBinding.tvAnalysisThresholdValue.text = String.format("%.2f", threshold)
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {
                autoSaveSettings()
            }
        })
        
        // 语言切换 -> 自动保存并重启
        dialogBinding.rgLanguage.setOnCheckedChangeListener { _, _ ->
            autoSaveSettings(andRecreate = true)
        }
        
        // 关闭按钮
        dialogBinding.btnCancel.text = getString(R.string.close)
        dialogBinding.btnCancel.setOnClickListener {
            dialog.dismiss()
        }
        
        // 隐藏保存按钮（已实现自动保存）
        dialogBinding.btnSave.visibility = View.GONE
        
        dialog.show()
    }
    
    /**
     * 请求存储权限并选择音频文件
     */
    private fun requestStoragePermissionAndPickAudio(requestCode: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ 使用 READ_MEDIA_AUDIO
            if (checkSelfPermission(Manifest.permission.READ_MEDIA_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                openAudioPicker(requestCode)
            } else {
                requestStoragePermissionLauncher.launch(Manifest.permission.READ_MEDIA_AUDIO)
            }
        } else {
            // Android 12 及以下使用 READ_EXTERNAL_STORAGE
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                openAudioPicker(requestCode)
            } else {
                requestStoragePermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
    }
    
    /**
     * 打开音频选择器
     */
    private fun openAudioPicker(requestCode: Int) {
        when (requestCode) {
            REQUEST_TIRED_AUDIO -> tiredAudioPicker.launch(arrayOf("audio/*"))
            REQUEST_SLIGHTLY_TIRED_AUDIO -> slightlyTiredAudioPicker.launch(arrayOf("audio/*"))
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
        binding.tvDriverState.text = viewModel.getDriverStateDisplayName(this, state.driverState)
        binding.tvDriverState.setTextColor(viewModel.getDriverStateColor(state.driverState))
        
        // 头部姿态 (国际化转换)
        val headPoseTexts = state.headPoses.map { poseName ->
            getHeadPoseDisplayName(poseName)
        }
        binding.tvHeadPose.text = if (headPoseTexts.isEmpty()) getString(R.string.unknown) else headPoseTexts.joinToString(", ")
        
        // 帧计数
        binding.tvFrameCount.text = getString(R.string.frame_count, state.frameCount)
        
        // 校准状态
        binding.tvCalibrationStatus.text = viewModel.getCalibrationStatusText(this)
        binding.tvCalibrationStatus.setTextColor(viewModel.getCalibrationStatusColor())
        
        // 手动旋转角度
        Log.d(TAG, "updateUI: manualRotation=${state.manualRotation}")
        binding.tvRotation.text = "${getString(R.string.rotation)}: ${state.manualRotation}°"
        
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
    
    /**
     * 获取头部姿态显示名称 (国际化)
     */
    private fun getHeadPoseDisplayName(poseName: String): String {
        return when (poseName) {
            "FACING_FORWARD" -> getString(R.string.head_pose_facing_forward)
            "HEAD_UP" -> getString(R.string.head_pose_head_up)
            "HEAD_DOWN" -> getString(R.string.head_pose_head_down)
            "HEAD_OFFSET" -> getString(R.string.head_pose_head_offset)
            "HEAD_TURNED" -> getString(R.string.head_pose_head_turned)
            "POSTURE_DEVIATION" -> getString(R.string.head_pose_posture_deviation)
            else -> poseName
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
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
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processFrame(imageProxy: ImageProxy) {
        if (detector == null || analyzer == null) {
            imageProxy.close()
            return
        }
        
        // 关闭上一帧的 ImageProxy（如果有）
        frameDataRef.getAndSet(null)?.imageProxy?.close()
        
        try {
            val nv21 = CameraUtils.imageProxyToNV21(imageProxy, nv21Buffer)
            nv21Buffer = nv21
            
            // 使用带错误处理的检测方法
            val detectResult = detector?.detectWithResult(nv21, imageProxy.width, imageProxy.height)
            
            // 处理检测结果
            detectResult?.let { result ->
                viewModel.processDetectResult(result)
                
                if (result is KeypointDetector.DetectResult.Success) {
                    // 保存帧数据（包含 ImageProxy，绘制后关闭）
                    frameDataRef.set(FrameData(result.keypoints, imageProxy))
                    
                    // 分析状态
                    val analysis = analyzer?.analyze(result.keypoints)
                    viewModel.processAnalysis(analysis)
                } else {
                    // 检测失败，立即关闭
                    imageProxy.close()
                    frameDataRef.set(null)
                }
            }
            
            // 只有成功检测时才触发绘制
            if (frameDataRef.get() != null) {
                runOnUiThread {
                    binding.overlayView.invalidate()
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
            imageProxy.close()
            frameDataRef.set(null)
        }
    }
    
    private fun allPermissionsGranted() = CameraUtils.hasCameraPermission(this)
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        KeypointDetector.releaseInstance()
    }
    
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        Log.d(TAG, "Configuration changed: ${if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE) "LANDSCAPE" else "PORTRAIT"}")
    }
}
