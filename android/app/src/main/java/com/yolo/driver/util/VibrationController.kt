package com.yolo.driver.util

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

/**
 * @writer: zhangheng
 * 震动控制器
 * 封装系统震动服务，支持多种震动模式
 */
class VibrationController(private val context: Context) {
    
    companion object {
        private const val TAG = "VibrationController"
        
        // 预设震动模式
        val PATTERN_SHORT = longArrayOf(100)  // 短震
        val PATTERN_LONG = longArrayOf(300)   // 长震
        val PATTERN_DOUBLE = longArrayOf(100, 100, 100)  // 双击
        val PATTERN_PULSE = longArrayOf(0, 200, 100, 200, 100, 200)  // 脉冲
    }
    
    /**
     * 震动模式枚举
     */
    enum class VibrationMode(val value: Int, val pattern: LongArray) {
        SHORT(0, PATTERN_SHORT),
        LONG(1, PATTERN_LONG),
        DOUBLE(2, PATTERN_DOUBLE),
        PULSE(3, PATTERN_PULSE);
        
        companion object {
            fun fromValue(value: Int): VibrationMode {
                return values().find { it.value == value } ?: SHORT
            }
        }
    }
    
    // 震动服务
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        vibratorManager?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }
    
    // 震动启用状态 (默认开启)
    private var vibrationEnabled: Boolean = true
    
    // 当前震动模式
    private var currentMode: VibrationMode = VibrationMode.SHORT
    
    // 震动冷却时间
    private var lastVibrateTime: Long = 0
    private var cooldownMs: Long = 500L  // 缩短冷却时间到500ms
    
    /**
     * 触发震动 (使用当前模式)
     */
    fun vibrate() {
        vibrate(currentMode)
    }
    
    /**
     * 触发指定模式的震动
     */
    fun vibrate(mode: VibrationMode) {
        Log.d(TAG, "vibrate called: mode=${mode.name}, vibrationEnabled=$vibrationEnabled")
        
        if (!vibrationEnabled) {
            Log.d(TAG, "Vibration disabled, skip")
            return
        }
        
        // 检查设备是否支持震动
        if (vibrator == null || !vibrator.hasVibrator()) {
            Log.w(TAG, "Device does not support vibration")
            return
        }
        
        // 冷却检查
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVibrateTime < cooldownMs) {
            Log.d(TAG, "In cooldown period, skip vibration")
            return
        }
        
        try {
            Log.d(TAG, "Executing vibration with pattern=${mode.pattern.toList()}")
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // Android 8.0+ 使用 VibrationEffect
                val effect = if (mode.pattern.size == 1) {
                    // 单次震动
                    VibrationEffect.createOneShot(mode.pattern[0], VibrationEffect.DEFAULT_AMPLITUDE)
                } else {
                    // 波形震动
                    VibrationEffect.createWaveform(mode.pattern, -1)
                }
                vibrator.vibrate(effect)
            } else {
                // 旧版本 API
                @Suppress("DEPRECATION")
                vibrator.vibrate(mode.pattern, -1)
            }
            
            lastVibrateTime = currentTime
            Log.d(TAG, "Vibration triggered with mode=${mode.name}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to vibrate", e)
        }
    }
    
    /**
     * 短震动
     */
    fun shortVibrate() {
        vibrate(VibrationMode.SHORT)
    }
    
    /**
     * 长震动
     */
    fun longVibrate() {
        vibrate(VibrationMode.LONG)
    }
    
    /**
     * 双击震动
     */
    fun doubleVibrate() {
        vibrate(VibrationMode.DOUBLE)
    }
    
    /**
     * 脉冲震动
     */
    fun pulseVibrate() {
        vibrate(VibrationMode.PULSE)
    }
    
    /**
     * 设置震动启用状态
     */
    fun setEnabled(enabled: Boolean) {
        vibrationEnabled = enabled
        Log.d(TAG, "Vibration enabled: $enabled")
    }
    
    /**
     * 获取震动启用状态
     */
    fun isEnabled(): Boolean = vibrationEnabled
    
    /**
     * 设置震动模式
     */
    fun setMode(mode: VibrationMode) {
        currentMode = mode
        Log.d(TAG, "Vibration mode set to: ${mode.name}")
    }
    
    /**
     * 设置震动模式 (通过整数值)
     */
    fun setMode(modeValue: Int) {
        setMode(VibrationMode.fromValue(modeValue))
    }
    
    /**
     * 获取当前震动模式
     */
    fun getMode(): VibrationMode = currentMode
    
    /**
     * 设置冷却时间
     */
    fun setCooldown(ms: Long) {
        cooldownMs = ms
    }
    
    /**
     * 取消震动
     */
    fun cancel() {
        vibrator?.cancel()
    }
}