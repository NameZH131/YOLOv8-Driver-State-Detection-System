package com.yolo.driver

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import java.util.Locale

/**
 * @writer: zhangheng
 * 全局 Application 类
 * 处理全局语言设置，确保所有 Activity 使用统一语言
 */
class DriverApplication : Application() {
    
    companion object {
        const val PREFS_NAME = "driver_monitor_prefs"
        const val KEY_LANGUAGE_MODE = "language_mode"
        
        // 姿态映射 key
        const val KEY_FRAME_HEAD_UP_DOWN = "frame_head_up_down"
        const val KEY_FRAME_HEAD_LEFT_RIGHT = "frame_head_left_right"
        const val KEY_FRAME_POSTURE_DEVIATION = "frame_posture_deviation"
        const val KEY_SLIDING_HEAD_UP_DOWN = "sliding_head_up_down"
        const val KEY_SLIDING_HEAD_LEFT_RIGHT = "sliding_head_left_right"
        const val KEY_SLIDING_POSTURE_DEVIATION = "sliding_posture_deviation"
        
        // 完整设置 key
        const val KEY_VIBRATION_ENABLED = "vibration_enabled"
        const val KEY_VIBRATION_MODE = "vibration_mode"
        const val KEY_AUDIO_ENABLED = "audio_enabled"
        const val KEY_AUDIO_VOLUME = "audio_volume"
        const val KEY_TIRED_AUDIO_URI = "tired_audio_uri"
        const val KEY_SLIGHTLY_TIRED_AUDIO_URI = "slightly_tired_audio_uri"
        const val KEY_WINDOW_DURATION_MS = "window_duration_ms"
        const val KEY_SLIDING_WINDOW_MODE = "sliding_window_mode"
        const val KEY_DRAW_THRESHOLD = "draw_threshold"
        const val KEY_ANALYSIS_THRESHOLD = "analysis_threshold"
        const val KEY_ALERT_REPEAT_MODE = "alert_repeat_mode"
        
        // 姿态状态值
        const val STATE_NORMAL = 0
        const val STATE_SLIGHTLY_TIRED = 1
        const val STATE_TIRED = 2
        
        // 提醒重复模式常量
        const val ALERT_REPEAT_ONCE = 0    // 状态变化时只播放一次
        const val ALERT_REPEAT_CONTINUOUS = 1  // 持续播放
        
        var instance: DriverApplication? = null
        
        /**
         * 保存语言设置到 SharedPreferences
         * @param context Context
         * @param languageMode 语言模式: 0=自动, 1=中文, 2=英文
         */
        fun saveLanguageMode(context: Context, languageMode: Int) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().putInt(KEY_LANGUAGE_MODE, languageMode).apply()
        }
        
        /**
         * 获取当前保存的语言设置
         * @param context Context
         * @return 语言模式: 0=自动, 1=中文, 2=英文
         */
        fun getLanguageMode(context: Context): Int {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return prefs.getInt(KEY_LANGUAGE_MODE, 0)
        }
        
        /**
         * 保存姿态映射设置
         */
        fun savePoseMappings(
            context: Context,
            frameMapping: PoseMappingData,
            slidingMapping: PoseMappingData
        ) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit()
                .putInt(KEY_FRAME_HEAD_UP_DOWN, frameMapping.headUpDown)
                .putInt(KEY_FRAME_HEAD_LEFT_RIGHT, frameMapping.headLeftRight)
                .putInt(KEY_FRAME_POSTURE_DEVIATION, frameMapping.postureDeviation)
                .putInt(KEY_SLIDING_HEAD_UP_DOWN, slidingMapping.headUpDown)
                .putInt(KEY_SLIDING_HEAD_LEFT_RIGHT, slidingMapping.headLeftRight)
                .putInt(KEY_SLIDING_POSTURE_DEVIATION, slidingMapping.postureDeviation)
                .apply()
        }
        
        /**
         * 加载逐帧模式姿态映射
         */
        fun getFramePoseMapping(context: Context): PoseMappingData {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return PoseMappingData(
                headUpDown = prefs.getInt(KEY_FRAME_HEAD_UP_DOWN, STATE_TIRED),
                headLeftRight = prefs.getInt(KEY_FRAME_HEAD_LEFT_RIGHT, STATE_NORMAL),
                postureDeviation = prefs.getInt(KEY_FRAME_POSTURE_DEVIATION, STATE_TIRED)
            )
        }
        
        /**
         * 加载滑动窗模式姿态映射
         */
        fun getSlidingPoseMapping(context: Context): PoseMappingData {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return PoseMappingData(
                headUpDown = prefs.getInt(KEY_SLIDING_HEAD_UP_DOWN, STATE_TIRED),
                headLeftRight = prefs.getInt(KEY_SLIDING_HEAD_LEFT_RIGHT, STATE_NORMAL),
                postureDeviation = prefs.getInt(KEY_SLIDING_POSTURE_DEVIATION, STATE_TIRED)
            )
        }
        
        /**
         * 保存完整设置
         */
        fun saveAllSettings(
            context: Context,
            vibrationEnabled: Boolean,
            vibrationMode: Int,
            audioEnabled: Boolean,
            audioVolume: Int,
            tiredAudioUri: String?,
            slightlyTiredAudioUri: String?,
            windowDurationMs: Long,
            isSlidingWindowMode: Boolean,
            drawThreshold: Float,
            analysisThreshold: Float,
            alertRepeatMode: Int = ALERT_REPEAT_ONCE
        ) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit()
                .putBoolean(KEY_VIBRATION_ENABLED, vibrationEnabled)
                .putInt(KEY_VIBRATION_MODE, vibrationMode)
                .putBoolean(KEY_AUDIO_ENABLED, audioEnabled)
                .putInt(KEY_AUDIO_VOLUME, audioVolume)
                .putString(KEY_TIRED_AUDIO_URI, tiredAudioUri)
                .putString(KEY_SLIGHTLY_TIRED_AUDIO_URI, slightlyTiredAudioUri)
                .putLong(KEY_WINDOW_DURATION_MS, windowDurationMs)
                .putBoolean(KEY_SLIDING_WINDOW_MODE, isSlidingWindowMode)
                .putFloat(KEY_DRAW_THRESHOLD, drawThreshold)
                .putFloat(KEY_ANALYSIS_THRESHOLD, analysisThreshold)
                .putInt(KEY_ALERT_REPEAT_MODE, alertRepeatMode)
                .apply()
        }
        
        /**
         * 加载所有设置
         */
        fun loadAllSettings(context: Context): AllSettingsData {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return AllSettingsData(
                vibrationEnabled = prefs.getBoolean(KEY_VIBRATION_ENABLED, true),
                vibrationMode = prefs.getInt(KEY_VIBRATION_MODE, 0),
                audioEnabled = prefs.getBoolean(KEY_AUDIO_ENABLED, true),
                audioVolume = prefs.getInt(KEY_AUDIO_VOLUME, 100),
                tiredAudioUri = prefs.getString(KEY_TIRED_AUDIO_URI, null),
                slightlyTiredAudioUri = prefs.getString(KEY_SLIGHTLY_TIRED_AUDIO_URI, null),
                windowDurationMs = prefs.getLong(KEY_WINDOW_DURATION_MS, 5000L),
                isSlidingWindowMode = prefs.getBoolean(KEY_SLIDING_WINDOW_MODE, false),
                drawThreshold = prefs.getFloat(KEY_DRAW_THRESHOLD, 0.5f),
                analysisThreshold = prefs.getFloat(KEY_ANALYSIS_THRESHOLD, 0.5f),
                alertRepeatMode = prefs.getInt(KEY_ALERT_REPEAT_MODE, ALERT_REPEAT_ONCE)
            )
        }
    }
    
    /**
     * 完整设置数据类
     */
    data class AllSettingsData(
        val vibrationEnabled: Boolean = true,
        val vibrationMode: Int = 0,
        val audioEnabled: Boolean = true,
        val audioVolume: Int = 100,
        val tiredAudioUri: String? = null,
        val slightlyTiredAudioUri: String? = null,
        val windowDurationMs: Long = 5000L,
        val isSlidingWindowMode: Boolean = false,
        val drawThreshold: Float = 0.5f,
        val analysisThreshold: Float = 0.5f,
        val alertRepeatMode: Int = ALERT_REPEAT_CONTINUOUS  // 0=只播放一次, 1=持续播放（默认）
    )
    
    /**
     * 姿态映射数据类
     */
    data class PoseMappingData(
        val headUpDown: Int = STATE_TIRED,
        val headLeftRight: Int = STATE_NORMAL,
        val postureDeviation: Int = STATE_TIRED
    )
    
    override fun attachBaseContext(base: Context) {
        instance = this
        // 应用语言配置后再调用 super
        val wrappedContext = applyLanguageConfig(base)
        super.attachBaseContext(wrappedContext)
    }
    
    override fun onCreate() {
        super.onCreate()
        instance = this
    }
    
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // 重新应用用户选择的语言配置，防止跟随系统语言变化
        applyLanguageConfig(this)
    }
    
    /**
     * 应用语言配置到 Context
     */
    fun applyLanguageToContext(context: Context): Context {
        return applyLanguageConfig(context)
    }
    
    /**
     * 应用语言配置
     */
    private fun applyLanguageConfig(context: Context): Context {
        val prefs = try {
            context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        } catch (e: Exception) {
            return context
        }
        
        val languageMode = try {
            prefs.getInt(KEY_LANGUAGE_MODE, 0)
        } catch (e: Exception) {
            return context
        }
        
        val locale = when (languageMode) {
            1 -> Locale.CHINESE
            2 -> Locale.ENGLISH
            else -> Locale.getDefault()
        }
        
        Locale.setDefault(locale)
        
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        
        return context.createConfigurationContext(config)
    }
}
