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
        
        var instance: DriverApplication? = null
    }
    
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
