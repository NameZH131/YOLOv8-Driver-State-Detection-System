package com.yolo.driver

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalContext
import com.yolo.driver.ui.compose.AppNavigation
import com.yolo.driver.ui.compose.theme.DriverMonitorTheme

/**
 * @writer: zhangheng
 * 主 Activity - Compose 容器
 * 使用单 Activity 架构，所有页面通过 Compose Navigation 管理
 */
class MainActivity : ComponentActivity() {
    
    override fun attachBaseContext(newBase: Context) {
        // 应用语言配置
        val app = newBase.applicationContext as? DriverApplication
        val context = app?.applyLanguageToContext(newBase) ?: newBase
        super.attachBaseContext(context)
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            DriverMonitorTheme(darkTheme = true) {
                AppNavigation(
                    onExitApp = { finish() }
                )
            }
        }
    }
}