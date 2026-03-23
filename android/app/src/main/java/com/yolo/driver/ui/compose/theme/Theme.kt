package com.yolo.driver.ui.compose.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// 深色主题配色方案
private val DarkColorScheme = darkColorScheme(
    primary = Primary,
    onPrimary = OnSurface,
    primaryContainer = PrimaryDark,
    onPrimaryContainer = OnSurface,
    secondary = Secondary,
    onSecondary = OnSurface,
    secondaryContainer = SecondaryDark,
    onSecondaryContainer = OnSurface,
    background = Background,
    onBackground = OnBackground,
    surface = Surface,
    onSurface = OnSurface,
    error = Error,
    onError = OnSurface
)

// 浅色主题配色方案
private val LightColorScheme = lightColorScheme(
    primary = Primary,
    onPrimary = OnBackgroundLight,
    primaryContainer = PrimaryLight,
    onPrimaryContainer = OnBackgroundLight,
    secondary = Secondary,
    onSecondary = OnBackgroundLight,
    secondaryContainer = Secondary,
    onSecondaryContainer = OnBackgroundLight,
    background = BackgroundLight,
    onBackground = OnBackgroundLight,
    surface = SurfaceLight,
    onSurface = OnSurfaceLight,
    error = Error,
    onError = OnBackgroundLight
)

/**
 * 应用主题
 */
@Composable
fun DriverMonitorTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // 动态颜色 (Android 12+)
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        content = content
    )
}
