package com.yolo.driver.ui.compose

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

/**
 * @writer: zhangheng
 * 应用导航配置
 */
sealed class Screen(val route: String) {
    object Main : Screen("main")
    object Calibration : Screen("calibration")
}

/**
 * 应用导航主机
 */
@Composable
fun AppNavigation(
    navController: NavHostController = rememberNavController(),
    onExitApp: () -> Unit = {}
) {
    NavHost(
        navController = navController,
        startDestination = Screen.Main.route
    ) {
        composable(Screen.Main.route) {
            MainScreen(
                onNavigateToCalibration = {
                    navController.navigate(Screen.Calibration.route)
                },
                onExitApp = onExitApp
            )
        }
        
        composable(Screen.Calibration.route) {
            CalibrationScreen(
                onNavigateBack = {
                    navController.popBackStack()
                },
                onCalibrationComplete = {
                    // 返回主界面并刷新校准状态
                    navController.popBackStack()
                }
            )
        }
    }
}
