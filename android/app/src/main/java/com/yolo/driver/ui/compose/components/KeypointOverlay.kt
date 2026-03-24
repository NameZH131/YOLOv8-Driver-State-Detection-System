package com.yolo.driver.ui.compose.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import com.yolo.driver.analyzer.KeypointDetector.KeyPoint
import com.yolo.driver.ui.compose.theme.BboxColor
import com.yolo.driver.ui.compose.theme.KeypointColor
import com.yolo.driver.ui.compose.theme.SkeletonColor

/**
 * @writer: zhangheng
 * 关键点覆盖层 Composable
 * 使用 Canvas 绘制关键点和骨架
 * 
 * 性能优化：
 * 1. 使用 derivedStateOf 避免不必要的重组
 * 2. 使用 remember 缓存计算参数
 * 3. 仅在相关参数变化时重新计算
 */
@Composable
fun KeypointOverlay(
    keypoints: List<KeyPoint>,
    frameWidth: Int,
    frameHeight: Int,
    rotation: Int,
    manualRotation: Int,
    bbox: FloatArray? = null,
    confidence: Float = 0f,
    modifier: Modifier = Modifier,
    keypointColor: Color = KeypointColor,
    skeletonColor: Color = SkeletonColor,
    bboxColor: Color = BboxColor,
    mirror: Boolean = true,  // 前置摄像头镜像
    confidenceThreshold: Float = 0.3f,
    // 相机传感器旋转角度（由 CameraX 提供）
    rotationDegrees: Int = 0
) {
    // 组合旋转角度：相机传感器旋转 + 用户手动调整
    val totalRotation = remember(rotationDegrees, manualRotation) {
        (rotationDegrees + manualRotation) % 360
    }
    
    // 根据旋转角度确定帧的有效尺寸（仅在旋转变化时重新计算）
    val isRotated = remember(totalRotation) {
        totalRotation == 90 || totalRotation == 270
    }
    val effectiveFrameWidth = remember(frameWidth, frameHeight, isRotated) {
        if (isRotated) frameHeight else frameWidth
    }
    val effectiveFrameHeight = remember(frameWidth, frameHeight, isRotated) {
        if (isRotated) frameWidth else frameHeight
    }
    
    // YOLOv8-Pose 骨架连接定义（静态常量）
    val skeletonConnections = remember {
        listOf(
            // 头部
            0 to 1, 0 to 2,  // 鼻子-眼睛
            1 to 3, 2 to 4,  // 眼睛-耳朵
            // 躯干
            5 to 6,          // 肩膀
            5 to 11, 6 to 12, // 肩膀-髋部
            11 to 12,        // 髋部
            // 左臂
            5 to 7, 7 to 9,
            // 右臂
            6 to 8, 8 to 10,
            // 左腿
            11 to 13, 13 to 15,
            // 右腿
            12 to 14, 14 to 16
        )
    }
    
    Canvas(modifier = modifier) {
        if (keypoints.isEmpty()) return@Canvas
        
        val viewWidth = size.width
        val viewHeight = size.height
        
        // FILL_CENTER: 取最大缩放比填充屏幕（与 PreviewView 一致）
        val scale = maxOf(
            viewWidth / effectiveFrameWidth,
            viewHeight / effectiveFrameHeight
        )
        
        val scaledWidth = effectiveFrameWidth * scale
        val scaledHeight = effectiveFrameHeight * scale
        
        // 居中偏移（FILL_CENTER 模式下可能是负值，表示裁剪区域）
        val offsetX = (viewWidth - scaledWidth) / 2
        val offsetY = (viewHeight - scaledHeight) / 2
        
        // 转换坐标的函数（内联优化）
        fun transformPoint(kp: KeyPoint): Pair<Float, Float>? {
            if (kp.confidence < confidenceThreshold) return null
            
            var x = kp.x
            var y = kp.y
            
            // 1. 先镜像（前置摄像头，基于原始坐标系）
            if (mirror) {
                x = frameWidth - x
            }
            
            // 2. 再旋转
            when (totalRotation) {
                90 -> {
                    // 顺时针90度: (x, y) -> (height - y, x)
                    val tmp = frameHeight - y
                    y = x
                    x = tmp
                }
                180 -> {
                    // 180度: (x, y) -> (width - x, height - y)
                    x = frameWidth - x
                    y = frameHeight - y
                }
                270 -> {
                    // 顺时针270度(逆时针90度): (x, y) -> (y, width - x)
                    val tmp = y
                    y = frameWidth - x
                    x = tmp
                }
                // 0度: 不变
            }
            
            // 3. 缩放和偏移
            x = x * scale + offsetX
            y = y * scale + offsetY
            
            return Pair(x, y)
        }
        
        // 预先转换所有关键点
        val transformedPoints = keypoints.map { transformPoint(it) }
        
        // 绘制骨架线
        skeletonConnections.forEach { (i, j) ->
            if (i < transformedPoints.size && j < transformedPoints.size) {
                val p1 = transformedPoints[i]
                val p2 = transformedPoints[j]
                
                if (p1 != null && p2 != null) {
                    drawLine(
                        color = skeletonColor,
                        start = Offset(p1.first, p1.second),
                        end = Offset(p2.first, p2.second),
                        strokeWidth = 3f
                    )
                }
            }
        }
        
        // 绘制关键点
        transformedPoints.forEach { point ->
            point?.let { (x, y) ->
                drawCircle(
                    color = keypointColor,
                    radius = 8f,
                    center = Offset(x, y)
                )
            }
        }
    }
}