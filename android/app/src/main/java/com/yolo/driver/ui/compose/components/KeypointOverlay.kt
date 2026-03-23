package com.yolo.driver.ui.compose.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import com.yolo.driver.analyzer.KeypointDetector.KeyPoint
import com.yolo.driver.ui.compose.theme.BboxColor
import com.yolo.driver.ui.compose.theme.KeypointColor
import com.yolo.driver.ui.compose.theme.SkeletonColor

/**
 * @writer: zhangheng
 * 关键点覆盖层 Composable
 * 使用 Canvas 绘制关键点和骨架
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
    confidenceThreshold: Float = 0.3f
) {
    Canvas(modifier = modifier) {
        if (keypoints.isEmpty()) return@Canvas
        
        val viewWidth = size.width
        val viewHeight = size.height
        
        // 使用手动旋转值（支持 0/90/180/270/360）
        // 360 和 0 效果相同
        val effectiveRotation = if (manualRotation == 360) 0 else manualRotation
        
        // 根据旋转角度确定帧的有效尺寸
        val isRotated = effectiveRotation == 90 || effectiveRotation == 270
        val effectiveFrameWidth = if (isRotated) frameHeight else frameWidth
        val effectiveFrameHeight = if (isRotated) frameWidth else frameHeight
        
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
        
        // 转换坐标的函数
        fun transformPoint(kp: KeyPoint): Pair<Float, Float>? {
            if (kp.confidence < confidenceThreshold) return null
            
            var x: Float
            var y: Float
            
            when (effectiveRotation) {
                90 -> {
                    // 顺时针90度: (x, y) -> (height - y, x)
                    x = frameHeight - kp.y
                    y = kp.x
                }
                180 -> {
                    // 180度: (x, y) -> (width - x, height - y)
                    x = frameWidth - kp.x
                    y = frameHeight - kp.y
                }
                270 -> {
                    // 顺时针270度(逆时针90度): (x, y) -> (y, width - x)
                    x = kp.y
                    y = frameWidth - kp.x
                }
                else -> {
                    // 0度: 不变
                    x = kp.x
                    y = kp.y
                }
            }
            
            // 镜像（前置摄像头）
            if (mirror) {
                x = effectiveFrameWidth - x
            }
            
            // 缩放和偏移
            x = x * scale + offsetX
            y = y * scale + offsetY
            
            return Pair(x, y)
        }
        
        // 绘制边界框
        if (bbox != null && bbox.size >= 4) {
            // TODO: 边界框也需要根据旋转转换
        }
        
        // YOLOv8-Pose 骨架连接定义
        val skeletonConnections = listOf(
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