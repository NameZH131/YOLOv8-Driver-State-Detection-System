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
import kotlin.math.abs

/**
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
    bboxColor: Color = BboxColor
) {
    Canvas(modifier = modifier) {
        if (keypoints.isEmpty()) return@Canvas
        
        val viewWidth = size.width
        val viewHeight = size.height
        
        // 计算坐标变换
        val transform = calculateTransform(
            frameWidth, frameHeight,
            rotation, manualRotation,
            viewWidth, viewHeight
        )
        
        // 绘制边界框
        if (bbox != null && bbox.size >= 4) {
            val (x1, y1) = transform.apply(bbox[0], bbox[1])
            val (x2, y2) = transform.apply(bbox[2], bbox[3])
            
            drawRect(
                color = bboxColor,
                topLeft = Offset(x1, y1),
                size = Size(x2 - x1, y2 - y1),
                style = Stroke(width = 4f)
            )
        }
        
        // 绘制骨架线
        drawSkeleton(keypoints, transform, skeletonColor)
        
        // 绘制关键点
        keypoints.forEach { point ->
            if (point.confidence > 0.3f) {
                val (x, y) = transform.apply(point.x, point.y)
                
                drawCircle(
                    color = keypointColor,
                    radius = 8f,
                    center = Offset(x, y)
                )
            }
        }
    }
}

/**
 * 绘制骨架线
 */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSkeleton(
    keypoints: List<KeyPoint>,
    transform: CoordinateTransform,
    color: Color
) {
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
    
    skeletonConnections.forEach { (i, j) ->
        if (i < keypoints.size && j < keypoints.size) {
            val p1 = keypoints[i]
            val p2 = keypoints[j]
            
            if (p1.confidence > 0.3f && p2.confidence > 0.3f) {
                val (x1, y1) = transform.apply(p1.x, p1.y)
                val (x2, y2) = transform.apply(p2.x, p2.y)
                
                drawLine(
                    color = color,
                    start = Offset(x1, y1),
                    end = Offset(x2, y2),
                    strokeWidth = 3f
                )
            }
        }
    }
}

/**
 * 坐标变换器
 */
data class CoordinateTransform(
    private val scaleX: Float,
    private val scaleY: Float,
    private val offsetX: Float,
    private val offsetY: Float,
    private val flipX: Boolean = false,
    private val flipY: Boolean = false
) {
    fun apply(x: Float, y: Float): Pair<Float, Float> {
        val transformedX = if (flipX) offsetX - x * scaleX else offsetX + x * scaleX
        val transformedY = if (flipY) offsetY - y * scaleY else offsetY + y * scaleY
        return transformedX to transformedY
    }
}

/**
 * 计算坐标变换矩阵
 */
private fun calculateTransform(
    frameWidth: Int,
    frameHeight: Int,
    rotation: Int,
    manualRotation: Int,
    viewWidth: Float,
    viewHeight: Float
): CoordinateTransform {
    // 计算有效旋转角度
    val effectiveRotation = (rotation + manualRotation).let { 
        when {
            it < 0 -> it + 360
            it >= 360 -> it - 360
            else -> it
        }
    }
    
    // 根据旋转角度确定帧的有效尺寸
    val isRotated = effectiveRotation == 90 || effectiveRotation == 270
    val effectiveFrameWidth = if (isRotated) frameHeight else frameWidth
    val effectiveFrameHeight = if (isRotated) frameWidth else frameHeight
    
    // 计算缩放以保持宽高比
    val scale = minOf(
        viewWidth / effectiveFrameWidth,
        viewHeight / effectiveFrameHeight
    )
    
    val scaledWidth = effectiveFrameWidth * scale
    val scaledHeight = effectiveFrameHeight * scale
    
    // 居中偏移
    val offsetX = (viewWidth - scaledWidth) / 2
    val offsetY = (viewHeight - scaledHeight) / 2
    
    return CoordinateTransform(
        scaleX = scale,
        scaleY = scale,
        offsetX = offsetX,
        offsetY = offsetY,
        flipX = effectiveRotation == 180 || effectiveRotation == 270,
        flipY = effectiveRotation == 90 || effectiveRotation == 180
    )
}
