package com.yolo.driver.util

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.yolo.driver.analyzer.KeypointDetector

/**
 * 关键点绘制工具类
 * 统一处理 YOLOv8-Pose 17 关键点的坐标转换和绘制
 */
object KeypointDrawer {
    
    // 骨骼连接关系 (YOLOv8-Pose 17点)
    private val CONNECTIONS = listOf(
        0 to 1, 0 to 2,  // nose -> eyes
        1 to 3, 2 to 4,  // eyes -> ears
        5 to 6,          // shoulders
        5 to 7, 7 to 9,  // left arm
        6 to 8, 8 to 10, // right arm
        5 to 11, 6 to 12, // shoulders -> hips
        11 to 12,        // hips
        11 to 13, 13 to 15, // left leg
        12 to 14, 14 to 16  // right leg
    )
    
    // 默认 Paint 配置
    private val DEFAULT_POINT_PAINT = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val DEFAULT_LINE_PAINT = Paint().apply {
        color = Color.GREEN
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    /**
     * 绘制关键点和骨骼连线
     * 
     * @param canvas 画布
     * @param keypoints 关键点列表
     * @param frameWidth 原始帧宽度
     * @param frameHeight 原始帧高度
     * @param rotationDegrees 旋转角度 (0, 90, 180, 270)
     * @param viewWidth 视图宽度
     * @param viewHeight 视图高度
     * @param pointPaint 点绘制配置 (可选)
     * @param linePaint 线绘制配置 (可选)
     * @param manualRotation 手动旋转覆盖 (调试用，默认 -1 表示不覆盖)
     */
    fun drawKeypoints(
        canvas: Canvas,
        keypoints: List<KeypointDetector.KeyPoint>,
        frameWidth: Int,
        frameHeight: Int,
        rotationDegrees: Int,
        viewWidth: Int,
        viewHeight: Int,
        pointPaint: Paint = DEFAULT_POINT_PAINT,
        linePaint: Paint = DEFAULT_LINE_PAINT,
        manualRotation: Int = -1
    ) {
        if (keypoints.isEmpty()) return
        
        // 使用手动旋转覆盖（调试模式）
        val effectiveRotation = if (manualRotation >= 0) manualRotation else rotationDegrees
        
        // 根据旋转角度计算坐标转换
        val isPortrait = effectiveRotation == 270 || effectiveRotation == 90
        
        val (scaleX, scaleY) = if (isPortrait) {
            viewWidth.toFloat() / frameHeight to viewHeight.toFloat() / frameWidth
        } else {
            viewWidth.toFloat() / frameWidth to viewHeight.toFloat() / frameHeight
        }
        
        // 转换所有有效关键点坐标
        val transformedPoints = keypoints.map { kp ->
            if (kp.confidence > 0.5f) {
                transformPoint(kp, frameWidth, frameHeight, effectiveRotation, scaleX, scaleY, viewWidth)
            } else {
                null
            }
        }
        
        // 绘制骨骼连线
        for ((i, j) in CONNECTIONS) {
            if (i < transformedPoints.size && j < transformedPoints.size) {
                val p1 = transformedPoints[i]
                val p2 = transformedPoints[j]
                if (p1 != null && p2 != null) {
                    canvas.drawLine(p1.first, p1.second, p2.first, p2.second, linePaint)
                }
            }
        }
        
        // 绘制关键点
        for (point in transformedPoints) {
            point?.let { (x, y) ->
                canvas.drawCircle(x, y, 8f, pointPaint)
            }
        }
    }
    
    /**
     * 单点坐标转换
     */
    private fun transformPoint(
        kp: KeypointDetector.KeyPoint,
        frameWidth: Int,
        frameHeight: Int,
        rotationDegrees: Int,
        scaleX: Float,
        scaleY: Float,
        viewWidth: Int
    ): Pair<Float, Float> {
        var x = kp.x
        var y = kp.y
        
        // 根据旋转角度调整坐标
        when (rotationDegrees) {
            90 -> {
                val tmp = x
                x = frameHeight - y
                y = tmp
            }
            180 -> {
                x = frameWidth - x
                y = frameHeight - y
            }
            270 -> {
                val tmp = x
                x = y
                y = frameWidth - tmp
            }
        }
        
        // 缩放到视图大小
        var screenX = x * scaleX
        val screenY = y * scaleY
        
        // 前置摄像头镜像
        screenX = viewWidth - screenX
        
        return screenX to screenY
    }
}
