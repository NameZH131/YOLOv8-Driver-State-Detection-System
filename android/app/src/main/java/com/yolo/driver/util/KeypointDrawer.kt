package com.yolo.driver.util

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import androidx.camera.core.ImageProxy
import com.yolo.driver.analyzer.KeypointDetector

/**
 * 关键点绘制工具类
 * 简化版：直接按比例映射坐标
 * YOLO 输入尺寸 640x640，直接映射到 PreviewView 尺寸
 */
object KeypointDrawer {
    
    private const val TAG = "KeypointDrawer"
    private const val YOLO_INPUT_SIZE = 640
    
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
     * 简化版关键点绘制
     * 直接按比例映射：YOLO 坐标 -> PreviewView 坐标
     * 
     * @param canvas 画布
     * @param keypoints 关键点列表（坐标已在原图尺寸范围）
     * @param imageProxy 图像代理（用于获取尺寸信息）
     * @param viewWidth PreviewView 宽度
     * @param viewHeight PreviewView 高度
     * @param pointPaint 点绘制配置
     * @param linePaint 线绘制配置
     * @param manualRotation 手动旋转角度 (0/90/180/270)
     * @param mirror 是否镜像（前置摄像头）
     */
    @androidx.camera.core.ExperimentalGetImage
    fun drawKeypoints(
        canvas: Canvas,
        keypoints: List<KeypointDetector.KeyPoint>,
        imageProxy: ImageProxy,
        viewWidth: Int,
        viewHeight: Int,
        pointPaint: Paint = DEFAULT_POINT_PAINT,
        linePaint: Paint = DEFAULT_LINE_PAINT,
        manualRotation: Int = 0,
        mirror: Boolean = true
    ) {
        if (keypoints.isEmpty()) return
        
        // 获取图像尺寸
        val imageWidth = imageProxy.width
        val imageHeight = imageProxy.height
        
        Log.d(TAG, "drawKeypoints: image=${imageWidth}x${imageHeight}, view=${viewWidth}x${viewHeight}, rotation=$manualRotation, mirror=$mirror")
        
        // 根据旋转角度确定显示尺寸
        // rotation 0/180: 显示尺寸 = 图像尺寸
        // rotation 90/270: 显示尺寸 = 图像尺寸交换 (width <-> height)
        val displayWidth = if (manualRotation == 90 || manualRotation == 270) imageHeight else imageWidth
        val displayHeight = if (manualRotation == 90 || manualRotation == 270) imageWidth else imageHeight
        
        // 计算缩放比例（保持宽高比）
        val scaleX = viewWidth.toFloat() / displayWidth
        val scaleY = viewHeight.toFloat() / displayHeight
        val scale = minOf(scaleX, scaleY)
        
        // 计算偏移（居中）
        val scaledWidth = displayWidth * scale
        val scaledHeight = displayHeight * scale
        val offsetX = (viewWidth - scaledWidth) / 2f
        val offsetY = (viewHeight - scaledHeight) / 2f
        
        Log.d(TAG, "display=${displayWidth}x${displayHeight}, scale=$scale, offset=($offsetX, $offsetY)")
        
        // 转换坐标
        val transformedPoints = keypoints.mapIndexed { index, kp ->
            if (kp.confidence > 0.5f) {
                // 1. 先根据旋转角度转换坐标
                var x: Float
                var y: Float
                
                when (manualRotation) {
                    90 -> {
                        // 顺时针90度: (x, y) -> (height - y, x)
                        x = imageHeight - kp.y
                        y = kp.x
                    }
                    180 -> {
                        // 180度: (x, y) -> (width - x, height - y)
                        x = imageWidth - kp.x
                        y = imageHeight - kp.y
                    }
                    270 -> {
                        // 顺时针270度(逆时针90度): (x, y) -> (y, width - x)
                        x = kp.y
                        y = imageWidth - kp.x
                    }
                    else -> {
                        // 0度: 不变
                        x = kp.x
                        y = kp.y
                    }
                }
                
                // 2. 镜像（前置摄像头）
                if (mirror) {
                    x = displayWidth - x
                }
                
                // 3. 缩放和偏移
                x = x * scale + offsetX
                y = y * scale + offsetY
                
                Log.d(TAG, "KP$index: (${kp.x}, ${kp.y}) --rot$manualRotation--> ($x, $y)")
                Pair(x, y)
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
     * 兼容旧接口
     */
    @androidx.camera.core.ExperimentalGetImage
    fun drawKeypointsWithTransform(
        canvas: Canvas,
        keypoints: List<KeypointDetector.KeyPoint>,
        imageProxy: ImageProxy,
        previewView: androidx.camera.view.PreviewView,
        pointPaint: Paint = DEFAULT_POINT_PAINT,
        linePaint: Paint = DEFAULT_LINE_PAINT,
        manualRotation: Int = 0,
        mirror: Boolean = true
    ) {
        drawKeypoints(
            canvas, keypoints, imageProxy,
            previewView.width, previewView.height,
            pointPaint, linePaint, manualRotation, mirror
        )
    }
}