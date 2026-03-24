package com.yolo.driver.util

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import android.content.Context

/**
 * 相机相关工具类
 * 提供NV21转换、权限检查等通用功能
 */
object CameraUtils {
    
    private const val TAG = "CameraUtils"
    
    /**
     * 将 ImageProxy 转换为 NV21 格式字节数组
     * 正确处理 YUV 格式的 rowStride 和 pixelStride
     * 分别处理 U 和 V planes 的内存布局
     * 
     * @param image CameraX ImageProxy
     * @param reuseBuffer 可复用的buffer（可为null）
     * @return NV21 字节数组
     */
    fun imageProxyToNV21(
        image: ImageProxy,
        reuseBuffer: ByteArray? = null
    ): ByteArray {
        val width = image.width
        val height = image.height
        
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        
        // 分别获取 Y, U, V planes 的参数
        val yRowStride = image.planes[0].rowStride
        val uRowStride = image.planes[1].rowStride
        val uPixelStride = image.planes[1].pixelStride
        val vRowStride = image.planes[2].rowStride
        val vPixelStride = image.planes[2].pixelStride
        
        // 输出调试信息
        Log.d(TAG, "Image size: ${width}x${height}, yRowStride=$yRowStride, " +
                   "uRowStride=$uRowStride, uPixelStride=$uPixelStride, " +
                   "vRowStride=$vRowStride, vPixelStride=$vPixelStride")
        
        // NV21 格式: Y (width * height) + UV (width * height / 2)
        val nv21Size = width * height * 3 / 2
        val nv21 = if (reuseBuffer != null && reuseBuffer.size == nv21Size) reuseBuffer else ByteArray(nv21Size)
        
        // Y plane: 逐行复制，处理 rowStride
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }
        
        // UV plane: NV21 格式是 V-U-V-U 交错
        // 分别处理 U 和 V planes，不假设它们使用相同布局
        for (row in 0 until height / 2) {
            for (col in 0 until width / 2) {
                // 分别计算 U 和 V 的索引
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                
                nv21[pos++] = vBuffer.get(vIndex)  // V 先
                nv21[pos++] = uBuffer.get(uIndex)  // U 后
            }
        }
        
        return nv21
    }
    
    /**
     * 检查相机权限是否已授予
     */
    fun hasCameraPermission(context: Context): Boolean {
        return ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * 计算合适的 NV21 buffer 大小
     * 用于预分配buffer
     */
    fun calculateNV21Size(width: Int, height: Int): Int {
        // NV21: Y (width * height) + UV (width * height / 2)
        return width * height * 3 / 2
    }
}