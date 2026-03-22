package com.yolo.driver.util

import android.Manifest
import android.content.pm.PackageManager
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import android.content.Context

/**
 * 相机相关工具类
 * 提供NV21转换、权限检查等通用功能
 */
object CameraUtils {
    
    /**
     * 将 ImageProxy 转换为 NV21 格式字节数组
     * 支持buffer复用以减少GC压力
     * 
     * @param image CameraX ImageProxy
     * @param reuseBuffer 可复用的buffer（可为null）
     * @return Pair<ByteArray, Boolean> 返回NV21数据和是否使用了新buffer
     */
    fun imageProxyToNV21(
        image: ImageProxy,
        reuseBuffer: ByteArray? = null
    ): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val totalSize = ySize + uSize + vSize
        
        // 复用或创建新buffer
        val nv21 = if (reuseBuffer != null && reuseBuffer.size == totalSize) {
            reuseBuffer
        } else {
            ByteArray(totalSize)
        }
        
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
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
