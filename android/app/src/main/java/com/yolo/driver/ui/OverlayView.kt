package com.yolo.driver.ui

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View

typealias KeypointDrawCallback = (canvas: Canvas, viewWidth: Int, viewHeight: Int) -> Unit

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private var drawCallback: KeypointDrawCallback? = null
    
    fun setKeypointDrawCallback(callback: KeypointDrawCallback) {
        drawCallback = callback
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawCallback?.invoke(canvas, width, height)
    }
}
