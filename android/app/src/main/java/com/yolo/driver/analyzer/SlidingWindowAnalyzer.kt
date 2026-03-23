package com.yolo.driver.analyzer

/**
 * @writer: zhangheng
 * 滑动窗分析器
 * 基于时间窗口累积状态，计算最终疲劳等级
 */
class SlidingWindowAnalyzer(
    windowSizeMs: Long = 5000L  // 默认5秒窗口
) {
    
    // 可变的窗口大小
    private var windowSizeMs: Long = windowSizeMs
    
    // 带时间戳的状态记录
    data class TimedState(
        val driverState: StateAnalyzer.DriverState,
        val headPoses: Set<StateAnalyzer.HeadPose>,
        val timestamp: Long
    )
    
    // 聚合结果
    data class AggregatedResult(
        val driverState: StateAnalyzer.DriverState,
        val normalRatio: Float,
        val slightlyTiredRatio: Float,
        val tiredRatio: Float,
        val windowSize: Int,
        val dominantHeadPoses: Set<StateAnalyzer.HeadPose>
    )
    
    // 状态队列
    private val stateQueue = ArrayDeque<TimedState>()
    
    // 疲劳评分系数
    companion object {
        // 状态优先级权重 (疲劳 > 轻度疲劳 > 正常)
        private const val TIRED_WEIGHT = 3.0f
        private const val SLIGHTLY_TIRED_WEIGHT = 2.0f
        private const val NORMAL_WEIGHT = 1.0f
        
        // 最终判定阈值
        private const val TIRED_THRESHOLD = 0.3f       // 疲劳占比 > 30% 则判定为疲劳
        private const val SLIGHTLY_TIRED_THRESHOLD = 0.2f  // 轻度疲劳占比 > 20% 则判定为轻度疲劳
    }
    
    /**
     * 添加新状态
     */
    fun addState(result: StateAnalyzer.AnalysisResult) {
        val timedState = TimedState(
            driverState = result.driverState,
            headPoses = result.headPoses,
            timestamp = result.timestamp
        )
        stateQueue.addLast(timedState)
        cleanExpiredStates()
    }
    
    /**
     * 清理过期状态
     */
    private fun cleanExpiredStates() {
        val currentTime = System.currentTimeMillis()
        while (stateQueue.isNotEmpty() && currentTime - stateQueue.first().timestamp > windowSizeMs) {
            stateQueue.removeFirst()
        }
    }
    
    /**
     * 获取聚合后的最终状态
     */
    fun getAggregatedState(): AggregatedResult {
        cleanExpiredStates()
        
        if (stateQueue.isEmpty()) {
            return AggregatedResult(
                driverState = StateAnalyzer.DriverState.NORMAL,
                normalRatio = 1.0f,
                slightlyTiredRatio = 0.0f,
                tiredRatio = 0.0f,
                windowSize = 0,
                dominantHeadPoses = emptySet()
            )
        }
        
        val totalFrames = stateQueue.size.toFloat()
        
        // 统计各状态占比
        var normalCount = 0
        var slightlyTiredCount = 0
        var tiredCount = 0
        
        // 统计头部姿态频率
        val headPoseCounts = mutableMapOf<StateAnalyzer.HeadPose, Int>()
        
        for (state in stateQueue) {
            when (state.driverState) {
                StateAnalyzer.DriverState.NORMAL -> normalCount++
                StateAnalyzer.DriverState.SLIGHTLY_TIRED -> slightlyTiredCount++
                StateAnalyzer.DriverState.TIRED -> tiredCount++
            }
            
            // 累计头部姿态
            state.headPoses.forEach { pose ->
                headPoseCounts[pose] = (headPoseCounts[pose] ?: 0) + 1
            }
        }
        
        val normalRatio = normalCount / totalFrames
        val slightlyTiredRatio = slightlyTiredCount / totalFrames
        val tiredRatio = tiredCount / totalFrames
        
        // 计算加权疲劳分数
        val fatigueScore = tiredRatio * TIRED_WEIGHT + 
                          slightlyTiredRatio * SLIGHTLY_TIRED_WEIGHT +
                          normalRatio * NORMAL_WEIGHT
        
        // 判定最终状态 (疲劳优先)
        val finalState = when {
            tiredRatio >= TIRED_THRESHOLD -> StateAnalyzer.DriverState.TIRED
            slightlyTiredRatio >= SLIGHTLY_TIRED_THRESHOLD -> StateAnalyzer.DriverState.SLIGHTLY_TIRED
            tiredRatio > 0.1f -> StateAnalyzer.DriverState.SLIGHTLY_TIRED  // 有少量疲劳帧时提示
            else -> StateAnalyzer.DriverState.NORMAL
        }
        
        // 找出主要头部姿态 (出现频率 > 30%)
        val dominantHeadPoses = headPoseCounts
            .filter { (_, count) -> count / totalFrames > 0.3f }
            .keys
        
        return AggregatedResult(
            driverState = finalState,
            normalRatio = normalRatio,
            slightlyTiredRatio = slightlyTiredRatio,
            tiredRatio = tiredRatio,
            windowSize = stateQueue.size,
            dominantHeadPoses = dominantHeadPoses
        )
    }
    
    /**
     * 获取窗口内状态数量
     */
    fun getWindowSize(): Int {
        cleanExpiredStates()
        return stateQueue.size
    }
    
    /**
     * 清空窗口
     */
    fun clear() {
        stateQueue.clear()
    }
    
    /**
     * 更新窗口大小
     */
    fun updateWindowSize(newSizeMs: Long) {
        windowSizeMs = newSizeMs
        cleanExpiredStates()  // 立即清理过期数据
    }
    
    /**
     * 获取当前窗口大小设置（毫秒）
     */
    fun getWindowSizeMs(): Long = windowSizeMs
}
