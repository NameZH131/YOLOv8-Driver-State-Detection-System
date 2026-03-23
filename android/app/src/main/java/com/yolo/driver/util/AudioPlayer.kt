package com.yolo.driver.util

import android.content.Context
import android.media.MediaPlayer
import android.net.Uri
import android.util.Log
import java.io.File

/**
 * @writer: zhangheng
 * 音频播放控制器
 * 支持优先级播放、音量控制、Uri播放
 */
class AudioPlayer(private val context: Context) {
    
    companion object {
        private const val TAG = "AudioPlayer"
        
        // 播放优先级常量
        const val PRIORITY_NORMAL = 0
        const val PRIORITY_SLIGHTLY_TIRED = 1
        const val PRIORITY_TIRED = 2
        
        // 默认音频资源
        private const val DEFAULT_TIRED_AUDIO = "manbo.mp3"
        private const val DEFAULT_SLIGHTLY_TIRED_AUDIO = "ohYeah.mp3"
    }
    
    // 当前播放器 (线程安全)
    @Volatile
    private var mediaPlayer: MediaPlayer? = null
    
    // 当前播放优先级
    @Volatile
    private var currentPriority: Int = -1
    
    // 当前播放状态
    @Volatile
    private var isPlaying: Boolean = false
    
    // 是否正在异步准备中
    @Volatile
    private var isPreparing: Boolean = false
    
    // 线程锁
    private val lock = Any()
    
    // 播放冷却时间 (避免连续高频播放)
    private var lastPlayTime: Long = 0
    private var cooldownMs: Long = 3000L  // 默认3秒冷却
    
    // 音频启用状态
    private var audioEnabled: Boolean = true
    
    // 音量 (0-100)
    private var volume: Int = 100
    
    // 自定义音频 Uri
    private var tiredAudioUri: Uri? = null
    private var slightlyTiredAudioUri: Uri? = null
    
    // 预加载默认音频 Uri
    private var defaultTiredAudioUri: Uri? = null
    private var defaultSlightlyTiredAudioUri: Uri? = null
    
    init {
        // 预加载默认音频
        preloadDefaultAudio()
    }
    
    /**
     * 预加载默认音频文件
     */
    private fun preloadDefaultAudio() {
        try {
            defaultTiredAudioUri = getAssetUri(DEFAULT_TIRED_AUDIO)
            defaultSlightlyTiredAudioUri = getAssetUri(DEFAULT_SLIGHTLY_TIRED_AUDIO)
            Log.d(TAG, "Default audio files preloaded")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to preload default audio files", e)
        }
    }
    
    /**
     * 播放音频
     * @param priority 优先级，数值越大优先级越高
     * @param uri 自定义音频 Uri (可选)
     */
    fun play(priority: Int, uri: Uri? = null) {
        if (!audioEnabled) {
            Log.d(TAG, "Audio disabled, skip playback")
            return
        }
        
        // 检查冷却时间
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastPlayTime < cooldownMs && priority <= currentPriority) {
            Log.d(TAG, "In cooldown period, skip playback")
            return
        }
        
        // 优先级检查：新音频优先级更高才播放
        if (isPlaying && priority <= currentPriority) {
            Log.d(TAG, "Current audio has higher or equal priority, skip")
            return
        }
        
        // 停止当前播放
        stop()
        
        synchronized(lock) {
            try {
                // 确定音频源
                val audioUri = uri ?: when (priority) {
                    PRIORITY_TIRED -> tiredAudioUri ?: defaultTiredAudioUri
                    PRIORITY_SLIGHTLY_TIRED -> slightlyTiredAudioUri ?: defaultSlightlyTiredAudioUri
                    else -> null
                }
                
                if (audioUri != null && audioUri != Uri.EMPTY) {
                    isPreparing = true
                    currentPriority = priority
                    lastPlayTime = currentTime
                    
                    val mp = MediaPlayer()
                    mp.setDataSource(context, audioUri)
                    mp.setOnPreparedListener { player ->
                        synchronized(lock) {
                            isPreparing = false
                            player.setVolume(volume / 100f, volume / 100f)
                            player.start()
                            isPlaying = true
                        }
                        Log.d(TAG, "Audio prepared and started playing")
                    }
                    mp.setOnCompletionListener {
                        release()
                    }
                    mp.setOnErrorListener { player, what, extra ->
                        Log.e(TAG, "MediaPlayer error: what=$what, extra=$extra")
                        synchronized(lock) {
                            isPreparing = false
                        }
                        player.release()
                        mediaPlayer = null
                        isPlaying = false
                        true
                    }
                    mp.prepareAsync()  // 异步准备，不阻塞线程
                    mediaPlayer = mp
                    
                    Log.d(TAG, "Started preparing audio with priority=$priority, volume=$volume, uri=$audioUri")
                } else {
                    Log.w(TAG, "No audio uri available for priority=$priority")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to play audio", e)
                synchronized(lock) {
                    isPreparing = false
                }
                release()
            }
        }
    }
    
/**
     * 播放疲劳音频
     */
    fun playTired() {
        play(PRIORITY_TIRED, null)
    }

    /**
     * 播放轻度疲劳音频
     */
    fun playSlightlyTired() {
        play(PRIORITY_SLIGHTLY_TIRED, null)
    }
    
    /**
     * 停止播放
     */
    fun stop() {
        synchronized(lock) {
            mediaPlayer?.let {
                try {
                    if (isPreparing) {
                        // 正在异步准备中，直接释放
                        Log.d(TAG, "MediaPlayer is preparing, force release")
                    } else if (it.isPlaying) {
                        it.stop()
                    }
                    it.release()
                } catch (e: Exception) {
                    Log.e(TAG, "Error stopping MediaPlayer", e)
                }
            }
            mediaPlayer = null
            isPlaying = false
            isPreparing = false
        }
    }
    
    /**
     * 释放资源
     */
    private fun release() {
        synchronized(lock) {
            mediaPlayer?.release()
            mediaPlayer = null
            isPlaying = false
            isPreparing = false
        }
    }
    
    /**
     * 设置音频启用状态
     */
    fun setEnabled(enabled: Boolean) {
        audioEnabled = enabled
        if (!enabled) {
            stop()
        }
    }
    
    /**
     * 设置音量 (0-100)
     */
    fun setVolume(volume: Int) {
        this.volume = volume.coerceIn(0, 100)
        val vol = this.volume / 100f
        mediaPlayer?.setVolume(vol, vol)
        Log.d(TAG, "Volume set to: ${this.volume}")
    }
    
    /**
     * 获取当前音量
     */
    fun getVolume(): Int = volume
    
    /**
     * 设置冷却时间
     */
    fun setCooldown(ms: Long) {
        cooldownMs = ms
    }
    
    /**
     * 设置自定义疲劳音频 Uri
     */
    fun setTiredAudioUri(uri: Uri?) {
        tiredAudioUri = uri
        Log.d(TAG, "Tired audio uri set to: $uri")
    }
    
    /**
     * 设置自定义轻度疲劳音频 Uri
     */
    fun setSlightlyTiredAudioUri(uri: Uri?) {
        slightlyTiredAudioUri = uri
        Log.d(TAG, "Slightly tired audio uri set to: $uri")
    }
    
    /**
     * 获取 Asset 文件的 Uri (通过复制到缓存目录)
     */
    private fun getAssetUri(assetName: String): Uri {
        val cacheFile = File(context.cacheDir, assetName)
        if (!cacheFile.exists()) {
            try {
                context.assets.open(assetName).use { input ->
                    cacheFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy asset: $assetName", e)
                return Uri.EMPTY
            }
        }
        return Uri.parse("file://${cacheFile.absolutePath}")
    }
    
    /**
     * 完全释放
     */
    fun destroy() {
        stop()
    }
}