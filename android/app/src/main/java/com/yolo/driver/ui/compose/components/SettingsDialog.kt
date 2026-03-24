package com.yolo.driver.ui.compose.components

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yolo.driver.MainViewModel
import com.yolo.driver.R

/**
 * @writer: zhangheng
 * 设置弹窗 Composable（紧凑版，支持自动保存）
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsDialog(
    currentSettings: MainViewModel.SettingsState,
    onDismiss: () -> Unit,
    onSave: (MainViewModel.SettingsState) -> Unit,
    onAutoSave: (MainViewModel.SettingsState) -> Unit = {}  // 自动保存回调
) {
    val context = LocalContext.current
    
    // 震动设置
    var vibrationEnabled by remember { mutableStateOf(currentSettings.vibrationEnabled) }
    var vibrationMode by remember { mutableIntStateOf(currentSettings.vibrationMode) }
    
    // 音频设置
    var audioEnabled by remember { mutableStateOf(currentSettings.audioEnabled) }
    var audioVolume by remember { mutableFloatStateOf(currentSettings.audioVolume.toFloat()) }
    var tiredAudioUri by remember { mutableStateOf(currentSettings.tiredAudioUri) }
    var slightlyTiredAudioUri by remember { mutableStateOf(currentSettings.slightlyTiredAudioUri) }
    
    // 检测模式
    var slidingWindowMode by remember { mutableStateOf(currentSettings.isSlidingWindowMode) }
    var windowDuration by remember { mutableFloatStateOf(currentSettings.windowDurationMs / 1000f) }
    var languageMode by remember { mutableIntStateOf(currentSettings.languageMode) }
    
    // 姿态状态映射
    var framePoseMapping by remember { mutableStateOf(currentSettings.framePoseMapping) }
    var slidingPoseMapping by remember { mutableStateOf(currentSettings.slidingPoseMapping) }
    
    // 关键点置信度阈值
    var drawThreshold by remember { mutableFloatStateOf(currentSettings.drawThreshold) }
    var analysisThreshold by remember { mutableFloatStateOf(currentSettings.analysisThreshold) }
    
    // 提醒播放模式
    var alertRepeatMode by remember { mutableIntStateOf(currentSettings.alertRepeatMode) }
    var showOnceConfirmDialog by remember { mutableStateOf(false) }
    
    // GPU 加速
    var gpuEnabled: Boolean by remember { mutableStateOf(currentSettings.gpuEnabled) }
    
    // 构建当前设置状态的辅助函数
    fun buildCurrentSettings() = MainViewModel.SettingsState(
        vibrationEnabled = vibrationEnabled,
        vibrationMode = vibrationMode,
        audioEnabled = audioEnabled,
        audioVolume = audioVolume.toInt(),
        tiredAudioUri = tiredAudioUri,
        slightlyTiredAudioUri = slightlyTiredAudioUri,
        isSlidingWindowMode = slidingWindowMode,
        windowDurationMs = (windowDuration * 1000).toLong(),
        languageMode = languageMode,
        framePoseMapping = framePoseMapping,
        slidingPoseMapping = slidingPoseMapping,
        drawThreshold = drawThreshold,
        analysisThreshold = analysisThreshold,
        alertRepeatMode = alertRepeatMode,
        gpuEnabled = gpuEnabled
    )
    
    // 音频选择器
    val tiredAudioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            context.contentResolver.takePersistableUriPermission(
                it,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            tiredAudioUri = it.toString()
            // 自动保存
            onAutoSave(buildCurrentSettings())
        }
    }
    
    val slightlyTiredAudioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            context.contentResolver.takePersistableUriPermission(
                it,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            slightlyTiredAudioUri = it.toString()
            // 自动保存
            onAutoSave(buildCurrentSettings())
        }
    }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { 
            Text(
                text = stringResource(R.string.settings),
                style = MaterialTheme.typography.titleMedium
            ) 
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                // ========== 检测模式 ==========
                SettingsSectionHeader(title = stringResource(R.string.detection_mode))
                
                // 逐帧检测
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = !slidingWindowMode,
                            onClick = { 
                                slidingWindowMode = false
                                onAutoSave(buildCurrentSettings().copy(isSlidingWindowMode = false))
                            },
                            role = Role.RadioButton
                        )
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = !slidingWindowMode, 
                        onClick = null,
                        modifier = Modifier.height(20.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = stringResource(R.string.frame_by_frame),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                
                // 滑动窗模式
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = slidingWindowMode,
                            onClick = { 
                                slidingWindowMode = true
                                onAutoSave(buildCurrentSettings().copy(isSlidingWindowMode = true))
                            },
                            role = Role.RadioButton
                        )
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = slidingWindowMode, 
                        onClick = null,
                        modifier = Modifier.height(20.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = stringResource(R.string.sliding_window),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                
                // 滑动窗时长
                if (slidingWindowMode) {
                    Text(
                        text = "${stringResource(R.string.window_duration)}: ${windowDuration.toInt()}s",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Slider(
                        value = windowDuration,
                        onValueChange = { windowDuration = it },
                        onValueChangeFinished = { 
                            onAutoSave(buildCurrentSettings())
                        },
                        valueRange = 1f..30f,
                        steps = 29,
                        modifier = Modifier.height(24.dp)
                    )
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 震动设置 ==========
                SettingsSectionHeader(title = stringResource(R.string.vibration_alert))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.enable_vibration),
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodySmall
                    )
                    Switch(
                        checked = vibrationEnabled,
                        onCheckedChange = { 
                            vibrationEnabled = it
                            onAutoSave(buildCurrentSettings())
                        },
                        modifier = Modifier.height(20.dp)
                    )
                }
                
                if (vibrationEnabled) {
                    Text(
                        text = stringResource(R.string.vibration_mode),
                        style = MaterialTheme.typography.bodySmall
                    )
                    val vibrationModes = listOf(
                        R.string.vibration_short,
                        R.string.vibration_long,
                        R.string.vibration_double,
                        R.string.vibration_pulse
                    )
                    vibrationModes.forEachIndexed { index, modeRes ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .selectable(
                                    selected = vibrationMode == index,
                                    onClick = { 
                                        vibrationMode = index
                                        onAutoSave(buildCurrentSettings())
                                    },
                                    role = Role.RadioButton
                                )
                                .padding(vertical = 1.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = vibrationMode == index, 
                                onClick = null,
                                modifier = Modifier.height(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = stringResource(modeRes),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 音频设置 ==========
                SettingsSectionHeader(title = stringResource(R.string.audio_alert))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.enable_audio),
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodySmall
                    )
                    Switch(
                        checked = audioEnabled,
                        onCheckedChange = { 
                            audioEnabled = it
                            onAutoSave(buildCurrentSettings())
                        },
                        modifier = Modifier.height(20.dp)
                    )
                }
                
                if (audioEnabled) {
                    Text(
                        text = "${stringResource(R.string.volume)}: ${audioVolume.toInt()}%",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Slider(
                        value = audioVolume,
                        onValueChange = { audioVolume = it },
                        onValueChangeFinished = { 
                            onAutoSave(buildCurrentSettings())
                        },
                        valueRange = 0f..100f,
                        modifier = Modifier.height(24.dp)
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    // 疲劳音频选择
                    Text(
                        text = stringResource(R.string.tired_audio),
                        style = MaterialTheme.typography.bodySmall
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(
                            onClick = { tiredAudioPicker.launch(arrayOf("audio/*")) },
                            modifier = Modifier.height(28.dp),
                            contentPadding = ButtonDefaults.ContentPadding
                        ) {
                            Text(
                                text = stringResource(R.string.select_custom_audio),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                        if (tiredAudioUri != null) {
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = stringResource(R.string.default_audio),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    // 轻度疲劳音频选择
                    Text(
                        text = stringResource(R.string.slightly_tired_audio),
                        style = MaterialTheme.typography.bodySmall
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(
                            onClick = { slightlyTiredAudioPicker.launch(arrayOf("audio/*")) },
                            modifier = Modifier.height(28.dp),
                            contentPadding = ButtonDefaults.ContentPadding
                        ) {
                            Text(
                                text = stringResource(R.string.select_custom_audio),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                        if (slightlyTiredAudioUri != null) {
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = stringResource(R.string.default_audio),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 提醒播放模式 ==========
                SettingsSectionHeader(title = stringResource(R.string.alert_repeat_mode))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.alert_repeat_once),
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodySmall
                    )
                    RadioButton(
                        selected = alertRepeatMode == 0,
                        onClick = { showOnceConfirmDialog = true },
                        modifier = Modifier.height(20.dp)
                    )
                }
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.alert_repeat_continuous),
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodySmall
                    )
                    RadioButton(
                        selected = alertRepeatMode == 1,
                        onClick = { 
                            alertRepeatMode = 1
                            onAutoSave(buildCurrentSettings())
                        },
                        modifier = Modifier.height(20.dp)
                    )
                }
                
                // "只播放一次"确认对话框
                if (showOnceConfirmDialog) {
                    AlertDialog(
                        onDismissRequest = { showOnceConfirmDialog = false },
                        title = { Text(stringResource(R.string.confirm_once_title)) },
                        text = { Text(stringResource(R.string.confirm_once_message)) },
                        confirmButton = {
                            TextButton(onClick = {
                                alertRepeatMode = 0
                                onAutoSave(buildCurrentSettings())
                                showOnceConfirmDialog = false
                            }) {
                                Text(stringResource(R.string.yes))
                            }
                        },
                        dismissButton = {
                            TextButton(onClick = {
                                showOnceConfirmDialog = false
                            }) {
                                Text(stringResource(R.string.no))
                            }
                        }
                    )
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 姿态状态映射 ==========
                SettingsSectionHeader(title = stringResource(R.string.pose_state_mapping))
                
                // 逐帧模式映射
                Text(
                    text = stringResource(R.string.frame_mode_mapping),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
                
                PoseMappingSection(
                    mapping = framePoseMapping,
                    onMappingChange = { 
                        framePoseMapping = it
                        onAutoSave(buildCurrentSettings())
                    }
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                // 滑动窗模式映射
                Text(
                    text = stringResource(R.string.sliding_mode_mapping),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
                
                PoseMappingSection(
                    mapping = slidingPoseMapping,
                    onMappingChange = { 
                        slidingPoseMapping = it
                        onAutoSave(buildCurrentSettings())
                    }
                )
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 关键点置信度阈值 ==========
                SettingsSectionHeader(title = stringResource(R.string.keypoint_settings))
                
                Text(
                    text = "${stringResource(R.string.draw_threshold)}: ${String.format("%.2f", drawThreshold)}",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = stringResource(R.string.draw_threshold_desc),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Slider(
                    value = drawThreshold,
                    onValueChange = { drawThreshold = it },
                    onValueChangeFinished = { 
                        onAutoSave(buildCurrentSettings())
                    },
                    valueRange = 0.3f..0.8f,
                    steps = 10,
                    modifier = Modifier.height(24.dp)
                )
                
                Text(
                    text = "${stringResource(R.string.analysis_threshold)}: ${String.format("%.2f", analysisThreshold)}",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = stringResource(R.string.analysis_threshold_desc),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Slider(
                    value = analysisThreshold,
                    onValueChange = { analysisThreshold = it },
                    onValueChangeFinished = { 
                        onAutoSave(buildCurrentSettings())
                    },
                    valueRange = 0.3f..0.8f,
                    steps = 10,
                    modifier = Modifier.height(24.dp)
                )
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 语言设置 ==========
                SettingsSectionHeader(title = stringResource(R.string.language))
                
                val languages = listOf(
                    R.string.language_auto,
                    R.string.language_chinese,
                    R.string.language_english
                )
                languages.forEachIndexed { index, langRes ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = languageMode == index,
                                onClick = { 
                                    languageMode = index
                                    onAutoSave(buildCurrentSettings())
                                },
                                role = Role.RadioButton
                            )
                            .padding(vertical = 1.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = languageMode == index, 
                            onClick = null,
                            modifier = Modifier.height(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = stringResource(langRes),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 6.dp))
                
                // ========== 计算后端 ==========
                SettingsSectionHeader(title = stringResource(R.string.compute_backend))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = stringResource(R.string.gpu_acceleration),
                            style = MaterialTheme.typography.bodySmall
                        )
                        Text(
                            text = stringResource(R.string.gpu_acceleration_desc),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 10.sp
                        )
                    }
                    Switch(
                        checked = gpuEnabled,
                        onCheckedChange = { 
                            gpuEnabled = it
                            onAutoSave(buildCurrentSettings())
                        },
                        modifier = Modifier.height(24.dp)
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onSave(buildCurrentSettings())
                    onDismiss()
                }
            ) {
                Text(
                    text = stringResource(R.string.save),
                    style = MaterialTheme.typography.labelMedium
                )
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(
                    text = stringResource(R.string.cancel),
                    style = MaterialTheme.typography.labelMedium
                )
            }
        }
    )
}

/**
 * 设置区块标题（紧凑版）
 */
@Composable
private fun SettingsSectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary
    )
    Spacer(modifier = Modifier.height(2.dp))
}

/**
 * 姿态映射区块（紧凑版）
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PoseMappingSection(
    mapping: MainViewModel.PoseStateMapping,
    onMappingChange: (MainViewModel.PoseStateMapping) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        // 抬头/低头
        PoseDropdownRow(
            label = stringResource(R.string.head_up_down),
            currentState = mapping.headUpDown,
            onStateChange = { newState ->
                onMappingChange(mapping.copy(headUpDown = newState))
            }
        )
        
        // 左右摆头
        PoseDropdownRow(
            label = stringResource(R.string.head_left_right),
            currentState = mapping.headLeftRight,
            onStateChange = { newState ->
                onMappingChange(mapping.copy(headLeftRight = newState))
            }
        )
        
        // 姿态偏移
        PoseDropdownRow(
            label = stringResource(R.string.posture_deviation),
            currentState = mapping.postureDeviation,
            onStateChange = { newState ->
                onMappingChange(mapping.copy(postureDeviation = newState))
            }
        )
    }
}

/**
 * 姿态下拉选择行（紧凑版）
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PoseDropdownRow(
    label: String,
    currentState: MainViewModel.DriverState,
    onStateChange: (MainViewModel.DriverState) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    var showNormalConfirmDialog by remember { mutableStateOf(false) }
    var pendingState: MainViewModel.DriverState? by remember { mutableStateOf(null) }
    
    val stateOptions = listOf(
        MainViewModel.DriverState.Normal to R.string.state_normal,
        MainViewModel.DriverState.SlightlyTired to R.string.state_slightly_tired,
        MainViewModel.DriverState.Tired to R.string.state_tired
    )
    
    val currentStateLabel = stateOptions.find { it.first == currentState }?.let {
        stringResource(it.second)
    } ?: stringResource(R.string.unknown)
    
    // Normal 状态确认对话框
    if (showNormalConfirmDialog) {
        AlertDialog(
            onDismissRequest = { 
                showNormalConfirmDialog = false
                pendingState = null
            },
            title = { Text(stringResource(R.string.confirm_normal_title)) },
            text = { Text(stringResource(R.string.confirm_normal_message)) },
            confirmButton = {
                TextButton(onClick = {
                    pendingState?.let { onStateChange(it) }
                    showNormalConfirmDialog = false
                    pendingState = null
                }) {
                    Text(stringResource(R.string.yes))
                }
            },
            dismissButton = {
                TextButton(onClick = {
                    showNormalConfirmDialog = false
                    pendingState = null
                }) {
                    Text(stringResource(R.string.no))
                }
            }
        )
    }
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodySmall
        )
        
        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { expanded = it }
        ) {
            OutlinedTextField(
                value = currentStateLabel,
                onValueChange = {},
                readOnly = true,
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                modifier = Modifier
                    .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                    .width(100.dp),
                textStyle = MaterialTheme.typography.labelSmall
            )
            
            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                stateOptions.forEach { (state, stringRes) ->
                    DropdownMenuItem(
                        text = { 
                            Text(
                                text = stringResource(stringRes),
                                style = MaterialTheme.typography.bodySmall
                            ) 
                        },
                        onClick = {
                            expanded = false
                            if (state == MainViewModel.DriverState.Normal) {
                                // Normal 需要确认
                                pendingState = state
                                showNormalConfirmDialog = true
                            } else {
                                // 其他状态直接设置
                                onStateChange(state)
                            }
                        }
                    )
                }
            }
        }
    }
}