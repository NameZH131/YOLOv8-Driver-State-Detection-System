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
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
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
import com.yolo.driver.MainViewModel
import com.yolo.driver.R

/**
 * @writer: zhangheng
 * 设置弹窗 Composable（完整版）
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsDialog(
    currentSettings: MainViewModel.SettingsState,
    onDismiss: () -> Unit,
    onSave: (MainViewModel.SettingsState) -> Unit
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
        }
    }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(text = stringResource(R.string.settings)) },
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
                            onClick = { slidingWindowMode = false },
                            role = Role.RadioButton
                        )
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(selected = !slidingWindowMode, onClick = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = stringResource(R.string.frame_by_frame))
                }
                
                // 滑动窗模式
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = slidingWindowMode,
                            onClick = { slidingWindowMode = true },
                            role = Role.RadioButton
                        )
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(selected = slidingWindowMode, onClick = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = stringResource(R.string.sliding_window))
                }
                
                // 滑动窗时长
                if (slidingWindowMode) {
                    Text(
                        text = "${stringResource(R.string.window_duration)}: ${windowDuration.toInt()}s",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Slider(
                        value = windowDuration,
                        onValueChange = { windowDuration = it },
                        valueRange = 1f..30f,
                        steps = 29
                    )
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // ========== 震动设置 ==========
                SettingsSectionHeader(title = stringResource(R.string.vibration_alert))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.enable_vibration),
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = vibrationEnabled,
                        onCheckedChange = { vibrationEnabled = it }
                    )
                }
                
                if (vibrationEnabled) {
                    Text(
                        text = stringResource(R.string.vibration_mode),
                        style = MaterialTheme.typography.bodyMedium
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
                                    onClick = { vibrationMode = index },
                                    role = Role.RadioButton
                                )
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(selected = vibrationMode == index, onClick = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(text = stringResource(modeRes))
                        }
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // ========== 音频设置 ==========
                SettingsSectionHeader(title = stringResource(R.string.audio_alert))
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(R.string.enable_audio),
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = audioEnabled,
                        onCheckedChange = { audioEnabled = it }
                    )
                }
                
                if (audioEnabled) {
                    Text(
                        text = "${stringResource(R.string.volume)}: ${audioVolume.toInt()}%",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Slider(
                        value = audioVolume,
                        onValueChange = { audioVolume = it },
                        valueRange = 0f..100f
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    // 疲劳音频选择
                    Text(
                        text = stringResource(R.string.tired_audio),
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(onClick = { tiredAudioPicker.launch(arrayOf("audio/*")) }) {
                            Text(text = stringResource(R.string.select_custom_audio))
                        }
                        if (tiredAudioUri != null) {
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = stringResource(R.string.default_audio),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    // 轻度疲劳音频选择
                    Text(
                        text = stringResource(R.string.slightly_tired_audio),
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Button(onClick = { slightlyTiredAudioPicker.launch(arrayOf("audio/*")) }) {
                            Text(text = stringResource(R.string.select_custom_audio))
                        }
                        if (slightlyTiredAudioUri != null) {
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = stringResource(R.string.default_audio),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // ========== 姿态状态映射 ==========
                SettingsSectionHeader(title = stringResource(R.string.pose_state_mapping))
                
                // 逐帧模式映射
                Text(
                    text = stringResource(R.string.frame_mode_mapping),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
                
                framePoseMapping = PoseMappingSection(
                    mapping = framePoseMapping,
                    onMappingChange = { framePoseMapping = it }
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // 滑动窗模式映射
                Text(
                    text = stringResource(R.string.sliding_mode_mapping),
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
                
                slidingPoseMapping = PoseMappingSection(
                    mapping = slidingPoseMapping,
                    onMappingChange = { slidingPoseMapping = it }
                )
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // ========== 关键点置信度阈值 ==========
                SettingsSectionHeader(title = stringResource(R.string.keypoint_settings))
                
                Text(
                    text = "${stringResource(R.string.draw_threshold)}: ${String.format("%.2f", drawThreshold)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = stringResource(R.string.draw_threshold_desc),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Slider(
                    value = drawThreshold,
                    onValueChange = { drawThreshold = it },
                    valueRange = 0.3f..0.8f,
                    steps = 10
                )
                
                Text(
                    text = "${stringResource(R.string.analysis_threshold)}: ${String.format("%.2f", analysisThreshold)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    text = stringResource(R.string.analysis_threshold_desc),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Slider(
                    value = analysisThreshold,
                    onValueChange = { analysisThreshold = it },
                    valueRange = 0.3f..0.8f,
                    steps = 10
                )
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
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
                                onClick = { languageMode = index },
                                role = Role.RadioButton
                            )
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(selected = languageMode == index, onClick = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = stringResource(langRes))
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onSave(
                        currentSettings.copy(
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
                            analysisThreshold = analysisThreshold
                        )
                    )
                }
            ) {
                Text(text = stringResource(R.string.save))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(text = stringResource(R.string.cancel))
            }
        }
    )
}

/**
 * 设置区块标题
 */
@Composable
private fun SettingsSectionHeader(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleMedium,
        color = MaterialTheme.colorScheme.primary
    )
    Spacer(modifier = Modifier.height(4.dp))
}

/**
 * 姿态映射区块
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PoseMappingSection(
    mapping: MainViewModel.PoseStateMapping,
    onMappingChange: (MainViewModel.PoseStateMapping) -> Unit
): MainViewModel.PoseStateMapping {
    var currentMapping = mapping
    
    Column(modifier = Modifier.fillMaxWidth()) {
        // 抬头/低头
        currentMapping = PoseDropdownRow(
            label = stringResource(R.string.head_up_down),
            currentState = currentMapping.headUpDown,
            onStateChange = { newState ->
                currentMapping = currentMapping.copy(headUpDown = newState)
                onMappingChange(currentMapping)
            }
        )
        
        // 左右摆头
        currentMapping = PoseDropdownRow(
            label = stringResource(R.string.head_left_right),
            currentState = currentMapping.headLeftRight,
            onStateChange = { newState ->
                currentMapping = currentMapping.copy(headLeftRight = newState)
                onMappingChange(currentMapping)
            }
        )
        
        // 姿态偏移
        currentMapping = PoseDropdownRow(
            label = stringResource(R.string.posture_deviation),
            currentState = currentMapping.postureDeviation,
            onStateChange = { newState ->
                currentMapping = currentMapping.copy(postureDeviation = newState)
                onMappingChange(currentMapping)
            }
        )
    }
    
    return currentMapping
}

/**
 * 姿态下拉选择行
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PoseDropdownRow(
    label: String,
    currentState: MainViewModel.DriverState,
    onStateChange: (MainViewModel.DriverState) -> Unit
): MainViewModel.PoseStateMapping {
    var expanded by remember { mutableStateOf(false) }
    
    val stateOptions = listOf(
        MainViewModel.DriverState.Normal to R.string.state_normal,
        MainViewModel.DriverState.SlightlyTired to R.string.state_slightly_tired,
        MainViewModel.DriverState.Tired to R.string.state_tired
    )
    
    val currentStateLabel = stateOptions.find { it.first == currentState }?.let {
        stringResource(it.second)
    } ?: stringResource(R.string.unknown)
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyMedium
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
                    .width(140.dp)
            )
            
            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                stateOptions.forEach { (state, stringRes) ->
                    DropdownMenuItem(
                        text = { Text(stringResource(stringRes)) },
                        onClick = {
                            onStateChange(state)
                            expanded = false
                        }
                    )
                }
            }
        }
    }
    
    // 返回一个空的 mapping，因为我们已经通过回调更新了
    return MainViewModel.PoseStateMapping()
}
