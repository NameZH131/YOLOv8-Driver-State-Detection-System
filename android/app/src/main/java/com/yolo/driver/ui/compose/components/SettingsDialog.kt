package com.yolo.driver.ui.compose.components

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
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.yolo.driver.MainViewModel
import com.yolo.driver.R

/**
 * 设置弹窗 Composable
 */
@Composable
fun SettingsDialog(
    currentSettings: MainViewModel.SettingsState,
    onDismiss: () -> Unit,
    onSave: (MainViewModel.SettingsState) -> Unit
) {
    var vibrationEnabled by remember { mutableStateOf(currentSettings.vibrationEnabled) }
    var vibrationMode by remember { mutableIntStateOf(currentSettings.vibrationMode) }
    var audioEnabled by remember { mutableStateOf(currentSettings.audioEnabled) }
    var audioVolume by remember { mutableFloatStateOf(currentSettings.audioVolume.toFloat()) }
    var slidingWindowMode by remember { mutableStateOf(currentSettings.isSlidingWindowMode) }
    var windowDuration by remember { mutableFloatStateOf(currentSettings.windowDurationMs / 1000f) }
    var languageMode by remember { mutableIntStateOf(currentSettings.languageMode) }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(text = stringResource(R.string.settings)) },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
            ) {
                // 检测模式
                Text(
                    text = stringResource(R.string.detection_mode),
                    style = MaterialTheme.typography.titleMedium
                )
                
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
                    RadioButton(
                        selected = !slidingWindowMode,
                        onClick = null
                    )
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
                    RadioButton(
                        selected = slidingWindowMode,
                        onClick = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = stringResource(R.string.sliding_window))
                }
                
                // 滑动窗时长（仅滑动窗模式显示）
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
                
                // 震动设置
                Text(
                    text = stringResource(R.string.vibration_alert),
                    style = MaterialTheme.typography.titleMedium
                )
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
                            RadioButton(
                                selected = vibrationMode == index,
                                onClick = null
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(text = stringResource(modeRes))
                        }
                    }
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // 音频设置
                Text(
                    text = stringResource(R.string.audio_alert),
                    style = MaterialTheme.typography.titleMedium
                )
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
                }
                
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                
                // 语言设置
                Text(
                    text = stringResource(R.string.language),
                    style = MaterialTheme.typography.titleMedium
                )
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
                        RadioButton(
                            selected = languageMode == index,
                            onClick = null
                        )
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
                            isSlidingWindowMode = slidingWindowMode,
                            windowDurationMs = (windowDuration * 1000).toLong(),
                            languageMode = languageMode
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