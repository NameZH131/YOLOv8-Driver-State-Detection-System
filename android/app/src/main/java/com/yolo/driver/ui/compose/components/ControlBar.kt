package com.yolo.driver.ui.compose.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.RotateRight
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.yolo.driver.R

/**
 * 底部控制栏 Composable
 */
@Composable
fun ControlBar(
    rotationText: String,
    isCalibrated: Boolean,
    onRotate: () -> Unit,
    onCalibrate: () -> Unit,
    onReset: () -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // 旋转按钮
        IconButton(onClick = onRotate) {
            Icon(
                imageVector = Icons.Default.RotateRight,
                contentDescription = stringResource(R.string.rotate)
            )
        }
        Text(text = rotationText)
        
        Spacer(modifier = Modifier.width(8.dp))
        
        // 校准按钮
        Button(onClick = onCalibrate) {
            Text(text = stringResource(R.string.calibrate))
        }
        
        Spacer(modifier = Modifier.width(8.dp))
        
        // 重置按钮
        IconButton(onClick = onReset) {
            Icon(
                imageVector = Icons.Default.Refresh,
                contentDescription = stringResource(R.string.reset)
            )
        }
        
        // 设置按钮
        IconButton(onClick = onSettings) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = stringResource(R.string.settings)
            )
        }
    }
}
