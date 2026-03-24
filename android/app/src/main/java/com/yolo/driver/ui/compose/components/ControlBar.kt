package com.yolo.driver.ui.compose.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.RotateRight
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Flip
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yolo.driver.R

/**
 * 底部控制栏 Composable
 * 使用 FlowRow 实现自适应换行
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ControlBar(
    rotationText: String,
    isCalibrated: Boolean,
    onRotate: () -> Unit,
    onCalibrate: () -> Unit,
    onReset: () -> Unit,
    onSettings: () -> Unit,
    zoomRatio: Float = 1.0f,
    minZoomRatio: Float = 1.0f,
    onZoomIn: (() -> Unit)? = null,
    onZoomOut: (() -> Unit)? = null,
    useFrontCamera: Boolean = true,
    onSwitchCamera: (() -> Unit)? = null,
    mirrorKeypoints: Boolean = true,
    onToggleMirror: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    FlowRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterHorizontally),
        verticalArrangement = Arrangement.spacedBy(4.dp),
        maxLines = 2
    ) {
        // 第一组：摄像头切换
        if (onSwitchCamera != null) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                IconButton(
                    onClick = onSwitchCamera,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Cameraswitch,
                        contentDescription = stringResource(R.string.switch_camera),
                        modifier = Modifier.size(20.dp)
                    )
                }
                Text(
                    text = if (useFrontCamera) "F" else "B",
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        
        // 第二组：镜像切换
        if (onToggleMirror != null) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                IconButton(
                    onClick = onToggleMirror,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Flip,
                        contentDescription = stringResource(R.string.mirror),
                        modifier = Modifier.size(20.dp)
                    )
                }
                Text(
                    text = if (mirrorKeypoints) stringResource(R.string.mirror) else stringResource(R.string.normal),
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        
        // 第三组：旋转
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            IconButton(
                onClick = onRotate,
                modifier = Modifier.size(36.dp)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.RotateRight,
                    contentDescription = stringResource(R.string.rotate),
                    modifier = Modifier.size(20.dp)
                )
            }
            Text(
                text = rotationText,
                fontSize = 12.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        
        // 第三组：缩放
        if (onZoomIn != null && onZoomOut != null) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                IconButton(
                    onClick = onZoomOut,
                    enabled = zoomRatio > minZoomRatio,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Remove,
                        contentDescription = stringResource(R.string.zoom_out),
                        modifier = Modifier.size(20.dp)
                    )
                }
                Text(
                    text = String.format("%.1fx", zoomRatio),
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                IconButton(
                    onClick = onZoomIn,
                    modifier = Modifier.size(36.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = stringResource(R.string.zoom_in),
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
        
        // 第四组：校准
        Button(
            onClick = onCalibrate,
            modifier = Modifier.height(36.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 4.dp)
        ) {
            Text(
                text = stringResource(R.string.calibrate),
                fontSize = 12.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        
        // 第五组：重置
        IconButton(
            onClick = onReset,
            modifier = Modifier.size(36.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Refresh,
                contentDescription = stringResource(R.string.reset),
                modifier = Modifier.size(20.dp)
            )
        }
        
        // 第六组：设置
        IconButton(
            onClick = onSettings,
            modifier = Modifier.size(36.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = stringResource(R.string.settings),
                modifier = Modifier.size(20.dp)
            )
        }
    }
}
