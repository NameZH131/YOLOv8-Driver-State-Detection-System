package com.yolo.driver.ui.compose.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.yolo.driver.MainViewModel
import com.yolo.driver.R
import com.yolo.driver.analyzer.StateAnalyzer
import com.yolo.driver.ui.compose.theme.Normal
import com.yolo.driver.ui.compose.theme.SlightlyTired
import com.yolo.driver.ui.compose.theme.Tired

/**
 * 状态面板 Composable
 */
@Composable
fun StatePanel(
    driverState: MainViewModel.DriverState,
    headPoses: Set<StateAnalyzer.HeadPose>,
    frameCount: Int,
    isCalibrated: Boolean,
    manualRotation: Int,
    windowFrameCount: Int,
    isSlidingWindowMode: Boolean,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black.copy(alpha = 0.6f))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // 驾驶员状态
        val (stateText, stateColor) = when (driverState) {
            is MainViewModel.DriverState.Normal -> 
                stringResource(R.string.state_normal) to Normal
            is MainViewModel.DriverState.SlightlyTired -> 
                stringResource(R.string.state_slightly_tired) to SlightlyTired
            is MainViewModel.DriverState.Tired -> 
                stringResource(R.string.state_tired) to Tired
        }
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = stringResource(R.string.driver_state),
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White
            )
            Text(
                text = stateText,
                style = MaterialTheme.typography.titleMedium,
                color = stateColor
            )
        }
        
        // 姿态（国际化显示）
        if (headPoses.isNotEmpty()) {
            val localizedPoses = headPoses.map { pose ->
                when (pose) {
                    StateAnalyzer.HeadPose.FACING_FORWARD -> stringResource(R.string.pose_facing_forward)
                    StateAnalyzer.HeadPose.HEAD_UP -> stringResource(R.string.pose_head_up)
                    StateAnalyzer.HeadPose.HEAD_DOWN -> stringResource(R.string.pose_head_down)
                    StateAnalyzer.HeadPose.HEAD_OFFSET -> stringResource(R.string.pose_head_offset)
                    StateAnalyzer.HeadPose.HEAD_TURNED -> stringResource(R.string.pose_head_turned)
                    StateAnalyzer.HeadPose.POSTURE_DEVIATION -> stringResource(R.string.pose_posture_deviation)
                }
            }
            Text(
                text = "${stringResource(R.string.pose)}: ${localizedPoses.joinToString(", ")}",
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.8f)
            )
        }
        
        Spacer(modifier = Modifier.height(4.dp))
        
        // 帧计数
        Text(
            text = "${stringResource(R.string.frame_count)}: $frameCount",
            style = MaterialTheme.typography.bodySmall,
            color = Color.White.copy(alpha = 0.7f)
        )
        
        // 滑动窗帧数（如果是滑动窗模式）
        if (isSlidingWindowMode) {
            Text(
                text = "${stringResource(R.string.window_frames)}: $windowFrameCount",
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.7f)
            )
        }
        
        // 校准状态
        val calibrationText = if (isCalibrated) 
            stringResource(R.string.calibrated) 
        else 
            stringResource(R.string.not_calibrated)
        
        Text(
            text = "${stringResource(R.string.calibration_status)}: $calibrationText",
            style = MaterialTheme.typography.bodySmall,
            color = if (isCalibrated) Normal else SlightlyTired
        )
        
        // 旋转角度
        Text(
            text = "${stringResource(R.string.rotation)}: ${manualRotation}°",
            style = MaterialTheme.typography.bodySmall,
            color = Color.White.copy(alpha = 0.7f)
        )
    }
}
