"""
@writer: zhangheng
Pydantic models for API request and response validation
"""
from pydantic import BaseModel


class FrameRequest(BaseModel):
    """Request model for frame analysis endpoints"""
    frame_base64: str


class APIResponse(BaseModel):
    """Standard API response model"""
    code: int
    msg: str
    data: dict | None = None


class HealthCheckData(BaseModel):
    """Health check response data"""
    analyzer_ready: bool
    device: str
    model_loaded: bool
    calibrated: bool


class CalibrationData(BaseModel):
    """Calibration response data"""
    calibrated: bool