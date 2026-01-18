"""
@writer: zhangheng
Main application entry point for YOLOv8 Driver State Detection System
"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import HOST, PORT
from analyzer import YOLODriverStateAnalyzer
from routes import router
from models import FrameRequest


# Global analyzer instance
analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ lifespan event handler for startup and shutdown """
    global analyzer

    # Startup
    print("\n" + "="*60)
    print("Initializing YOLOv8 Driver State Analyzer...")
    print("="*60)

    try:
        analyzer = YOLODriverStateAnalyzer(use_gpu=True)
        print("\n" + "="*60)
        print("YOLOv8 Driver State Analyzer Initialized Successfully!")
        print(f"Model Load Status: {'Success' if analyzer.yolo_model else 'Failed'}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\nAnalyzer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        analyzer = None

    yield

    # Shutdown
    print("\nShutting down application...")


# Create FastAPI application
app = FastAPI(
    title="YOLOv8 Driver State Detection API",
    description="Real-time driver fatigue detection using YOLOv8-Pose model",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["*"]
)


# Include routes with dependency injection
@app.post("/analyze_driver")
def analyze_driver_route(request: FrameRequest):
    from routes import analyze_driver
    return analyze_driver(request, analyzer)


@app.post("/analyze_driver_silence")
def analyze_driver_silence_route(request: FrameRequest):
    from routes import analyze_driver_silence
    return analyze_driver_silence(request, analyzer)


@app.post("/calibrate")
def calibrate_route(request: FrameRequest):
    from routes import calibrate
    return calibrate(request, analyzer)


@app.get("/health")
def health_check_route():
    from routes import health_check
    return health_check(analyzer)


@app.post("/reset")
def reset_counter_route():
    from routes import reset_counter
    return reset_counter(analyzer)


if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("YOLOv8 Driver State Detection Service")
    print("="*60)
    print(f"Service Address: http://{HOST}:{PORT}")
    print("Available Interfaces:")
    print("  POST /analyze_driver - Analyze driving state (with marked frame)")
    print("  POST /analyze_driver_silence - Analyze driving state (without marked frame)")
    print("  POST /calibrate - User calibration")
    print("  GET  /health - Service health check")
    print("  POST /reset - Reset statistical data")
    print("="*60)

    # Start service with uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
