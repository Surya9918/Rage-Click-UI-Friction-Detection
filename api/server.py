import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FeatureExtractor
from src.inference import predict
from src.utils import get_logger, load_config

logger = get_logger("api.server")
config = load_config(PROJECT_ROOT / "config.yaml")

app = FastAPI(
    title="Rage-Click & UI Friction Detection API",
    description="Inference server using a Feed-Forward Neural Network (FFNN) to predict real-time UI friction and rage clicks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_extractor: Optional[FeatureExtractor] = None

@app.on_event("startup")
def startup_event():
    global feature_extractor
    feature_extractor = FeatureExtractor(config)
    logger.info("Server startup initialization completed.")

class FrictionFeatures(BaseModel):
    click_frequency: float = Field(..., description="Clicks per second over window duration", ge=0.0)
    rapid_fire_clicks: float = Field(..., description="Consecutive clicks on same element within 300ms", ge=0.0)
    maximum_cursor_velocity: float = Field(..., description="Max cursor speed in px/s", ge=0.0)
    erratic_direction_changes: float = Field(..., description="Movement angle changes > 90 degrees", ge=0.0)
    scroll_thrashing: float = Field(..., description="Total reversed vertical scroll distance in px", ge=0.0)

class PredictionResponse(BaseModel):
    friction_probability: float
    friction_detected: bool
    feature_importance: Optional[Dict[str, float]] = None

class RawTelemetryEvent(BaseModel):
    timestamp: float
    event_type: str
    x_coordinate: float = 0.0
    y_coordinate: float = 0.0
    target_element: str = ""
    scroll_y: float = 0.0

class WindowTelemetryPayload(BaseModel):
    events: List[RawTelemetryEvent]
    window_duration_seconds: Optional[float] = 5.0

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_type": "Feed Forward Neural Network (Dense 16 -> ReLU -> Dense 8 -> ReLU -> Dense 1)",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_friction(features: FrictionFeatures):
    try:
        feat_dict = features.dict()
        result = predict(feat_dict, models_dir=PROJECT_ROOT / "models")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/telemetry/window", response_model=PredictionResponse)
def process_telemetry_window(payload: WindowTelemetryPayload):
    global feature_extractor
    raw_event_dicts = [e.dict() for e in payload.events]
    extracted_features = feature_extractor.extract_features(
        raw_event_dicts, window_duration=payload.window_duration_seconds
    )
    result = predict(extracted_features, models_dir=PROJECT_ROOT / "models")
    return PredictionResponse(**result)

demo_dir = PROJECT_ROOT / "demo"
if demo_dir.exists():
    app.mount("/static", StaticFiles(directory=str(demo_dir)), name="static")

@app.get("/")
def serve_index():
    index_file = demo_dir / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "demo/index.html not found on server"},
        )
    return FileResponse(index_file)
