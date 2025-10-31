from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import joblib
import json
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
from models.predictive_maintenance import PredictiveMaintenanceModel
from data.connector import DatabaseConnector, MaintenancePrediction
from utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AUTOMOTTO Predictive Maintenance API", version="1.0.0")

# Dependency
security = HTTPBearer()

config = load_config("config.json")
connector = DatabaseConnector(os.getenv("DATABASE_URL"))
model_path = "models/maintenance_model.pkl"

try:
    model = PredictiveMaintenanceModel(config)
    model.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class MaintenanceRequest(BaseModel):
    vehicle_id: str
    days_horizon: int = 90

class MaintenanceResponse(BaseModel):
    vehicle_id: str
    predictions: List[Dict]
    risk_factors: List[Dict]
    recommendations: List[Dict]
    overall_health_score: float
    next_service_date: str
    urgency_level: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": getattr(model, 'model_version', 'unknown'),
        "database_connected": connector.engine.has_connection(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/maintenance")
async def predict_maintenance(request: MaintenanceRequest, 
                            credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Predict maintenance needs for a vehicle
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Verify user owns vehicle
        with connector.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id FROM vehicles WHERE id = :vid AND user_id = :uid"),
                {"vid": request.vehicle_id, "uid": credentials.credentials}
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=403, detail="Unauthorized vehicle access")
        
        # Get vehicle data
        telemetry = connector.get_vehicle_telemetry_data(request.vehicle_id, 90)
        maintenance = connector.get_maintenance_history(request.vehicle_id, 2)
        health = connector.get_health_snapshots(request.vehicle_id, 180)
        
        if telemetry.empty:
            return {
                "error": "Insufficient data for prediction",
                "vehicle_id": request.vehicle_id,
                "recommendations": [
                    {
                        "action": "Schedule regular maintenance",
                        "priority": "routine",
                        "urgency": "low"
                    }
                ]
            }
        
        # Feature engineering
        engineer = FeatureEngineer(config)
        features = engineer.create_features(telemetry, maintenance, health)
        X = engineer.select_features(features)
        
        # Make predictions
        urgency_predictions = model.predict_maintenance_urgency(X.iloc[[-1]])[0]
        days_prediction = model.predict_days_until_failure(X.iloc[[-1]])[0]
        risk_factors = model.get_risk_factors(X.iloc[[-1]])
        recommendations = model.generate_maintenance_recommendations(
            [urgency_predictions], risk_factors
        )
        
        # Overall health summary
        health_summary = connector.get_vehicle_risk_score(request.vehicle_id)
        
        # Format response
        urgency_map = {
            "Critical - Immediate Attention": "critical",
            "Warning - Schedule Soon": "warning",
            "Routine Maintenance": "routine",
            "No Action Needed": "healthy"
        }
        
        return MaintenanceResponse(
            vehicle_id=request.vehicle_id,
            predictions=[
                {
                    "component": "Overall System",
                    "current_status": urgency_map[urgency_predictions],
                    "days_until_needed": int(days_prediction),
                    "confidence": 0.85,  # Placeholder
                    "priority": 1 if "Critical" in urgency_predictions else 2
                }
            ],
            risk_factors=risk_factors[:5],
            recommendations=recommendations[0],
            overall_health_score=health_summary.overall_health_score,
            next_service_date=health_summary.next_service_date,
            urgency_level=urgency_map[urgency_predictions]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vehicles/{vehicle_id}/risk-assessment")
async def get_risk_assessment(vehicle_id: str, 
                           credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get comprehensive risk assessment for vehicle"""
    try:
        # Verify access
        with connector.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id FROM vehicles WHERE id = :vid AND user_id = :uid"),
                {"vid": vehicle_id, "uid": credentials.credentials}
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Get recent data
        health_summary = connector.get_vehicle_risk_score(vehicle_id)
        
        return {
            "vehicle_id": vehicle_id,
            "health_score": round(health_summary.overall_health_score, 2),
            "risk_category": "low" if health_summary.overall_health_score > 80 else "medium",
            "next_action": health_summary.next_service_date,
            "risk_factors": health_summary.risk_factors[:5],
            "last_updated": health_summary.last_prediction
        }
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
